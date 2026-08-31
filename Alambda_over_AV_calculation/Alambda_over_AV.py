import numpy as np

from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D

from dust_extinction.parameter_averages import CCM89, O94, F99, G16


# --------------------------------------------------------------------
# 1. PHOENIX HiRes SED loader: wavelength from WAVE_*.fits, flux from SED
# --------------------------------------------------------------------
def load_phoenix_hires_sed(wave_fits_path, sed_fits_path):
    """
    Load PHOENIX HiRes spectrum and wavelength grid.

    Parameters
    ----------
    wave_fits_path : str
        Path to WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
        (wavelength array; usually in Angstrom).
    sed_fits_path : str
        Path to a single PHOENIX HiRes SED, e.g.
        lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits

    Returns
    -------
    wave : Quantity
        Wavelength array (Angstrom).
    flux : Quantity
        F_lambda in erg/s/cm^2/Angstrom.
    """

    # Load wavelength grid
    with fits.open(wave_fits_path) as hw:
        wave_raw = hw[0].data  # Typically in Angstrom for PHOENIX HiRes
    wave = wave_raw * u.AA

    # Load flux (surface F_lambda) with unit erg/s/cm^2/cm (from header BUNIT)
    with fits.open(sed_fits_path) as hs:
        flux_cm = hs[0].data  # shape matches wavelength grid

    # Convert from per cm to per Angstrom:
    # 1 cm = 1e8 Angstrom -> F_lambda[per A] = F_lambda[per cm] * dλ_cm/dλ_A = F_cm * 1e-8
    flux_A = flux_cm * 1e-8
    flux = flux_A * (u.erg / (u.s * u.cm**2 * u.AA))

    return wave, flux


def make_synphot_source_spectrum(wave, flux):
    """
    Build a synphot SourceSpectrum from a PHOENIX SED.

    Parameters
    ----------
    wave : Quantity [Angstrom]
    flux : Quantity [erg/s/cm^2/Angstrom]
    """
    # Empirical1D wants dimensionless x internally; synphot will carry wave_unit.
    model = Empirical1D(points=wave.to(u.AA).value,
                        lookup_table=flux.to(u.erg / (u.s * u.cm**2 * u.AA)).value)

    src = SourceSpectrum(model, wave_unit=u.AA)
    return src


# --------------------------------------------------------------------
# 2. Roman effective area -> normalized throughput bandpasses
# --------------------------------------------------------------------
def load_roman_bandpasses(csv_path, band_names=("F062", "F087", "F146", "F213")):
    """
    Load Roman effective area from a plain CSV (no ECSV/YAML header)
    and construct synphot SpectralElement bandpasses normalized to
    max throughput = 1.

    Parameters
    ----------
    csv_path : str
        Path to Roman_effarea_SCA11_plain.csv.
        Column 'Wave' in microns; columns per filter in m^2.
    band_names : iterable of str
        Filter columns to extract.

    Returns
    -------
    bandpasses : dict
        {band_name: SpectralElement}
    """
    tab = Table.read(csv_path, format="ascii.csv")

    # Wave in microns -> Angstrom
    wave_micron = np.array(tab["Wave"], dtype=float)
    wave = (wave_micron * 1e4) * u.AA  # 1 micron = 1e4 Å

    bandpasses = {}

    for band in band_names:
        if band not in tab.colnames:
            raise ValueError(f"Band {band} not in columns: {tab.colnames}")

        area = np.array(tab[band], dtype=float)  # m^2

        # Use only non-zero entries
        mask = area > 0.0
        if not np.any(mask):
            raise ValueError(f"No non-zero area values for band {band}")

        wave_band = wave[mask]
        area_band = area[mask]

        # Normalize to max=1 for throughput shape
        throughput = (area_band / np.nanmax(area_band)).astype(float)

        # Empirical1D expects dimensionless x; SpectralElement carries wave_unit.
        model = Empirical1D(points=wave_band.to(u.AA).value,
                            lookup_table=throughput)
        bp = SpectralElement(model, wave_unit=u.AA)
        bandpasses[band] = bp

    return bandpasses


# --------------------------------------------------------------------
# 3. Effective wavelength and photometric width via synphot
# --------------------------------------------------------------------
# def compute_efflam_and_width(source, bandpasses):
#     """
#     Compute effective wavelength and photometric width for each band.

#     Parameters
#     ----------
#     source : SourceSpectrum
#         Synphot source SED.
#     bandpasses : dict
#         {band_name: SpectralElement}

#     Returns
#     -------
#     results : dict
#         {band_name: {"efflam": efflam, "photbw": width}}
#         where efflam and photbw are astropy Quantities.
#     """
#     results = {}
#     for name, bp in bandpasses.items():
#         obs = Observation(source, bp, force="extrap")
#         efflam = obs.effective_wavelength()
# #         width = obs.photbw()
# #         results[name] = {"efflam": efflam, "photbw": width}
#         results[name] = {"efflam": efflam}
#     return results


def compute_efflam_and_width(source, bandpasses):
    """
    Compute effective wavelength and an approximate photometric width
    for each band by direct integration of source(λ) × throughput(λ).

    Parameters
    ----------
    source : SourceSpectrum
        Synphot source SED.
    bandpasses : dict
        {band_name: SpectralElement}

    Returns
    -------
    results : dict
        {band_name: {"efflam": efflam, "photbw": width}}
        where efflam and photbw are astropy Quantities in Angstrom.
    """
    results = {}

    for name, bp in bandpasses.items():
        # Wavelength grid for this bandpass
        lam = bp.waveset.to(u.AA)              # Quantity [Å]
        lam_val = lam.value                    # plain float array

        # Throughput curve (dimensionless)
        thr = bp(lam)                          # synphot samples SpectralElement
        thr_val = np.array(thr, dtype=float)

        # Source flux at these wavelengths
        f = source(lam)                        # Quantity, e.g. erg/s/cm^2/Å
        f_val = f.value                        # we only need relative weighting

        # Weight function for band-averaged quantities
        w = lam_val * thr_val * f_val

        # Guard against degenerate band (should not happen for Roman filters)
        if np.all(w <= 0):
            raise RuntimeError(f"No overlap or zero throughput for band {name}")

        # Effective wavelength:
        #   λ_eff = ∫ λ w(λ) dλ / ∫ w(λ) dλ
        num = np.trapz(lam_val * w, lam_val)
        den = np.trapz(w, lam_val)
        lam_eff = (num / den) * u.AA

        # A simple "width": weighted standard deviation of λ
        #   σ_λ^2 = ∫ (λ - λ_eff)^2 w(λ) dλ / ∫ w(λ) dλ
        var = np.trapz((lam_val - lam_eff.value)**2 * w, lam_val) / den
        width = np.sqrt(var) * u.AA

        results[name] = {"efflam": lam_eff, "photbw": width}

    return results


# --------------------------------------------------------------------
# 4. Band-averaged A_band/A_V using dust_extinction + numeric integration
# --------------------------------------------------------------------
def get_extinction_model(law_name="F99", Rv=3.1):
    """
    Return a dust_extinction extinction model for a given law and Rv.

    law_name : {"F99", "CCM89", "O94", "G16"}
    """
    law_name = law_name.upper()
    if law_name == "F99":
        return F99(Rv=Rv)
    elif law_name == "CCM89":
        return CCM89(Rv=Rv)
    elif law_name == "O94":
        return O94(Rv=Rv)
    elif law_name == "G16":
        return G16(Rv=Rv)
    else:
        raise ValueError(f"Unknown extinction law: {law_name}")


def compute_A_band_over_Av_from_sed(
    wave_sed,
    flux_sed,
    bandpasses,
    Rv=3.1,
    law_name="F99",
    Av=1.0,
):
    """
    Compute band-averaged A_band/Av for each band by integrating
    PHOENIX SED x Roman throughput x extinction curve.

    Parameters
    ----------
    wave_sed : Quantity
        PHOENIX wavelength array (Angstrom).
    flux_sed : Quantity
        PHOENIX F_lambda (erg/s/cm^2/Angstrom).
    bandpasses : dict
        {band_name: SpectralElement}
    Rv : float
        Total-to-selective extinction, A_V/E(B-V).
    law_name : str
        Extinction law name ("F99", "CCM89", "G16", ...).
    Av : float
        Value of A_V (mag) to use for the band integration.  Since
        A_band/Av is independent of the *value* of Av, Av=1 is fine.

    Returns
    -------
    A_band_over_Av : dict
        {band_name: A_band/Av (float)}
    """
    ext_model = get_extinction_model(law_name=law_name, Rv=Rv)

    # Strip units for interpolation; keep scalars for integrals
    lam_sed = wave_sed.to(u.AA).value
    flux_sed_val = flux_sed.to(u.erg / (u.s * u.cm**2 * u.AA)).value

    A_band_over_Av = {}

    for name, bp in bandpasses.items():
        lam_band = bp.waveset.to(u.AA).value
        thr_band = bp(bp.waveset)  # dimensionless throughput array

        # Interpolate SED onto bandpass grid
        F_band = np.interp(lam_band, lam_sed, flux_sed_val)

        # Unextincted band flux (arbitrary normalization)
        F0 = np.trapz(F_band * lam_band * thr_band, lam_band)

        #  ext_model(lam) returns A(λ)/A(V)
        A_lambda_over_AV = ext_model(lam_band * u.AA)

        # For a given Av (mag), total extinction in band is A(λ) = Av * axav
        A_lambda = Av * A_lambda_over_AV  # mag

        # Transmission factor at each λ
        trans = 10.0 ** (-0.4 * A_lambda)

        # Extincted band flux
        F_ext = np.trapz(F_band * lam_band * thr_band * trans, lam_band)

        # Band extinction
        A_band = -2.5 * np.log10(F_ext / F0)  # mag

        # For Av input, A_band/Av = A_band (if Av=1). Explicitly:
        A_band_over_Av[name] = A_band / Av

    return A_band_over_Av



# --------------------------------------------------------------------
# 5. Example main: compare G2V vs ~0.5 Msun M dwarf
# --------------------------------------------------------------------
if __name__ == "__main__":
    # ---- user paths (EDIT THESE) ----
    data_dir = "./"  # change this

    WAVE_FILE  = data_dir + "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

    # G2V-like SED (Teff ~ 5800 K, log g = 4.5, [Fe/H] = 0)
    PHX_G2V    = data_dir + "lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    # ~0.5 Msun M dwarf SED (you choose an appropriate PHOENIX file, e.g. Teff ~ 3700–3900 K)
    PHX_MDWARF = data_dir + "lte03800-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    # Roman effective area for one SCA (e.g. SCA01)
    ROMAN_ECSV = data_dir + "Roman_effarea_v8_SCA11_20240301.ecsv"

    # Roman bands to inspect
    BANDS = ("F062", "F087", "F106", "F146", "F213")

    # ---- load PHOENIX SEDs ----
    wave_g2v, flux_g2v = load_phoenix_hires_sed(WAVE_FILE, PHX_G2V)
    wave_md, flux_md = load_phoenix_hires_sed(WAVE_FILE, PHX_MDWARF)

    src_g2v = make_synphot_source_spectrum(wave_g2v, flux_g2v)
    src_md  = make_synphot_source_spectrum(wave_md, flux_md)

    # ---- load Roman bandpasses ----
    bandpasses = load_roman_bandpasses(ROMAN_ECSV, band_names=BANDS)

    # ---- effective wavelength & width (for M dwarf, as representative) ----
    eff_results_g2v = compute_efflam_and_width(src_g2v, bandpasses)
    print("Effective wavelength and photometric width (G2V SED):")
    for band in BANDS:
        efflam = eff_results_g2v[band]["efflam"]
        bw = eff_results_g2v[band]["photbw"]
        print(f"  {band}: efflam = {efflam:.1f}, photbw = {bw:.1f}")
#         print(f"  {band}: efflam = {efflam:.1f}")
        
    eff_results_md = compute_efflam_and_width(src_md, bandpasses)
    print("Effective wavelength and photometric width (M dwarf SED):")
    for band in BANDS:
        efflam = eff_results_md[band]["efflam"]
        bw = eff_results_md[band]["photbw"]
        print(f"  {band}: efflam = {efflam:.1f}, photbw = {bw:.1f}")
#         print(f"  {band}: efflam = {efflam:.1f}")

    # ---- A_band/Av for G2V vs M dwarf, for a given Rv and law ----
    Rv = 2.5
    law_name = "CCM89" # "O94" # "F99" # 

    AoverAv_g2v = compute_A_band_over_Av_from_sed(
        wave_g2v, flux_g2v, bandpasses, Rv=Rv, law_name=law_name, Av=1.0
    )
    AoverAv_md = compute_A_band_over_Av_from_sed(
        wave_md, flux_md, bandpasses, Rv=Rv, law_name=law_name, Av=1.0
    )

    print(f"\nA_band/A_V for law={law_name}, R_V={Rv} (G2V vs M dwarf):")
    for band in BANDS:
        print(
            f"  {band}: G2V = {AoverAv_g2v[band]:.4f}, "
            f"M-dwarf = {AoverAv_md[band]:.4f}, "
#             f"Δ = {AoverAv_md[band] - AoverAv_g2v[band]:+.4f}"
        )
    print('M-dwarf', f"{Rv}, {AoverAv_md['F062']:.4f}, {AoverAv_md['F087']:.4f}, {AoverAv_md['F213']:.4f}")
#     print('G2V', f"{Rv}, {AoverAv_g2v['F062']:.4f}, {AoverAv_g2v['F087']:.4f}, {AoverAv_g2v['F213']:.4f}")
