# ============================================================
# MC PHYSICAL FORCE MODEL
# simulates N FRIs based on observed distributions
# works out force ratios:
#
#   R1 = F_ram / F_buoy
#
#   R2 = F_ram /
#        (F_buoy + F_expansion + F_momentum)
#
# and determines morphology based on these
# ============================================================


# import modules

from scipy.ndimage import gaussian_filter1d
from colossus.halo import profile_nfw
import dpmhalo
from dpmhalo import myutils, modelparams
from synchro import SynchSource
import os
import sys
import glob
import math
import time
import warnings
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ks_2samp, norm, gaussian_kde, pearsonr
from scipy.ndimage import gaussian_filter1d

from astropy.io import fits
import astropy.io.fits as pyfits
from astropy import table
from astropy.table import Table, join
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.visualization import (imshow_norm,AsymmetricPercentileInterval, SqrtStretch, MinMaxInterval)
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo

warnings.filterwarnings("ignore")

print("modules loaded")

# saving settings

plot_dir = "MC_model_comparison_size"
os.makedirs(plot_dir, exist_ok=True)
print("will save results to", plot_dir)

path = (
    "/beegfs/lofar/lclews/FRI_evolution/subclassification/inspection_results/buoyancy/"
)

# Physical constants

G = 6.67430e-11
M_solar = 1.98847e30
m_p = 1.67262192e-27
kpc_m = 3.085677581e19
h = 0.6766
gamma = 5.0 / 3.0
conc = 4.0
c = const.c.value
v_jet = 0.4 * c
L_to_P = 1e38 / 3e27  # Luminosity to jet power conversion Eq. 22 Hardcastle+2018
LOFAR_BEAM_ARCSEC = 6.0
FS = 13
h = 0.6766

# LOAD DATA


def get_label(row, keyword):

    morph_label = row["Morph"]

    # Treat empty/null as missing
    if morph_label not in (None, "", " ", "nan"):

        source_label = morph_label

    else:

        source_label = row["subclass_JC"]

    if isinstance(source_label, str) and keyword in source_label.lower():

        return True

    return False


complete = Table.read(
    "/beegfs/lofar/lclews/FRI_evolution/subclassification/inspection_results/new_env_final_classes_with_forces_and_pressures_model3.fits"
)


complete["tailed"] = [get_label(row, "tailed") for row in complete]
complete["lobed"] = [get_label(row, "lobed") for row in complete]
complete["transitional"] = [get_label(row, "transitional") for row in complete]


print("data loaded")

for segfrac in [1 / 6, 1 / 2, 5 / 6]:

    Rvals = []

    for row in complete:

        M500 = row["M500_bcg"]

        if not np.isfinite(M500):
            continue

        z = row["z_best"]

        M200 = myutils.lM500c_to_lM200c(np.log10(M500 / h), z)

        R200 = myutils.R200c_from_lM200c(M200, z)

        r_kpc = segfrac * row["Size"]

        Rvals.append(r_kpc / R200)

    print(segfrac, np.nanmedian(Rvals))

obs_tail_frac = np.sum(complete["tailed"]) / (
    np.sum(complete["tailed"])
    + np.sum(complete["lobed"])
    + np.sum(complete["transitional"])
)

print("Observed tail fraction =", obs_tail_frac)


def calculate_environment(M500_raw, z, r_kpc):
    """
    Function to calculate environmental quantities rho_ICM, ne, P (ICM), v_lobe, P_ram, g, R200, R_over_R200

    Input Parameters:
    ----------
    M500_raw : float
        M500 in solar masses.

    z : float
        Redshift.

    r_kpc : float
        Radius from cluster centre in kpc.

    Returns:
    -------
    dict (with  rho_ICM, ne, P (ICM), v_lobe, P_ram, g, R200, R_over_R200) or None
    """
    if not np.isfinite(M500_raw) or M500_raw <= 0:

        return None

    if not np.isfinite(z) or z <= 0:

        return None

    if not np.isfinite(r_kpc) or r_kpc <= 0:

        return None

    # Convert M500 to units needed by Colosuss

    M500 = M500_raw / h
    M200 = myutils.lM500c_to_lM200c(np.log10(M500), z)

    R200 = myutils.R200c_from_lM200c(M200, z)

    # DPM  halo modelling

    modelID = "Model3"

    params = modelparams.returnmodelparams(modelID)
    model = dpmhalo.ModelMFlexGNFW(10**M200, z, *params)

    R_over_R200 = (
        r_kpc / R200
    )  # compare distances at consistent scales in clusters of different sizes

    P = model.calcP(R_over_R200, 10**M200, z) * 1.38e-17  # ICM pressure
    ne = model.calcne(R_over_R200, 10**M200, z) * 1e6  # Electron density
    rho_ICM = 1.14 * ne * m_p  # ne to  ntot
    v_lobe = np.sqrt(gamma * P / rho_ICM)  # lobe speed (approx as sound speed)
    P_ram = rho_ICM * v_lobe**2

    prof = profile_nfw.NFWProfile(M=M500, c=conc, z=z, mdef="vir")
    r_nfw = max(r_kpc * h, 0.1)
    m_enc = prof.enclosedMass(np.array([r_nfw]))[0]  # enclosed halo mass
    m_enc_kg = m_enc * M_solar / h  # convert to kg

    r_m = r_kpc * kpc_m
    g = G * m_enc_kg / r_m**2

    return { "rho": rho_ICM,  "ne": ne,    "P": P,   "v_lobe": v_lobe,  "P_ram": P_ram,"g": g,  "R200": R200,"R_over_R200": R_over_R200,}


def calculate_expansion_pressure(
    flux_mJy, seg_number, z, volume_m3, flux_fractions, kappa
):
    """
    Function to calculate internal pressure in Pa
    """

    if not np.isfinite(flux_mJy) or flux_mJy <= 0:

        return np.nan

    if not np.isfinite(z) or z <= 0:

        return np.nan

    if not np.isfinite(volume_m3) or volume_m3 <= 0:

        return np.nan

    flux_Jy_total = flux_mJy / 1000  # mjy to jy
    flux_Jy_segment = flux_Jy_total * flux_fractions[seg_number]

    rsph_m = (3.0 * volume_m3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    try:

        synch_out = SynchSource(
            type="sphere",
            rsph=rsph_m,
            gmin=10,
            gmax=100000,
            injection=2.0,
            cosmology=cosmo,
            z=z,
        )
        synch_out.normalize(
            144e6,
            flux_Jy_segment,
            method="equipartition",
            brange=(1e-11, 1e-6),
            zeta=(1.0 + kappa),
        )

        P_int = synch_out.total_energy_density / 3.0

        return P_int

    except Exception:

        return np.nan


# initialise MC modelling


def sample_independent_kdes(df, N):

    print("making MC sample")

    # Sample L144 and Total_flux together

    radio_df = df[["L_144", "Total_flux"]].dropna()
    rng = np.random.default_rng()
    idx = rng.integers(0, len(radio_df), size=N)
    L144_mc = radio_df["L_144"].values[idx]
    flux_mc = radio_df["Total_flux"].values[idx]

    # sample others independently
    kde_logL500 = gaussian_kde(np.log10(df["L500_bgsub_bcg"]))
    kde_logSize = gaussian_kde(np.log10(df["Size"]))
    kde_logM500 = gaussian_kde(np.log10(df["M500_bcg"]))
    kde_logR500 = gaussian_kde(np.log10(df["R500_bcg"]))
    kde_z = gaussian_kde(df["z_best"])

    logL500 = kde_logL500.resample(N).ravel()
    logSize = kde_logSize.resample(N).ravel()
    logM500 = kde_logM500.resample(N).ravel()
    logR500 = kde_logR500.resample(N).ravel()

    return {
        "L144": L144_mc,
        "total_flux": flux_mc,
        "L500": 10**logL500,
        "size": 10**logSize,
        "M500": 10**logM500,
        "R500": 10**logR500,
        "z": kde_z.resample(N).ravel(),
    }


def plot_ratio_diagnostics(ratio, obs_ratio, ratio_label, model_name, plot_dir, suffix):
    """
    Produce histogram + KS diagnostic for one force ratio.
    """

    obs_ratio = np.asarray(obs_ratio)

    ratio = np.asarray(ratio)

    obs_ratio = obs_ratio[np.isfinite(obs_ratio)]

    ratio = ratio[np.isfinite(ratio)]

    obs_med = np.nanmedian(obs_ratio)

    mc_med = np.nanmedian(ratio)

    ks_stat, ks_p = ks_2samp(obs_ratio, ratio)

    pdf_name = f"{plot_dir}/" f"{model_name}_" f"{suffix}_diagnostics.pdf"

    png_name = f"{plot_dir}/" f"{model_name}_" f"{suffix}_histogram.png"

    pdf = PdfPages(pdf_name)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(obs_ratio, bins=30, density=True, alpha=0.5, color="k", label="Observed")

    ax.hist(
        ratio,
        bins=30,
        density=True,
        alpha=0.5,
        color="cornflowerblue",
        label="Physical MC",
    )

    ax.axvline(obs_med, color="k", ls="--", lw=3, label=(f"Obs median={obs_med:.2f}"))

    ax.axvline(
        mc_med, color="cornflowerblue", ls="--", lw=3, label=(f"MC median={mc_med:.2f}")
    )

    ax.set_xlabel(ratio_label)

    ax.set_ylabel("Density")

    ax.legend()

    ax.set_title(
        f"{model_name}\n"
        f"{ratio_label}\n"
        f"Obs sigma={np.std(obs_ratio):.2f}  "
        f"MC sigma={np.std(ratio):.2f}\n"
        f"KS={ks_stat:.3f}, "
        f"p={ks_p:.3e}"
    )

    plt.tight_layout()

    pdf.savefig(fig)

    fig.savefig(png_name, dpi=300, bbox_inches="tight")

    plt.close(fig)

    pdf.close()

    return {
        "median_obs": obs_med,
        "median_mc": mc_med,
        "ks": ks_stat,
        "ks_p": ks_p,
        "obs_std": np.std(obs_ratio),
        "mc_std": np.std(ratio),
        "pdf": pdf_name,
        "png": png_name,
    }



def make_threshold_plot( ratio, conditions, obs_tail_frac, model_name, plot_dir, suffix, ratio_label):
    """
    Calculate tail fraction versus ratio threshold
    and save the threshold plot.
    """

    tail_fracs = np.array([np.mean(ratio > threshold) for threshold in conditions])

    best_idx = np.argmin(np.abs(tail_fracs - obs_tail_frac))

    best_threshold_log = conditions[best_idx]

    best_tail_frac = tail_fracs[best_idx]

    pdf_name = f"{plot_dir}/" f"{model_name}_" f"{suffix}_threshold.pdf"

    png_name = f"{plot_dir}/" f"{model_name}_" f"{suffix}_threshold.png"

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(conditions, tail_fracs, "-o", ms=3, color="cornflowerblue")

    ax.axhline(
        obs_tail_frac,
        color="r",
        ls="--",
        lw=2,
        label=(f"Observed tail fraction = " f"{obs_tail_frac:.3f}"),
    )

    ax.axvline(
        best_threshold_log,
        color="k",
        ls=":",
        lw=2,
        label=(f"Best threshold = " f"{best_threshold_log:.2f}"),
    )

    ax.set_xlabel(ratio_label + " threshold")

    ax.set_ylabel("Tail fraction")

    ax.legend()

    ax.set_title(f"{model_name}\n" f"{ratio_label}")

    plt.tight_layout()

    fig.savefig(png_name, dpi=300, bbox_inches="tight")

    pdf = PdfPages(pdf_name)

    pdf.savefig(fig)

    pdf.close()

    plt.close(fig)

    return {
        "tail_fracs": tail_fracs,
        "best_threshold_log": best_threshold_log,
        "best_threshold_linear": 10**best_threshold_log,
        "best_tail_frac": best_tail_frac,
        "png": png_name,
        "pdf": pdf_name,
    }




def make_morphology_plots(mc_ratio, L144_valid, size_valid,  conditions, model_name, plot_dir,suffix, Nplot):
    """
    Make pages showing the simulated morphology split
    for every threshold.
    """

    Nplot_actual = min(Nplot, len(mc_ratio))

    if Nplot_actual == 0:

        return None

    rng = np.random.default_rng(12345)

    plot_idx = rng.choice(len(mc_ratio), Nplot_actual, replace=False)

    L144_plot = np.log10(L144_valid[plot_idx])

    size_plot = np.log10(size_valid[plot_idx])

    ratio_plot = mc_ratio[plot_idx]

    pdf_name = f"{plot_dir}/" f"{model_name}_" f"{suffix}_morphology.pdf"

    pdf = PdfPages(pdf_name)

    ncols = 3

    nrows = 2

    n_per_page = 6

    for start in range(0, len(conditions), n_per_page):

        page_conditions = conditions[start : start + n_per_page]

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(12, 8), sharex=True, sharey=True
        )

        axes = axes.ravel()

        for ax, threshold in zip(axes, page_conditions):

            lobed = ratio_plot <= threshold

            tailed = ratio_plot > threshold

            f_tail = np.mean(mc_ratio > threshold)

            ax.scatter(L144_plot[lobed], size_plot[lobed], s=10, alpha=0.5, color="k")

            ax.scatter(
                L144_plot[tailed], size_plot[tailed], s=10, alpha=0.5, color="indianred"
            )

            ax.set_title(f"T={threshold:.2f}\n" f"f_tail={f_tail:.2f}", fontsize=10)

        for ax in axes[len(page_conditions) :]:

            fig.delaxes(ax)

        # Only label axes that actually exist

        existing_axes = axes[: len(page_conditions)]

        if len(existing_axes) > 0:

            existing_axes[0].set_ylabel(r"log$_{10}$(Size)")

        if len(existing_axes) > 3:

            existing_axes[3].set_ylabel(r"log$_{10}$(Size)")

        for ax in existing_axes:

            ax.set_xlabel(r"log$_{10}$(L$_{144}$)")

        fig.suptitle(f"{model_name}: {suffix}", fontsize=14)

        plt.tight_layout()

        pdf.savefig(fig)

        plt.close(fig)

    pdf.close()

    return pdf_name


def run_physical_mc_model(mc, ram_col, buoy_col, model_name,=aspect_ratio,  flux_fractions, alpha_radio=DEFAULT_ALPHA_RADIO, 
                          kappa=DEFAULT_KAPPA, conditions_ram_buoy=np.arange(-0.5, 1.5, 0.05),conditions_total=np.arange(-0.5, 1.5, 0.05),
                          Nplot=10000, output_label=None,):
    """
    Run the  MC force model.

    F_ram = P_ram * cross-sectional area
    F_buoy = rho * g * volume
    F_expansion = P_int * cross-sectional area
    F_momentum = P_jet / (0.4*c)

    Ratio 1: log10(F_ram / F_buoy )


    Ratio 2:log10(
            F_ram / ( F_buoy + F_expansion + F_momentum  )  )



    aspect_ratio = length / width

    Each source half is divided into three equal sections:

        length = Size / 6
        -> width = length / aspect_ratio
    """

    if aspect_ratio <= 0:

        raise ValueError("aspect_ratio must be > 0")

    if output_label is None:
        output_label = model_name


    size_mc = np.asarray(mc["size"], dtype=float)

    M500_mc = np.asarray(mc["M500"], dtype=float)

    z_mc = np.asarray(mc["z"], dtype=float)

    L144_mc = np.asarray(mc["L144"], dtype=float)
    flux_mc = np.asarray(mc["total_flux"], dtype=float)

    jet_power_mc = L144_mc * L_to_P

 

    F_momentum_mc = jet_power_mc / (0.4 * const.c.value)


    N = len(size_mc)

    print("Number of MC sources =", N)

    print("Aspect ratio =", aspect_ratio)

    # SOURCE GEOMETRY
    

    # Three equal sections along one half-source.

    length_mc = size_mc / 6.0

    # aspect_ratio = length / width

    width_mc = length_mc / aspect_ratio

    width_m = width_mc * kpc_m

    length_m = length_mc * kpc_m

    radius_m = width_m / 2.0

    area_mc = np.pi * radius_m**2

    
    volume_mc = area_mc * length_m

    

    R200_mc = np.full(N, np.nan)

    for i in range(N):

        M500 = M500_mc[i] / h

        M200 = myutils.lM500c_to_lM200c(np.log10(M500), z_mc[i])

        R200_mc[i] = myutils.R200c_from_lM200c(M200, z_mc[i])
    r1_mc = (1 / 3) * 0.2 * R200_mc
    r2_mc = (2 / 3) * 0.2 * R200_mc
    r3_mc = (3 / 3) * 0.2 * R200_mc

    radii = [r1_mc, r2_mc, r3_mc]

    
    # setup output arrays
    
    F_ram_segments = []

    F_buoy_segments = []

    F_expansion_segments = []

    ram_fraction_segments = []
    buoy_fraction_segments = []
    exp_fraction_segments = []
    mom_fraction_segments = []

    rho_segments = []

    P_segments = []

    P_ram_segments = []

    P_int_segments = []

    cs_segments = []

    g_segments = []

    R200_segments = []
    R_over_R200_segments = []

    # precompute internal pressure to save time
    P_int_segments_precomputed = {}

    for seg_number in [1, 2, 3]:

        P_int = np.full(N, np.nan)

        for i in range(N):

            P_int[i] = calculate_expansion_pressure(
                flux_mJy=flux_mc[i],
                seg_number=seg_number,
                z=z_mc[i],
                volume_m3=volume_mc[i],
                flux_fractions=flux_fractions,
                kappa=kappa,
            )

        P_int_segments_precomputed[seg_number] = P_int
   
    for seg_number, r_mc in enumerate(radii, start=1):

        print()
        print(f"Calculating segment {seg_number}")

        
        # Force arrays
        F_ram = np.full(N, np.nan)

        F_buoy = np.full(N, np.nan)

        F_expansion = np.full(N, np.nan)


        # Environment arrays
       
        rho_arr = np.full(N, np.nan)
        R200_arr = np.full(N, np.nan)
        R_over_R200_arr = np.full(N, np.nan)

        P_arr = np.full(N, np.nan)

        P_ram_arr = np.full(N, np.nan)

        P_int_arr = np.full(N, np.nan)

        cs_arr = np.full(N, np.nan)

        g_arr = np.full(N, np.nan)
       
        # Loop through MC galaxies
        
        for i in range(N):

            env = calculate_environment(M500_raw=M500_mc[i], z=z_mc[i], r_kpc=r_mc[i])

            if env is None:

                continue

            rho = env["rho"]

            P = env["P"]

            R200 = env["R200"]

            R_over_R200 = env["R_over_R200"]

            v_lobe = env["v_lobe"]

            P_ram = env["P_ram"]

            g = env["g"]


            F_buoy[i] = rho * g * volume_mc[i]
     
            F_ram[i] = P_ram * area_mc[i]

            P_int = P_int_segments_precomputed[seg_number][i]

            if not np.isfinite(P_int):

                continue

            
            F_expansion[i] = P_int * area_mc[i]

            # Store environmental quantities
            

            rho_arr[i] = rho

            P_arr[i] = P

            P_ram_arr[i] = P_ram

            P_int_arr[i] = P_int

            cs_arr[i] = v_lobe

            g_arr[i] = g

            R200_arr[i] = R200

            R_over_R200_arr[i] = R_over_R200
        
        # Store segment
       
        F_ram_segments.append(F_ram)

        F_buoy_segments.append(F_buoy)

        F_expansion_segments.append(F_expansion)

        total_force = F_ram + F_buoy + F_expansion + F_momentum_mc

        ram_frac = F_ram / total_force
        buoy_frac = F_buoy / total_force
        exp_frac = F_expansion / total_force
        mom_frac = F_momentum_mc / total_force

        ram_fraction_segments.append(ram_frac)
        buoy_fraction_segments.append(buoy_frac)
        exp_fraction_segments.append(exp_frac)
        mom_fraction_segments.append(mom_frac)

        rho_segments.append(rho_arr)

        P_segments.append(P_arr)

        P_ram_segments.append(P_ram_arr)

        P_int_segments.append(P_int_arr)

        cs_segments.append(cs_arr)

        g_segments.append(g_arr)

        R200_segments.append(R200_arr)

        R_over_R200_segments.append(R_over_R200_arr)

        valid_seg = (
            np.isfinite(F_ram)
            & np.isfinite(F_buoy)
            & np.isfinite(F_expansion)
            & (F_ram > 0)
            & (F_buoy > 0)
            & (F_expansion >= 0)
        )

        print("Valid:", np.sum(valid_seg), "/", N)

  

    force_arrays = {
        "END": (F_ram_segments[2], F_buoy_segments[2], F_expansion_segments[2]),
        "POS1": (F_ram_segments[0], F_buoy_segments[0], F_expansion_segments[0]),
        "POS2": (F_ram_segments[1], F_buoy_segments[1], F_expansion_segments[1]),
        "POS3": (F_ram_segments[2], F_buoy_segments[2], F_expansion_segments[2]),
    }

    if model_name not in force_arrays:

        raise ValueError("model_name must be one of " "'POS1', 'POS2', 'POS3', 'END'")

    (Fram_model, Fbuoy_model, Fexp_model) = force_arrays[model_name]

    obs_segment_cols = {
        "POS1": (
            "F_ram_1",
            "F_buoy_1",
            "Pint1",
        ),
        "POS2": (
            "F_ram_2",
            "F_buoy_2",
            "Pint2",
        ),
        "POS3": (
            "F_ram_3",
            "F_buoy_3",
            "Pint3",
        ),
        "END": (
            "F_ram_3",
            "F_buoy_3",
            "Pint3",
        ),
    }

    obs_ram_col, obs_buoy_col, obs_pint_col = obs_segment_cols[model_name]

    obs_ram = np.asarray(complete[obs_ram_col], dtype=float)

    obs_buoy = np.asarray(complete[obs_buoy_col], dtype=float)

    obs_valid_1 = (
        np.isfinite(obs_ram) & np.isfinite(obs_buoy) & (obs_ram > 0) & (obs_buoy > 0)
    )

    obs_ratio_ram_buoy = np.full(len(complete), np.nan)

    obs_ratio_ram_buoy[obs_valid_1] = np.log10(
        obs_ram[obs_valid_1] / obs_buoy[obs_valid_1]
    )

    obs_ratio_ram_buoy_valid = obs_ratio_ram_buoy[np.isfinite(obs_ratio_ram_buoy)]

    
    if obs_pint_col not in complete.colnames:

        raise KeyError(f"{obs_pint_col} not found in catalogue.")

    if "L_144" not in complete.colnames:

        raise KeyError(
            "The observed catalogue does not contain "
            "'L_144', which is required to calculate "
            "observed F_momentum."
        )

    obs_pint = np.asarray(complete[obs_pint_col], dtype=float)
    obs_L144 = np.asarray(complete["L_144"], dtype=float)

    obs_jet_power = obs_L144 * L_to_P

    obs_area = np.asarray(complete["Cross_sectional_A (m2)"], dtype=float)

    obs_expansion = obs_pint * obs_area
    obs_F_momentum = obs_jet_power / (0.4 * const.c.value)

    obs_valid_2 = (
        np.isfinite(obs_ram)
        & np.isfinite(obs_buoy)
        & np.isfinite(obs_expansion)
        & np.isfinite(obs_F_momentum)
        & (obs_ram > 0)
        & (obs_buoy > 0)
        & (obs_expansion >= 0)
        & (obs_F_momentum > 0)
    )

    obs_ratio_ram_total = np.full(len(complete), np.nan)

    obs_ratio_ram_total[obs_valid_2] = np.log10(
        obs_ram[obs_valid_2]
        / (
            obs_buoy[obs_valid_2]
            + obs_expansion[obs_valid_2]
            + obs_F_momentum[obs_valid_2]
        )
    )

    obs_ratio_ram_total_valid = obs_ratio_ram_total[np.isfinite(obs_ratio_ram_total)]

    valid = (
        np.isfinite(Fram_model)
        & np.isfinite(Fbuoy_model)
        & np.isfinite(Fexp_model)
        & np.isfinite(F_momentum_mc)
        & (Fram_model > 0)
        & (Fbuoy_model > 0)
        & (Fexp_model >= 0)
        & (F_momentum_mc > 0)
    )

    
    print("Valid MC realisations =", np.sum(valid), "/", N)


    mc_ratio_ram_buoy = np.log10(Fram_model[valid] / Fbuoy_model[valid])


    mc_ratio_ram_total = np.log10(
        Fram_model[valid]
        / (Fbuoy_model[valid] + Fexp_model[valid] + F_momentum_mc[valid])
    )

    size_valid = size_mc[valid]


    obs_size = np.asarray(complete["Size"], dtype=float)

    obs_corr_mask_1 = np.isfinite(obs_size) & np.isfinite(obs_ratio_ram_buoy)

    obs_corr_mask_2 = np.isfinite(obs_size) & np.isfinite(obs_ratio_ram_total)

    if np.sum(obs_corr_mask_1) >= 3:

        r_obs_1, p_obs_1 = pearsonr(
            np.log10(obs_size[obs_corr_mask_1]), obs_ratio_ram_buoy[obs_corr_mask_1]
        )

    else:

        r_obs_1 = np.nan
        p_obs_1 = np.nan

    if np.sum(obs_corr_mask_2) >= 3:

        r_obs_2, p_obs_2 = pearsonr(
            np.log10(obs_size[obs_corr_mask_2]), obs_ratio_ram_total[obs_corr_mask_2]
        )

    else:

        r_obs_2 = np.nan
        p_obs_2 = np.nan

   

    if len(mc_ratio_ram_buoy) >= 3:

        r_mc_1, p_mc_1 = pearsonr(np.log10(size_valid), mc_ratio_ram_buoy)

    else:

        r_mc_1 = np.nan
        p_mc_1 = np.nan

    if len(mc_ratio_ram_total) >= 3:

        r_mc_2, p_mc_2 = pearsonr(np.log10(size_valid), mc_ratio_ram_total)

    else:

        r_mc_2 = np.nan
        p_mc_2 = np.nan

    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print("Aspect ratio =", aspect_ratio)

    print("Ratio 1 = log10(F_ram / F_buoy)")

    print(f"OBS corr(Size, ratio 1) = " f"{r_obs_1:.3f}")

    print(f"MC  corr(Size, ratio 1) = " f"{r_mc_1:.3f}")

    
    print("Ratio 2 = log10(" "F_ram / " "(F_buoy + F_expansion + F_momentum)" ")")

    print(f"OBS corr(Size, ratio 2) = " f"{r_obs_2:.3f}")

    print(f"MC  corr(Size, ratio 2) = " f"{r_mc_2:.3f}")


    print(
        "Observed lobed median ratio 1 =",
        np.nanmedian(obs_ratio_ram_buoy[np.asarray(complete["lobed"])]),
    )

    print(
        "Observed tailed median ratio 1 =",
        np.nanmedian(obs_ratio_ram_buoy[np.asarray(complete["tailed"])]),
    )


    print(
        "Observed lobed median ratio 2 =",
        np.nanmedian(obs_ratio_ram_total[np.asarray(complete["lobed"])]),
    )

    print(
        "Observed tailed median ratio 2 =",
        np.nanmedian(obs_ratio_ram_total[np.asarray(complete["tailed"])]),
    )



    print(
        "Observed lobed median size =",
        np.nanmedian(obs_size[np.asarray(complete["lobed"])]),
    )

    print(
        "Observed tailed median size =",
        np.nanmedian(obs_size[np.asarray(complete["tailed"])]),
    )

    print("Median MC F_ram =", np.nanmedian(Fram_model[valid]))

    print("Median MC F_buoy =", np.nanmedian(Fbuoy_model[valid]))

    print("Median MC F_expansion =", np.nanmedian(Fexp_model[valid]))

    print("Median MC F_momentum =", np.nanmedian(F_momentum_mc[valid]))

  

    obs_med_1 = np.nanmedian(obs_ratio_ram_buoy_valid)

    mc_med_1 = np.nanmedian(mc_ratio_ram_buoy)

    ks_stat_1, ks_p_1 = ks_2samp(obs_ratio_ram_buoy_valid, mc_ratio_ram_buoy)

    print("log10(F_ram / F_buoy)")
    
    print(f"Observed mean = " f"{np.mean(obs_ratio_ram_buoy_valid):.3f}")

    print(f"Observed std = " f"{np.std(obs_ratio_ram_buoy_valid):.3f}")

    print(f"Observed median = " f"{obs_med_1:.3f}")

    print(f"MC mean = " f"{np.mean(mc_ratio_ram_buoy):.3f}")

    print(f"MC std = " f"{np.std(mc_ratio_ram_buoy):.3f}")

    print(f"MC median = " f"{mc_med_1:.3f}")

    print(f"KS statistic = " f"{ks_stat_1:.3f}")

    print(f"KS p-value = " f"{ks_p_1:.3e}")

   
    obs_med_2 = np.nanmedian(obs_ratio_ram_total_valid)

    mc_med_2 = np.nanmedian(mc_ratio_ram_total)

    ks_stat_2, ks_p_2 = ks_2samp(obs_ratio_ram_total_valid, mc_ratio_ram_total)

    print("log10(F_ram / " "(F_buoy + F_expansion + F_momentum))")
    print(f"Observed mean = " f"{np.mean(obs_ratio_ram_total_valid):.3f}")
    print(f"Observed std = " f"{np.std(obs_ratio_ram_total_valid):.3f}")
    print(f"Observed median = " f"{obs_med_2:.3f}")
    print(f"MC mean = " f"{np.mean(mc_ratio_ram_total):.3f}")
    print(f"MC std = " f"{np.std(mc_ratio_ram_total):.3f}")
    print(f"MC median = " f"{mc_med_2:.3f}")
    print(f"KS statistic = " f"{ks_stat_2:.3f}")

    print(f"KS p-value = " f"{ks_p_2:.3e}")

    ratio_label_1 = r"log$_{10}$(F$_{ram}$/F$_{buoy}$)"

    ratio_label_2 = ( r"log$_{10}$" r"[F$_{ram}$ / " r"(F$_{buoy}$ + F$_{expansion}$ " r"+ F$_m$)]" )


    diag_1 = plot_ratio_diagnostics(
        ratio=mc_ratio_ram_buoy,
        obs_ratio=obs_ratio_ram_buoy_valid,
        ratio_label=ratio_label_1,
        model_name=output_label,
        plot_dir=plot_dir,
        suffix="ram_buoy",
    )

   

    diag_2 = plot_ratio_diagnostics(
        ratio=mc_ratio_ram_total,
        obs_ratio=obs_ratio_ram_total_valid,
        ratio_label=ratio_label_2,
        model_name=output_label,
        plot_dir=plot_dir,
        suffix="ram_total",
    )

    threshold_1 = make_threshold_plot(
        ratio=mc_ratio_ram_buoy,
        conditions=conditions_ram_buoy,
        obs_tail_frac=obs_tail_frac,
        model_name=output_label,
        plot_dir=plot_dir,
        suffix="ram_buoy",
        ratio_label=ratio_label_1,
    )


    threshold_2 = make_threshold_plot(
        ratio=mc_ratio_ram_total,
        conditions=conditions_total,
        obs_tail_frac=obs_tail_frac,
        model_name=output_label,
        plot_dir=plot_dir,
        suffix="ram_total",
        ratio_label=ratio_label_2,
    )

    print()

    print("==========================================")

    print("RATIO 1 BEST THRESHOLD")

    print("==========================================")

    print("Threshold log10 =", threshold_1["best_threshold_log"])

    print("Threshold linear =", threshold_1["best_threshold_linear"])

    print("MC tail fraction =", threshold_1["best_tail_frac"])

    print()

    print("==========================================")

    print("RATIO 2 BEST THRESHOLD")

    print("==========================================")

    print("Threshold log10 =", threshold_2["best_threshold_log"])

    print("Threshold linear =", threshold_2["best_threshold_linear"])

    print("MC tail fraction =", threshold_2["best_tail_frac"])

    # Morph plots
    
    L144_valid = L144_mc[valid]

    size_valid = size_mc[valid]

    morphology_1 = make_morphology_plots(
        mc_ratio=mc_ratio_ram_buoy,
        L144_valid=L144_valid,
        size_valid=size_valid,
        conditions=conditions_ram_buoy,
        model_name=output_label,
        plot_dir=plot_dir,
        suffix="ram_buoy",
        Nplot=Nplot,
    )

    morphology_2 = make_morphology_plots(
        mc_ratio=mc_ratio_ram_total,
        L144_valid=L144_valid,
        size_valid=size_valid,
        conditions=conditions_total,
        model_name=output_label,
        plot_dir=plot_dir,
        suffix="ram_total",
        Nplot=Nplot,
    )

  
    return {
        "model_name": output_label,
        "force_position": model_name,
        "aspect_ratio": aspect_ratio,
        "alpha_radio": alpha_radio,
        "kappa": kappa,
       
        "ratio_labels": {
            "ram_buoy": "log10(F_ram / F_buoy)",
            "ram_total": "log10(F_ram / " "(F_buoy + F_expansion + F_momentum))",},
        
        "ratio_ram_buoy": mc_ratio_ram_buoy,
        "ratio_ram_total": mc_ratio_ram_total,
        
        "obs_ratio_ram_buoy": obs_ratio_ram_buoy,
        "obs_ratio_ram_total": obs_ratio_ram_total,
        
        "best_threshold_ram_buoy_log": threshold_1["best_threshold_log"],
        "best_threshold_ram_buoy": threshold_1["best_threshold_linear"],
        "best_threshold_ram_total_log": threshold_2["best_threshold_log"],
        "best_threshold_ram_total": threshold_2["best_threshold_linear"],
        
        "ks_ram_buoy": ks_stat_1,
        "ks_p_ram_buoy": ks_p_1,
        "ks_ram_total": ks_stat_2,
        "ks_p_ram_total": ks_p_2,
        
        # Force arrays
        
        "F_ram": Fram_model,
        "F_buoy": Fbuoy_model,
        "F_expansion": Fexp_model,
        "F_momentum": F_momentum_mc,
        
        # Segment forces
      
        "F_ram_1": F_ram_segments[0],
        "F_ram_2": F_ram_segments[1],
        "F_ram_3": F_ram_segments[2],
        "F_buoy_1": F_buoy_segments[0],
        "F_buoy_2": F_buoy_segments[1],
        "F_buoy_3": F_buoy_segments[2],
        "F_expansion_1": F_expansion_segments[0],
        "F_expansion_2": F_expansion_segments[1],
        "F_expansion_3": F_expansion_segments[2],
        
        
        "P_int_1": P_int_segments[0],
        "P_int_2": P_int_segments[1],
        "P_int_3": P_int_segments[2],
        
        # Environment
        
        "rho_1": rho_segments[0],
        "rho_2": rho_segments[1],
        "rho_3": rho_segments[2],
        "P_1": P_segments[0],
        "P_2": P_segments[1],
        "P_3": P_segments[2],
        "P_ram_1": P_ram_segments[0],
        "P_ram_2": P_ram_segments[1],
        "P_ram_3": P_ram_segments[2],
        "g_1": g_segments[0],
        "g_2": g_segments[1],
        "g_3": g_segments[2],
        "R200_1": R200_segments[0],
        "R200_2": R200_segments[1],
        "R200_3": R200_segments[2],
        "R_over_R200_1": R_over_R200_segments[0],
        "R_over_R200_2": R_over_R200_segments[1],
        "R_over_R200_3": R_over_R200_segments[2],
        "ram_fraction_1": ram_fraction_segments[0],
        "ram_fraction_2": ram_fraction_segments[1],
        "ram_fraction_3": ram_fraction_segments[2],
        "buoy_fraction_1": buoy_fraction_segments[0],
        "buoy_fraction_2": buoy_fraction_segments[1],
        "buoy_fraction_3": buoy_fraction_segments[2],
        "exp_fraction_1": exp_fraction_segments[0],
        "exp_fraction_2": exp_fraction_segments[1],
        "exp_fraction_3": exp_fraction_segments[2],
        "mom_fraction_1": mom_fraction_segments[0],
        "mom_fraction_2": mom_fraction_segments[1],
        "mom_fraction_3": mom_fraction_segments[2],
        "cs_1": cs_segments[0],
        "cs_2": cs_segments[1],
        "cs_3": cs_segments[2],
        "jet_power": jet_power_mc,
        "size": size_mc,
        "width": width_mc,
        "length": length_mc,
        "area": area_mc,
        "volume": volume_mc,
        "flux_144": flux_mc,
        "valid": valid,
        "n_valid": np.sum(valid),
        "n_total": N,
        # Diagnostics
        "diagnostics_ram_buoy": diag_1,
        "diagnostics_ram_total": diag_2,
        "threshold_ram_buoy": threshold_1,
        "threshold_ram_total": threshold_2,
        "morphology_ram_buoy": morphology_1,
        "morphology_ram_total": morphology_2,
    }


def run_diagnostics(result):

    print("\n")
    print("=" * 80)
    print("MODEL:", result["model_name"])
    print("=" * 80)

    valid = result["valid"]

    for key in ["F_ram", "F_buoy", "F_expansion", "F_momentum"]:

        arr = np.asarray(result[key])

        print(
            f"{key:15s}",
            f"Nvalid={np.sum(np.isfinite(arr))}",
            f"median={np.nanmedian(arr):.3e}",
            f"min={np.nanmin(arr):.3e}",
            f"max={np.nanmax(arr):.3e}",
        )

    print()

    ratios = {
        "Fexp/Fbuoy": result["F_expansion"] / result["F_buoy"],
        "Fmom/Fbuoy": result["F_momentum"] / result["F_buoy"],
        "Fexp/Fmom": result["F_expansion"] / result["F_momentum"],
        "Fram/Fbuoy": result["F_ram"] / result["F_buoy"],
        "Fram/(Fb+Fe)": result["F_ram"] / (result["F_buoy"] + result["F_expansion"]),
        "Fram/(Fb+Fe+Fm)": result["F_ram"]
        / (result["F_buoy"] + result["F_expansion"] + result["F_momentum"]),
    }

    for label, arr in ratios.items():

        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            continue

        print(
            f"{label:20s}",
            f"median={np.nanmedian(arr):.3e}",
            f"16-84%=[{np.nanpercentile(arr,16):.3e}, "
            f"{np.nanpercentile(arr,84):.3e}]",
        )

    print()

   

    Fb = result["F_buoy"]
    Fe = result["F_expansion"]
    Fm = result["F_momentum"]

    denom = Fb + Fe + Fm

    buoy_frac = Fb / denom
    exp_frac = Fe / denom
    mom_frac = Fm / denom

    print("Median denominator fractions")

    print("Buoyancy:", np.nanmedian(buoy_frac))

    print("Expansion:", np.nanmedian(exp_frac))

    print("Momentum:", np.nanmedian(mom_frac))


    # Expansion pressure
   

    for seg in [1, 2, 3]:

        Pint = result[f"P_int_{seg}"]

        print(f"P_int_{seg}", np.nanmedian(Pint), np.nanmin(Pint), np.nanmax(Pint))

    print()

    # Segment force evolution
    

    for seg in [1, 2, 3]:

        print(
            f"Segment {seg}",
            "Fram=",
            np.nanmedian(result[f"F_ram_{seg}"]),
            "Fbuoy=",
            np.nanmedian(result[f"F_buoy_{seg}"]),
            "Fexp=",
            np.nanmedian(result[f"F_expansion_{seg}"]),
        )

    print()
    print("\nMedian total-force fractions")

    for seg in [1, 2, 3]:

        print(
            f"POS{seg}",
            "Ram =",
            np.nanmedian(result[f"ram_fraction_{seg}"]),
            "Buoy =",
            np.nanmedian(result[f"buoy_fraction_{seg}"]),
            "Exp =",
            np.nanmedian(result[f"exp_fraction_{seg}"]),
            "Mom =",
            np.nanmedian(result[f"mom_fraction_{seg}"]),
        )

   

    print("MC ratio1 median:", np.nanmedian(result["ratio_ram_buoy"]))

    print("MC ratio2 median:", np.nanmedian(result["ratio_ram_total"]))

    print("KS ratio1:", result["ks_ram_buoy"])

    print("KS ratio2:", result["ks_ram_total"])

    print("=" * 80)

    print("\nMedian R/R200")

    for seg in [1, 2, 3]:

        print(f"Segment {seg}:", np.nanmedian(result[f"R_over_R200_{seg}"]))
    fig, ax = plt.subplots(figsize=(6, 4))

    for seg, colour in zip([1, 2, 3], ["tab:blue", "tab:orange", "tab:red"]):

        ax.hist(
            result[f"R_over_R200_{seg}"],
            bins=30,
            density=True,
            histtype="step",
            lw=2,
            color=colour,
            label=f"POS{seg}",
        )

    ax.set_xlabel(r"$R/R_{200}$")
    ax.set_ylabel("Density")

    ax.legend()

    outfile = f"{plot_dir}/" f"{result['model_name']}_R_over_R200.png"

    plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.close()


threshold_store = {}

N = 10000  # number of fake FRIS to simulate
Nplot = 500
mc = sample_independent_kdes(df=complete.to_pandas(), N=N)  # set up mc sample
print("\nObserved flux range (mJy)")
print(
    np.nanmin(complete["Total_flux"]),
    np.nanmedian(complete["Total_flux"]),
    np.nanmax(complete["Total_flux"]),
)

print("\nMC flux range (mJy)")
print(
    np.nanmin(mc["total_flux"]),
    np.nanmedian(mc["total_flux"]),
    np.nanmax(mc["total_flux"]),
)
flux_models = {
    "baseline": {1: 0.30, 2: 0.125, 3: 0.075}
    # },
    # "uniform": {
    #     1:1/6,
    #     2:1/6,
    #     3:1/6
    # },
    # "more_outer_flux": {
    #     1:0.25,
    #     2:0.15,
    #     3:0.10
    # },
    # "more_inner_flux": {
    #     1:0.35,
    #     2:0.10,
    #     3:0.05
    # }
}
summary_rows = []
all_results = {}

for flux_name, flux_fractions in flux_models.items():

    print()
    print("=" * 80)
    print("FLUX MODEL:", flux_name)
    print("=" * 80)

    for AR in [1, 2, 3, 4, 5, 6, 7, 8]:
        for kappa in [1, 3, 5, 10, 30, 50, 100]:

            for model in ["POS1", "POS2", "POS3"]:

                label = f"{flux_name}_{model}_" f"AR{AR:.1f}_" f"K{kappa}"

                print("running for label", label)

                result = run_physical_mc_model(
                    mc=mc,
                    ram_col="F_ram",
                    buoy_col="F_buoy",
                    model_name=model,
                    aspect_ratio=AR,
                    flux_fractions=flux_fractions,
                    alpha_radio=0.5,
                    kappa=kappa,
                    conditions_ram_buoy=np.arange(-1, 1.5, 0.05),
                    conditions_total=np.arange(-1, 1.5, 0.05),
                    Nplot=Nplot,
                    output_label=label,
                )

                threshold_store[model] = result["best_threshold_ram_total"]
                ratio = result["ratio_ram_total"]

                threshold = result["best_threshold_ram_total_log"]

                tailed = ratio > threshold
                lobed = ratio <= threshold
                all_results[label] = result
                L500_valid = np.log10(mc["L500"][result["valid"]])

                L144_valid = np.log10(result["jet_power"] / L_to_P)[result["valid"]]

                # plt.close()
                run_diagnostics(result)

                summary_rows.append(
                    {
                        "FluxModel": flux_name,
                        "Segment": model,
                        "AR": AR,
                        "kappa": kappa,
                        # KS tests
                        "KS_R1": result["ks_ram_buoy"],
                        "KS_R2": result["ks_ram_total"],
                        "KS_p_R1": result["ks_p_ram_buoy"],
                        "KS_p_R2": result["ks_p_ram_total"],
                        # thresholds (linear)
                        "Thr_R1": result["best_threshold_ram_buoy"],
                        "Thr_R2": result["best_threshold_ram_total"],
                        # thresholds (log)
                        "ThrLog_R1": result["best_threshold_ram_buoy_log"],
                        "ThrLog_R2": result["best_threshold_ram_total_log"],
                        # medians
                        "Med_R1": np.nanmedian(result["ratio_ram_buoy"]),
                        "Med_R2": np.nanmedian(result["ratio_ram_total"]),
                    }
                )

            # Total force budget plot

            ram = []
            buoy = []
            exp = []
            mom = []
            R = []
            thresholds = {
                1: threshold_store["POS1"],
                2: threshold_store["POS2"],
                3: threshold_store["POS3"],
            }
            segment_results = {
                1: all_results[f"{flux_name}_POS1_AR{AR:.1f}_K{kappa}"],
                2: all_results[f"{flux_name}_POS2_AR{AR:.1f}_K{kappa}"],
                3: all_results[f"{flux_name}_POS3_AR{AR:.1f}_K{kappa}"],
            }

            for seg in [1, 2, 3]:

                seg_result = segment_results[seg]

                ratio = seg_result[f"F_ram_{seg}"] / (
                    seg_result[f"F_buoy_{seg}"]
                    + seg_result[f"F_expansion_{seg}"]
                    + seg_result["F_momentum"]
                )

                threshold = thresholds[seg]

                mask = np.abs(ratio - threshold) < 0.1 * threshold

                print(
                    f"POS{seg}", "Threshold =", threshold, "N selected =", np.sum(mask)
                )

                R.append(np.nanmedian(seg_result[f"R_over_R200_{seg}"]))

                ram.append(np.nanmedian(seg_result[f"ram_fraction_{seg}"][mask]))

                buoy.append(np.nanmedian(seg_result[f"buoy_fraction_{seg}"][mask]))

                exp.append(np.nanmedian(seg_result[f"exp_fraction_{seg}"][mask]))

                mom.append(np.nanmedian(seg_result[f"mom_fraction_{seg}"][mask]))

            x = np.arange(len(R))

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.bar(x, ram, label="Ram pressure")

            ax.bar(x, buoy, bottom=ram, label="Buoyancy")

            ax.bar(x, exp, bottom=np.array(ram) + np.array(buoy), label="Expansion")

            ax.bar(
                x,
                mom,
                bottom=np.array(ram) + np.array(buoy) + np.array(exp),
                label="Momentum",
            )

            ax.set_xticks(x)

            ax.set_xticklabels([f"{r:.3f}" for r in R])

            ax.set_xlabel(r"$R/R_{200}$")

            ax.set_ylabel("Fraction of total force budget")

            ax.legend()

            plt.tight_layout()

            outfile = f"{plot_dir}/" f"transition_force_budget_AR{AR:.1f}_K{kappa}.pdf"

            plt.savefig(outfile, dpi=300, bbox_inches="tight")

            plt.close()

            print("\nSaved total force budget plot:", outfile)

            for pos in [1, 2, 3]:

                result = segment_results[pos]

                ratio = result["ratio_ram_total"]

                threshold = result["best_threshold_ram_total_log"]

                tailed = ratio > threshold
                lobed = ratio <= threshold

                valid = result["valid"]

                with PdfPages(
                    f"{plot_dir}/POS{pos}_AR{AR:.1f}_K{kappa}_Pint_Pext_lobed_tailed.pdf"
                ) as pdf:

                    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

                    region_names = {1: "tip", 2: "midpoint", 3: "end"}

                    for seg, ax in zip([1, 2, 3], axs):

                        pressure_ratio = result[f"P_int_{seg}"] / result[f"P_{seg}"]

                        pressure_ratio = pressure_ratio[valid]

                        lobed_ratio = pressure_ratio[lobed]
                        tailed_ratio = pressure_ratio[tailed]

                        lobed_ratio = lobed_ratio[np.isfinite(lobed_ratio)]

                        tailed_ratio = tailed_ratio[np.isfinite(tailed_ratio)]

                        allvals = np.concatenate([lobed_ratio, tailed_ratio])

                        bins = np.logspace(
                            np.log10(np.nanpercentile(allvals, 1)),
                            np.log10(np.nanpercentile(allvals, 99)),
                            15,
                        )

                        ax.hist(
                            tailed_ratio,
                            bins=bins,
                            alpha=0.5,
                            color="indianred",
                            label=f"Tailed (med={np.nanmedian(tailed_ratio):.2f})",
                        )

                        ax.hist(
                            lobed_ratio,
                            bins=bins,
                            alpha=0.5,
                            color="k",
                            hatch="////",
                            label=f"Lobed (med={np.nanmedian(lobed_ratio):.2f})",
                        )

                        ax.axvline(
                            np.nanmedian(tailed_ratio), color="indianred", ls="--"
                        )

                        ax.axvline(np.nanmedian(lobed_ratio), color="k", ls="--")

                        ax.set_xscale("log")

                        ax.set_xlabel(rf"$P_{{int}}/P_{{ext}}$ ({region_names[seg]})")

                        ax.set_ylabel("Number")

                        ax.legend()

                    fig.suptitle(f"POS{pos} morphology model", fontsize=14)

                    plt.tight_layout()

                    pdf.savefig(fig)

                    plt.close(fig)
print("modelling finished successfully,saving output")
summary = pd.DataFrame(summary_rows)

summary.to_csv("MC_force_summary_full_run.csv", index=False)
import pickle

with open("MC_all_results.pkl", "wb") as f:

    pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved MC results to MC_all_results.pkl")

with open("MC_sample.pkl", "wb") as f:

    pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)

print(summary.head())
print("output saved")
