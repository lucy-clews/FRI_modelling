#!/usr/bin/env python
# coding: utf-8

#code to model all forces acting in/on 3 segments along LoTSS radio jets. NOTE: Has to be run a inside virtual env with cololussus and dpmhalo installed!! (i.e. do source /beegfs/lofar/lclews/NG1044/pysynch/astropy-env/bin/activate,  then python3 FRI_modelling_ridge_segment.py)

#import modules 
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

import os
import math
import glob
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.modeling.models import Ellipse2D
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy import constants as const
from astropy import table
from astropy.table import QTable, Table
from astropy.wcs import WCS
from matplotlib import pyplot as plt,cm
from matplotlib import colors
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
from sizeflux_tools import Flood
from ridge_toolkitDR2 import TotalFluxSelector,GetCutoutArray
import glob as glob
import pandas as pd
import dpmhalo
from dpmhalo import myutils,modelprofiles,modelparams
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sympy.geometry as gm
from scipy.ndimage import gaussian_filter,label
from scipy import interpolate
from scipy.interpolate import splrep, BSpline,splprep,splev, make_interp_spline, CubicSpline
from scipy import interpolate as itp
import subprocess
import re
import pandas as pd
from regions import Regions,RectanglePixelRegion,PixCoord
from matplotlib.patches import Circle
from skimage.draw import line as bresenham_line

from scipy.interpolate import make_interp_spline
from skimage.draw import line as bresenham_line

import dpmhalo
from dpmhalo import myutils, modelparams
from colossus.halo import profile_nfw
from synchro import SynchSource
from sizeflux_tools import Flood


# Constants

conc = 4
m_p = 1.67262192e-27        # kg
u_e = 0.6                   # mean molecular weight per electron (not directly used here)
h = 0.6766                   # H0/100
G = 6.67e-11                # SI
M_solar = 1.988e30          # kg

KAPPA_VALUES = list(range(0, 120, 5))
# Functions

def make_geometry_diagnostics( row, source, out_dir,image,  wcs_cutout,    xnew_u, ynew_u,  x_pix, y_pix,    x_host, y_host,host_idx,
                               end_idx, x0, y0, m_perp, x05, y05, m_perp5, mask,  xs_sel, ys_sel, seg_masks, u_ridge, xs, ys, best_pair,
                               width_kpc_deconvolved,  centres=None,  dirs=None,  widths_as_deconv=None, heights_as_deconv=None,):
    """
    Function to make diagnostic plots of how geometry calcs progressed at each steps makes a plot.
    
    Creates following figs as pdf:

    1) step1_ridge.pdf - ridge drawing, host location and highlights which end got picked for force calcs
    2) step2_side_selection.pdf -  ridge drawing, host location and highlights pixels used as end picked for force calcs
    3) step3_width.pdf - indicates denconvoled width of source (at jet end)
    4) step4_uridge.pdf - shows all pixels beyond host galaxy on selected side, colour codes them by 'normalised distance along ridge'
    5) step5_segments.pdf - shows all pixels beyond host galaxy on selected side, colour codes them by segment allocated
    6) step6_region_fits.pdf - shows segment geometry (widths, heights) through rectangles (couldn't plot 3d cylinder, use this as proxy)
    7) step7_pressure_ratios.pdf - shows ridge, host, 3 segments (as rectangles), pressure ratio labelled in each segment
    """
    # ============================================================
# GEOMETRY DIAGNOSTIC PLOTS
# - Reduced whitespace
# - Step label on every plot
# - Plot 4 colourbar separated from axes
# - Plot 5 colourbar labelled "Segment number"
# ============================================================

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    FS = 16
    TICKFS = 14
    
    finite_mask = np.isfinite(image)
    
    ys_all, xs_all = np.nonzero(finite_mask)
    
    pad = 10
    
    ymin = max(ys_all.min() - pad, 0)
    ymax = min(ys_all.max() + pad, image.shape[0])
    
    xmin = max(xs_all.min() - pad, 0)
    xmax = min(xs_all.max() + pad, image.shape[1])
    
    
    def setup_axes():
    
        fig = plt.figure(figsize=(8, 8))
    
        ax = fig.add_axes(
            [0.12, 0.12, 0.72, 0.78],
            projection=wcs_cutout
        )
    
        vmin = np.nanpercentile(image, 5)
        vmax = np.nanpercentile(image, 99)
    
        ax.imshow(
            image,
            origin='lower',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap='inferno'
        )
    
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
        ax.coords[0].set_axislabel(
            "RA (J2000)",
            fontsize=FS
        )
    
        ax.coords[1].set_axislabel(
            "DEC (J2000)",
            fontsize=FS
        )
    
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
    
        ax.coords[0].set_ticklabel(size=TICKFS)
        ax.coords[1].set_ticklabel(size=TICKFS)
    
        return fig, ax
    
    
    def add_step_label(ax, step):
    
        ax.text(
            0.03,
            0.03,
            f"Step {step}",
            transform=ax.transAxes,
            fontsize=FS-2,
            color='white',
            ha='left',
            va='bottom',
            bbox=dict(
                facecolor='black',
                alpha=0.7,
                edgecolor='none'
            )
        )
    
    
    # ============================================================
    # PLOT 1
    # ============================================================
    
    fig, ax = setup_axes()
    
    ax.plot(
        xnew_u,
        ynew_u,
        color='black',
        lw=2,
        transform=ax.get_transform("world"),
        label='Spline'
    )
    
    ax.plot(
        x_host,
        y_host,
        marker='*',
        color='r',
        ms=15,
        transform=ax.get_transform("pixel"),
        label='Host'
    )
    
    ax.plot(
        x_pix[end_idx],
        y_pix[end_idx],
        'co',
        ms=10,
        transform=ax.get_transform("pixel"),
        label='Chosen end'
    )
    
    ax.legend(fontsize=FS)
    add_step_label(ax, 1)
    
    plt.savefig(
        f"{out_dir}/{source}_geometry_step1_ridge.pdf",
        bbox_inches='tight',
        pad_inches=0.02
    )
    
    plt.close()
    
    
    # ============================================================
    # PLOT 2
    # ============================================================
    
    fig, ax = setup_axes()
    
    yy_mask, xx_mask = np.nonzero(mask)
    
    ax.scatter(
        xx_mask,
        yy_mask,
        c='cyan',
        s=2,
        alpha=0.5,
        transform=ax.get_transform("pixel"),
        label='Chosen side'
    )
    
    ax.plot(
        xnew_u,
        ynew_u,
        color='black',
        lw=2,
        transform=ax.get_transform("world"),
        label='Ridgeline spline'
    )
    
    ax.plot(
        x_host,
        y_host,
        marker='*',
        color='r',
        ms=15,
        transform=ax.get_transform("pixel"),
        label='Host'
    )
    
    ax.legend(fontsize=FS)
    add_step_label(ax, 2)
    
    plt.savefig(
        f"{out_dir}/{source}_geometry_step2_side_selection.pdf",
        bbox_inches='tight',
        pad_inches=0.02
    )
    
    plt.close()
    
    
    # ============================================================
    # PLOT 3
    # ============================================================
    
    fig, ax = setup_axes()
    
    ax.scatter(
        xs_sel,
        ys_sel,
        c='red',
        s=3,
        alpha=0.7,
        transform=ax.get_transform("pixel"),
        label='Selected pixels'
    )
    
    if best_pair is not None:
    
        pt1, pt2 = best_pair
    
        ax.plot(
            [pt1.ra.deg, pt2.ra.deg],
            [pt1.dec.deg, pt2.dec.deg],
            lw=3,
            color='cyan',
            transform=ax.get_transform("world"),
            label=f"Deconvolved width = {width_kpc_deconvolved:.1f} kpc"
        )
    
    ax.legend(fontsize=FS)
    add_step_label(ax, 3)
    
    plt.savefig(
        f"{out_dir}/{source}_geometry_step3_width.pdf",
        bbox_inches='tight',
        pad_inches=0.02
    )
    
    plt.close()
    
    
    # ============================================================
    # PLOT 4
    # ============================================================
    
    fig, ax = setup_axes()
    
    sc = ax.scatter(
        xs,
        ys,
        c=u_ridge,
        cmap='viridis',
        s=3,
        transform=ax.get_transform("pixel")
    )
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes(
        "right",
        size="4%",
        pad=0.08
    )
    
    cbar = fig.colorbar(
        sc,
        cax=cax
    )
    
    cbar.set_label(
        "Normalised distance along ridge",
        fontsize=FS
    )
    
    cbar.ax.tick_params(
        labelsize=TICKFS
    )
    
    add_step_label(ax, 4)
    
    plt.savefig(
        f"{out_dir}/{source}_geometry_step4_uridge.pdf",
        bbox_inches='tight',
        pad_inches=0.02
    )
    
    plt.close()
    
    
    # ============================================================
    # PLOT 5
    # ============================================================
    
    fig, ax = setup_axes()
    
    all_x = []
    all_y = []
    all_seg = []
    
    for i, seg_mask in enumerate(seg_masks):
    
        yy, xx = np.nonzero(seg_mask)
    
        all_x.extend(xx)
        all_y.extend(yy)
    
        all_seg.extend(
            np.full(len(xx), i + 1)
        )
    
    sc = ax.scatter(
        all_x,
        all_y,
        c=all_seg,
        cmap='viridis',
        s=4,
        alpha=0.5,
        transform=ax.get_transform("pixel")
    )
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes(
        "right",
        size="4%",
        pad=0.08
    )
    
    cbar = fig.colorbar(
        sc,
        cax=cax,
        ticks=[1, 2, 3]
    )
    
    cbar.set_label(
        "Segment number",
        fontsize=FS
    )
    
    cbar.ax.tick_params(
        labelsize=TICKFS
    )
    
    ax.plot(
        x_host,
        y_host,
        marker='*',
        color='r',
        ms=15,
        transform=ax.get_transform("pixel"),
        label='Host'
    )
    
    ax.legend(fontsize=FS)
    
    add_step_label(ax, 5)
    
    plt.savefig(
        f"{out_dir}/{source}_geometry_step5_segments.pdf",
        bbox_inches='tight',
        pad_inches=0.02
    )
    
    plt.close()
    
    
    # ============================================================
    # PLOT 6
    # ============================================================
    
    if centres is not None:
    
        fig, ax = setup_axes()
    
        cols = ['cyan', 'lime', 'magenta']
    
        for i in range(3):
    
            if centres[i] is None:
                continue
    
            if widths_as_deconv[i] is None:
                continue
    
            x_c, y_c = centres[i]
            px, py = dirs[i]
    
            pixscale_as = abs(
                wcs_cutout.wcs.cdelt[0]
            ) * 3600.
    
            L = widths_as_deconv[i] / pixscale_as
            H = heights_as_deconv[i] / pixscale_as
    
            theta = np.degrees(
                np.arctan2(py, px)
            )
    
            rect = Rectangle(
                (x_c - L/2, y_c - H/2),
                L,
                H,
                edgecolor=cols[i],
                facecolor='none',
                lw=2
            )
    
            trans = (
                transforms.Affine2D()
                .rotate_deg_around(
                    x_c,
                    y_c,
                    theta
                )
                + ax.get_transform("pixel")
            )
    
            rect.set_transform(trans)
    
            ax.add_patch(rect)
    
        add_step_label(ax, 6)
    
        plt.savefig(
            f"{out_dir}/{source}_geometry_step6_region_fits.pdf",
            bbox_inches='tight',
            pad_inches=0.02
        )
    
        plt.close()
        
    # plot 7
  

    if centres is not None:

        fig, ax = setup_axes()

        cols = ['cyan', 'lime', 'magenta']
        labels = ['Inner', 'Middle', 'Outer']
        offsets = [
            (+5, +10),   # inner
            (+5, -10),   # middle
            (-15, +5),   # outer
        ]
        for i in range(3):

            seg_mask = seg_masks[i]

            yy, xx = np.nonzero(seg_mask)

            ax.scatter(
                xx,
                yy,
                s=4,
                c=cols[i],
                alpha=0.25,
                transform=ax.get_transform("pixel")
            )

            if centres[i] is None:
                continue

            if widths_as_deconv[i] is None:
                continue

            x_c, y_c = centres[i]
            px, py = dirs[i]
            dx, dy = offsets[i]


            pixscale_as = abs(
                wcs_cutout.wcs.cdelt[0]
            ) * 3600.

            L = widths_as_deconv[i] / pixscale_as
            H = heights_as_deconv[i] / pixscale_as

            theta = np.degrees(
                np.arctan2(py, px)
            )

            # Draw a cylindrical side-view (rounded capsule)
            radius = min(L, H) / 2
            
            cylinder = FancyBboxPatch(
                (x_c - L/2, y_c - H/2),
                L,
                H,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                edgecolor=cols[i],
                facecolor='none',
                linewidth=2
            )
            
            trans = (
                transforms.Affine2D()
                .rotate_deg_around(
                    x_c,
                    y_c,
                    theta
                )
                + ax.get_transform("pixel")
            )
            
            cylinder.set_transform(trans)
            ax.add_patch(cylinder)

            Pi = row.get(f'Pint/Pext{i+1}', np.nan)

            if np.isfinite(Pi):

                txt = (
                    f"{labels[i]}\n"
                    f"Pint/Pext = {Pi:.2f}"
                )

                ax.text(
                    x_c + dx,
                    y_c + dy,
                    txt,
                    color='white',
                    fontsize=FS,
                    transform=ax.get_transform("pixel"),
                    bbox=dict(
                        facecolor='black',
                        alpha=0.6,
                        edgecolor='none'
                    )
                )

            ax.plot(
                x_c,
                y_c,
                'o',
                color=cols[i],
                ms=6,
                transform=ax.get_transform("pixel"),label='Segment centre'
            )

        ax.plot(
            xnew_u,
            ynew_u,
            'k-',
            lw=1.5,
            transform=ax.get_transform("world"),label='Ridgeline spline'
        )

        ax.plot(
            x_host,
            y_host,
            '*',
            color='r',
            ms=15,
            transform=ax.get_transform("pixel"),label='Host'
        )

     #   ax.set_title(
      #      f"{source}: pressure regions and Pint/Pext"
       # )
        ax.legend(fontsize=FS)
        plt.savefig(
            f"{out_dir}/{source}_geometry_step7_pressure_ratios.pdf")

        plt.close()

    # ============================================================
    # SUMMARY FIGURE (2 x 3)
    # Run AFTER all geometry quantities have been calculated
    # Uses existing FS and TICKFS values
    # ============================================================
    
    fig = plt.figure(figsize=(12,16))
    
    gs = fig.add_gridspec(
        3, 2,
        left=0.04,
        right=0.95,
        bottom=0.05,
        top=0.98,
        wspace=0.08,
        hspace=0.08
    )
    
    vmin = np.nanpercentile(image, 5)
    vmax = np.nanpercentile(image, 99)
    
    titles = [
        "Step 1: Ridgeline spline",
        "Step 2: Side selection",
        "Step 3: Deconvolved width measurement",
        "Step 4: Normalised distance along the ridge",
        "Step 5: Segmentation",
        "Step 6: Allocate regions"
    ]
    
    axes = []
    
    for i in range(6):
    
        ax = fig.add_subplot(
            gs[i // 2, i % 2],
            projection=wcs_cutout
        )
        row = i // 2
        col = i % 2
        ax.imshow(
            image,
            origin='lower',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap='inferno'
        )
    
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
        if row < 2:

            ax.coords[0].set_axislabel("")
            ax.coords[0].set_ticklabel_visible(False)
        
        else:
        
            ax.coords[0].set_axislabel(
                "RA (J2000)",
                fontsize=int(FS )
            )
    
        if col == 0:
            ax.coords[1].set_axislabel(
                "DEC (J2000)",
                fontsize=int(FS)
            )
        else:
            ax.coords[1].set_axislabel("")
            ax.coords[1].set_ticklabel_visible(False)
            
        ax.coords[0].set_ticks_position('b')
        ax.coords[0].set_ticklabel_position('b')
        ax.coords[1].set_ticks_position('l')
        ax.coords[1].set_ticklabel_position('l')
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
    
        ax.coords[0].set_ticklabel(
            size=int(TICKFS )
        )
    
        ax.coords[1].set_ticklabel(
            size=int(TICKFS )
        )
    
        ax.set_title(
            titles[i],
            fontsize=int(FS )
        )
    
        axes.append(ax)
    
    ax1, ax2, ax3, ax4, ax5, ax6 = axes
    
    # ============================================================
    # PANEL 1
    # ============================================================
    
    ax1.plot(
        xnew_u,
        ynew_u,
        color='black',
        lw=2,
        transform=ax1.get_transform("world"),
        label='Spline'
    )
    
    ax1.plot(
        x_host,
        y_host,
        '*',
        color='r',
        ms=12,
        transform=ax1.get_transform("pixel"),
        label='Host'
    )
    
    ax1.plot(
        x_pix[end_idx],
        y_pix[end_idx],
        'co',
        ms=8,
        transform=ax1.get_transform("pixel"),
        label='Chosen end'
    )
    
    ax1.legend(
        fontsize=int(FS),
        loc='upper left'
    )
    
    # ============================================================
    # PANEL 2
    # ============================================================
    
    yy_mask, xx_mask = np.nonzero(mask)
    
    ax2.scatter(
        xx_mask,
        yy_mask,
        s=10,
        c='cyan',
        alpha=0.9,
        transform=ax2.get_transform("pixel"),label='Chosen end'
    )
    
    ax2.plot(
        xnew_u,
        ynew_u,
        color='black',
        lw=2,
        transform=ax2.get_transform("world"),
        label='Spline'
    )
    
    ax2.plot(
        x_host,
        y_host,
        '*',
        color='r',
        ms=12,
        transform=ax2.get_transform("pixel"),
        label='Host'
    )
    ax2.legend(
        fontsize=int(FS),
        loc='upper left'
    )
    # ============================================================
    # PANEL 3
    # ============================================================
    
    ax3.scatter(
        xs_sel,
        ys_sel,
        c='cyan',
        s=10,
        alpha=0.9,
        transform=ax3.get_transform("pixel"),label='Chosen end'
    )
    
    if best_pair is not None:
    
        pt1, pt2 = best_pair
    
        ax3.plot(
            [pt1.ra.deg, pt2.ra.deg],
            [pt1.dec.deg, pt2.dec.deg],
            color='g',
            lw=3,
            transform=ax3.get_transform("world"),label= f"Deconvolved width = {width_kpc_deconvolved:.1f} kpc" )
    
    ax3.legend(
        fontsize=int(FS),
        loc='upper left'
    )
    
    # ============================================================
    # PANEL 4
    # ============================================================
    
    sc4 = ax4.scatter(
        xs,
        ys,
        c=u_ridge,
        cmap='viridis',
        s=10,
        transform=ax4.get_transform("pixel")
    )
    
    cb4 = fig.colorbar(
        sc4,
        ax=ax4,
        fraction=0.046,
        pad=0.0
    )
    
    cb4.set_label(
        "Normalised distance",
        fontsize=int(FS )
    )
    
    cb4.ax.tick_params(
        labelsize=int(TICKFS)
    )
    
    # ============================================================
    # PANEL 5
    # ============================================================
    
    all_x = []
    all_y = []
    all_seg = []
    from matplotlib.colors import ListedColormap, BoundaryNorm

    segment_colours = ['yellow','cyan', 'magenta']
    
    cmap_seg = ListedColormap(segment_colours)
    
    norm_seg = BoundaryNorm(
        [0.5, 1.5, 2.5, 3.5],
        cmap_seg.N
    )
    for i, seg_mask in enumerate(seg_masks):
    
        yy, xx = np.nonzero(seg_mask)
    
        all_x.extend(xx)
        all_y.extend(yy)
        all_seg.extend(
            np.full(len(xx), i + 1)
        )
    
    sc5 = ax5.scatter(
    all_x,
    all_y,
    c=all_seg,
    cmap=cmap_seg,
    norm=norm_seg,
    s=10,
    alpha=0.9,
    transform=ax5.get_transform("pixel")
)
    
    ax5.plot(
        x_host,
        y_host,
        '*',
        color='r',
        ms=12,
        transform=ax5.get_transform("pixel"),label='Host'
    )
    
    cb5 = fig.colorbar(
    sc5,
    ax=ax5,
    ticks=[1, 2, 3],
    fraction=0.046,
    pad=0.0
)

    cb5.ax.set_yticklabels(
        ['1', '2', '3']
    )
    
    cb5.set_label(
        "Segment",
        fontsize=int(FS ))
    
    cb5.ax.tick_params(
        labelsize=int(TICKFS )
    )
    ax5.legend(
        fontsize=int(FS ),
        loc='upper right'
    )
    # ============================================================
    # PANEL 6
    # ============================================================
    
    if centres is not None:
    
        cols = ['yellow','cyan', 'magenta']
    
        for i in range(3):
    
            if centres[i] is None:
                continue
    
            if widths_as_deconv[i] is None:
                continue
    
            x_c, y_c = centres[i]
            px, py = dirs[i]
    
            pixscale_as = (
                abs(wcs_cutout.wcs.cdelt[0]) * 3600.
            )
    
            L = widths_as_deconv[i] / pixscale_as
            H = heights_as_deconv[i] / pixscale_as
    
            theta = np.degrees(
                np.arctan2(py, px)
            )
                
                        # Draw a cylindrical side-view (rounded capsule)
            radius = min(L, H) / 2
            
            cylinder = FancyBboxPatch(
                (x_c - L/2, y_c - H/2),
                L,
                H,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                edgecolor=cols[i],
                facecolor='none',
                linewidth=3
            )
            
            trans = (
                transforms.Affine2D()
                .rotate_deg_around(
                    x_c,
                    y_c,
                    theta
                )
                + ax6.get_transform("pixel")
            )
            
            cylinder.set_transform(trans)
            ax6.add_patch(cylinder)
    
   
    # ============================================================
    # SAVE SUMMARY FIGURE
    # ============================================================

    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(
        out_dir, f"{source}_geometry_summary.pdf"
    )

    png_path = os.path.join(
        out_dir, f"{source}_geometry_summary.png"
    )

    print("Saving summary figure...")
    print("Output directory:", os.path.abspath(out_dir))
    print("PDF:", pdf_path)
    print("PNG:", png_path)

    # Save the actual summary figure
    fig.savefig(
        pdf_path,
        bbox_inches='tight',
        dpi=300
    )

    fig.savefig(
        png_path,
        bbox_inches='tight',
        dpi=300
    )

    # Verify that both files exist
    print("PDF exists:", os.path.exists(pdf_path))
    print("PNG exists:", os.path.exists(png_path))

    if os.path.exists(pdf_path):
        print("PDF size:", os.path.getsize(pdf_path), "bytes")

    if os.path.exists(png_path):
        print("PNG size:", os.path.getsize(png_path), "bytes")

    plt.close(fig)

def smaller_side_mask(mask_above, mask_below):
    """
    function to check which side is smaller 
    """
    return mask_above if np.sum(mask_above) < np.sum(mask_below) else mask_below


def load_ridgeline(source):
    """
    Function to load existing ridge info
    """
    pattern1 = f"/beegfs/lofar/mjh/rgz/*/ridgeline/*/ridges/{source}_ridge1.txt"
    pattern2 = f"/beegfs/lofar/mjh/rgz/*/ridgeline/*/ridges/{source}_ridge2.txt"

    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2)

    if len(files1) == 0 or len(files2) == 0:
        raise FileNotFoundError(
            f"Ridgeline files not found for {source}\n"
            f"Looked for:\n  {pattern1}\n  {pattern2}"
        )

    r1 = np.loadtxt(files1[0])
    r2 = np.loadtxt(files2[0])
    r2 = np.delete(r2, 0, axis=0)[::-1]
    ridge = np.vstack((r2, r1))
    return ridge[~np.isnan(ridge).any(axis=1)]


def spline_u(ra, dec):
    """
    function to plot ridgeline spline, taken from Clews+2025
    """
    # Parametric spline along the ridge in world coords
    length_on_spline = [0]
    percentage_length_along_spline = []
    u_param = np.arange(len(ra))
    p = np.array([ra, dec])
    dp = p[:, 1:] - p[:, :-1]
    l = (dp**2).sum(axis=0)
    u_cord = np.sqrt(l).cumsum()
    u_cord = np.r_[0, u_cord]

    u_test = np.linspace(0, len(ra) - 1, 200)
    spline = make_interp_spline(u_param, p.T, bc_type='natural')
    xnew_u, ynew_u = spline(u_test).T

    dx_du, dy_du = spline.derivative(nu=1)(u_test).T
    d2x_du2, d2y_du2 = spline.derivative(nu=2)(u_test).T

    distance = 0
    for j in range(len(xnew_u) - 1):
        distance += np.sqrt((xnew_u[j + 1] - xnew_u[j])**2 +
                            (ynew_u[j + 1] - ynew_u[j])**2)
        length_on_spline.append(distance)

    spline_length = length_on_spline[-1]
    for p_ in length_on_spline:
        percentage_length_along_spline.append(((p_ / spline_length) * 100))

    return (xnew_u, ynew_u, dx_du, dy_du, d2x_du2, d2y_du2,
            length_on_spline, percentage_length_along_spline)


def get_spline_cached(source, ridgeline, cache_dir):
    """
    if spline already computed, function will call file instead of re-calculating it 
    """
    
    fn = f"{cache_dir}/{source}_spline.npz"
    if os.path.exists(fn):
        d = np.load(fn)
        return tuple(d[k] for k in d.files)

    out = spline_u(ridgeline[:, 0], ridgeline[:, 1])
    np.savez(
        fn,
        xnew_u=out[0],
        ynew_u=out[1],
        dx_du=out[2],
        dy_du=out[3],
        d2x_du2=out[4],
        d2y_du2=out[5],
        length_on_spline=out[6],
        percentage_length=out[7],
    )
    return out


def gen_floodfill_cutout(row, source_name, splinex, spliney):
    """
    Function to create floodfill cutout at 5rms threshold and translate ridge coords to match it
    """
    source = source_name
    fits_path = glob.glob(f'/beegfs/lofar/mjh/rgz/*/ridgeline/*/fits/{source}.fits')
    fits_cutout_path = glob.glob(f'/beegfs/lofar/mjh/rgz/*/ridgeline/*/fits_cutouts/{source}-cutout.fits')

    hdu = fits.open(fits_path[0])[0]
    wcs = WCS(hdu.header)
    hdu2 = fits.open(fits_cutout_path[0])[0]
    wcs2 = WCS(hdu2.header)

    sx = np.array(splinex)
    sy = np.array(spliney)
    coords = pd.DataFrame({'x': sx, 'y': sy})
    ra_dec = wcs2.all_pix2world(coords.to_numpy(), 0).T
    ra = ra_dec[0]
    dec = ra_dec[1]

    comp_path = fits_path[0].split('fits', 1)[0]
    with fits.open(comp_path + '/components.fits') as data:
        comp = Table(data[1].data)

    Lra = np.array(row['RA'])
    Ldec = np.array(row['DEC'])
    isize = np.array(row['Size'])
    lmsize = row['Size'] / (3600.0 * 0.0004166667)
    rms = row['Isl_rms'] / 1000

    size = (lmsize, lmsize)
    centre = SkyCoord(Lra * u.deg, Ldec * u.deg, frame='fk5')

    ffo = Flood(comp)
    flux_array = np.load((comp_path + '/rms4/') + source + '.npy')
    hdu.data = flux_array

    cinc, cexc = ffo.select(source, Lra, Ldec, isize, comp, verbose=False)
    flood_mask_array = ffo.mask(source, cinc, cexc, hdu, rmsthres=5 * rms, verbose=False)

    cutout_header = hdu.header.copy()
    cutout = Cutout2D(flood_mask_array[1], centre, size, wcs=wcs)
    cutout_wcs_to_header = cutout.wcs.to_header()
    cutout_wcs_to_header["WCSAXES"] = hdu.header['NAXIS']
    cutout_header.update(cutout_wcs_to_header)
    hdu_cutout = fits.PrimaryHDU(data=cutout.data, header=cutout_header)

    cutout2D = cutout.data
    x, y = cutout.wcs.world_to_pixel_values(ra, dec)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    in_bounds = ((x < cutout2D.shape[1]) & (y < cutout2D.shape[0]))
    x = x[in_bounds]
    y = y[in_bounds]

    ra_valid = ra[in_bounds]
    dec_valid = dec[in_bounds]

    finite_mask = np.isfinite(cutout2D[y, x])
    ra_valid = ra_valid[finite_mask]
    dec_valid = dec_valid[finite_mask]

    savepath = (f'/beegfs/lofar/lclews/FRI_evolution/subclassification/' f'inspection_results/buoyancy/{source}-floodfillcutout_5rms.fits')
    hdu_cutout.writeto(savepath, overwrite=True)

    
    return savepath, ra_valid, dec_valid


def get_floodfill_cutout_cached(row, source, xnew_u, ynew_u, cache_dir):
    
    """
    what this does: if floodfill cutout already exists, use it instead of redoing it
    """
    
    fits_file = f"{cache_dir}/{source}_floodfillcutout_5rms.fits"
    coord_file = f"{cache_dir}/{source}_radec.npz"
    if os.path.exists(fits_file) and os.path.exists(coord_file):
        d = np.load(coord_file)
        return fits_file, d["ra"], d["dec"]

    fits_path, ra, dec = gen_floodfill_cutout(row, source, xnew_u, ynew_u)
    os.rename(fits_path, fits_file)
    np.savez(coord_file, ra=ra, dec=dec)
    return fits_file, ra, dec


def get_width(xs, ys, wcs_cutout, dist, image):

    """
    compute (convolved) width of selected part of source in kpc
    """
    
    # Distance between two pixels furthest apart, excluding gaps
    distance_kpc = dist.value * 1e3  # Mpc → kpc
    world = wcs_cutout.all_pix2world(xs, ys, 0)
    ras = np.array(world[0])
    decs = np.array(world[1])

    coords = SkyCoord(ra=ras * u.deg, dec=decs * u.deg) #convert pixel to sky coords
    
    N = len(coords)
    max_sep = 0 * u.deg
    best_pair = None

    for i in range(N):
        seps = coords[i].separation(coords[i + 1:]) #ang seperation between all coords
        if len(seps) == 0:
            continue
        j = np.argmax(seps)
        candidate_sep = seps[j]
        cand_1 = coords[i]
        cand_2 = coords[i + 1:][j]
        
        
        p1x, p1y = wcs_cutout.all_world2pix(cand_1.ra.deg, cand_1.dec.deg, 0) 
        p2x, p2y = wcs_cutout.all_world2pix(cand_2.ra.deg, cand_2.dec.deg, 0)

        rr, cc = bresenham_line(
            int(round(float(p1y))), int(round(float(p1x))),
            int(round(float(p2y))), int(round(float(p2x)))
        ) #checks no nan pixels (that path is continuous) along path between pixels, so not calc width across gaps in source

        in_bounds = ((rr >= 0) & (rr < image.shape[0]) &
                     (cc >= 0) & (cc < image.shape[1]))
        if not np.all(in_bounds):
            continue

        vals = image[rr, cc]
        if not np.all(np.isfinite(vals)):
            continue

        if candidate_sep > max_sep:
            max_sep = candidate_sep
            best_pair = (cand_1, cand_2) #returns pair with largest valid angular seperation
            
    #convert coord separation to physical size
    
    theta_rad = max_sep.to(u.rad).value 
    width_kpc = (theta_rad * distance_kpc)
    
    return width_kpc, best_pair


def estimate_cross_section(width_m):
    
    # approximate cross-section as a circle
    
    r = width_m / 2.0
    
    return np.pi * r**2

def get_M500(row):
    """
    function to pick M500 from env cat, put in when cat columns changed. Prefers M500_bcg
    """
    
    m_est = row["M500_est"]
    m_bcg = row.get("M500_bcg", np.nan) #new env uses M500_bcg

    if np.isfinite(m_bcg) and m_bcg > 0:
        return m_bcg, "bcg"
    elif np.isfinite(m_est) and m_est > 0:
        return m_est, "est"
    else:
        return np.nan, None
        
def compute_forces(row):
    
    """
    what it does: computes ram pressure, jet power, sound speed, force from momentum flux (from sound speed and 0.4c)
    """
    
    z = row["z_best"]
    M500_raw, m_source = get_M500(row)

    if not np.isfinite(M500_raw):
        print(f"No valid M500 for {row['Source_Name']}")
        return (np.nan, np.nan, np.nan, np.nan)
  
    M500 = M500_raw / h
    size = row["Size"] / h  # convert to kpc/h
    dist = cosmo.angular_diameter_distance(z)

    # scale luminosity to jet power
    L = row["L_144"]
    L_to_P = 1e38 / 3e27  # Hardcastle+2018 Eq. 22 
    jet_power = L * L_to_P
    row['Jet_power_Hard [W]'] = jet_power

    
    # --------------------------------------------------
    # Cavagnolo et al. (2010)
    # --------------------------------------------------
    
    L144 = row["L_144"]   # W Hz^-1
    
    alpha = 0.5
    
    L1400 = L144 * (1400.0/144.0)**(-alpha)
    
    P1400 = L1400 / 1e24
    
    logQ = 0.75 * np.log10(P1400) + 1.91
    
    jet_power = (10**logQ) * 1e35   # W
    
    row['Jet_power_Cav [W]'] = jet_power
   
    # NFW-derived at outer radius 
    prof = profile_nfw.NFWProfile(M=M500, c=conc, z=z, mdef="vir")
    r = np.arange(0.1, size, 5)  # kpc/h
    m_enc = (prof.enclosedMass(r)) * (M_solar / h)  # convert to kg
    r_m = (r[-1] / h) * 3.0856776e19  # kpc→m
    g = (G * m_enc[-1]) / (r_m**2)

    # DPM halo
    M200 = myutils.lM500c_to_lM200c(np.log10(M500), z)
    R200 = myutils.R200c_from_lM200c(M200, z)
    modelID = "Model3"
    params = modelparams.returnmodelparams(modelID)
    model = dpmhalo.ModelMFlexGNFW(10**M200, z, *params)

    R = row["Size"] / R200  # dimensionless R/R200 using Size as proxy
    P500 = model.calcP(R, 10**M200, z)
    P = P500 * 1.38e-17  # K/cm^3 > Pa
    ne = model.calcne(R, 10**M200, z) * 1e6  # m^-3
    rho_ICM = 1.18 * ne * m_p
    gamma = 5 / 3
    v = np.sqrt((gamma * P) / rho_ICM)  # m/s
    P_ram = rho_ICM * v**2

    F_momentum = jet_power / v
    row['F_momentum'] = F_momentum
    F_m = jet_power / (0.4 * const.c.value)
    row['F_m'] = F_m

    return dict(
        g=g, P_ICM=P, rho_ICM=rho_ICM, P_ram=P_ram, v_lobe=v, F_momentum=F_momentum
    )



def compute_external_at_rkpc(row, r_kpc, z):
   

    M500_raw, m_source = get_M500(row) # calls function to get M500,  uses 'M500_bcg' if available

    if not np.isfinite(M500_raw):
        print(f"No valid M500 for {row['Source_Name']}")
        return ( np.nan,  np.nan, np.nan,  np.nan, np.nan)
    
   
    
    M500 = M500_raw / h #units required by subsequent code
    M200 = myutils.lM500c_to_lM200c(np.log10(M500), z)
    R200 = myutils.R200c_from_lM200c(M200, z)

    modelID = "Model3"
    params = modelparams.returnmodelparams(modelID)
    model = dpmhalo.ModelMFlexGNFW(10**M200, z, *params) #NFW profile given M200

    R_over_R200 = r_kpc / R200 #scale to ratio of R200
   # print('r_kpc / R200=', r_kpc / R200)
    

    P500_local = model.calcP(R_over_R200, 10**M200, z) # work out pressure at r_kpc
    P_local = P500_local * 1.38e-17  # K/cm^3 to Pa conversion

    ne_local = model.calcne(R_over_R200, 10**M200, z) * 1e6  # work out density at r_kpc and convert to m^-3
    rho_local = 1.18 * ne_local * m_p

    gamma = 5 / 3
    v_local = np.sqrt((gamma * P_local) / rho_local) #estimate velocity as sound speed
    P_ram_local = rho_local * v_local**2 #ram pressure at r_kpc (not actually used in later analysis) 

    return P_local, rho_local, v_local, P_ram_local,R_over_R200





def measure_region(seg_mask, wcs_cutout, image, dist, map_hdr,  beam_kpc, row, r_kpc, z,  x_pix, y_pix, xs, ys, 
                   nearest_ridge_idx, seg_id, k):
    
    """
    #function to work out deconvolved width, height, flux  and internal pressure of a region. Also calls compute_external_at_rkpc to measure external pressure locally.
    
    #-------------------------------
    #inputs: 
    #mask = region of interest
    #wcs_cutout = wcs of floodfilled fits cutout
    #image = fits cutout image
    #map_hdr = header of fits cutout
    #beam_kpc = beam size in kpc
    #row = row of input fits table for this source
    #r_kpc = distance from host galaxy  to middle of region in kpc
    #z = redshift (z_best column of input table)
    #vx =  pixel distance between x coordinate of host galaxy and end point of spline in the region
    #vy =  pixel distance between y coordinate of host galaxy and end point of spline in the region
    

    #-------------------------------
    #what it does:
    #1. select pixels in region
    #2. meaure width and height of region (convolved to begin with)
    #3. Convert from pixel to angular width and height
    #4. deconvolve region width and height from beam
    #5. Measure flux in Jy from Jy/beam map
    #6. Measure internal pressure via equipartition, kappa=kappa in pysynch
    #7.  call compute_external_at_rkpc to measure external pressure locally (in the same way as  I did for  the ends of jets before)

    #----------------------------
    #Returns:
    #P_int, P_ext, P_ram, flux, x_c, y_c, px, py,width_pix,height_pix

    #-------------------------------
    """
    #1. select pixels in region
 
    peak = np.nanmax(image)
    print("Peak Jy/beam:", peak)

    idxs = nearest_ridge_idx[seg_id == k]

    if len(idxs) < 2:
        return (np.nan,)*15
    
    
        
    sel = (seg_id == k)
    
    xs_sel = xs[sel]
    ys_sel = ys[sel]
   

    # centre
    x_c = np.mean(xs_sel)
    y_c = np.mean(ys_sel)
    
    # covariance matrix (shape)
    coords = np.vstack([xs_sel - x_c, ys_sel - y_c])
    cov = np.cov(coords)
    
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # sort: largest = major axis
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # major/minor directions
    vx, vy = eigvecs[:, 0]   # along jet (height)
    px, py = eigvecs[:, 1]   # across jet (width)
    
    # project points
    s = (xs_sel - x_c)*vx + (ys_sel - y_c)*vy
    t = (xs_sel - x_c)*px + (ys_sel - y_c)*py
    
    # robust extents
    s_lo, s_hi = np.percentile(s, [5, 95])
    t_lo, t_hi = np.percentile(t, [5, 95])
    
    height_pix = s_hi - s_lo
    width_pix  = t_hi - t_lo

        
    half_height = height_pix / 2
    half_width  = width_pix / 2
    
    inside = (abs(s) <= half_height) & (abs(t) <= half_width)


    # 3. Convert from pixel to angular width and height
    pixscale_as = abs(map_hdr['CDELT1']) * 3600.0
    width_as   = width_pix   * pixscale_as
    height_as = height_pix * pixscale_as

    # 4. deconvolve region width and height from beam
    bmaj_as = map_hdr['BMAJ'] * 3600.0
    bmin_as = map_hdr['BMIN'] * 3600.0
    
    
    width_as_deconv  = np.sqrt(max(width_as**2  - bmaj_as**2, 0.0))
    height_as_deconv = np.sqrt(max(height_as**2 - bmin_as**2, 0.0))
    

    
    if width_as_deconv>0 : #dont allow 0 width or height
        width=width_as_deconv
    else:
        width= width_as
    radius=width/2
    if height_as_deconv>0 :
        height=height_as_deconv
    else:
        height= height_as
    print("height arcsec", height, "radius arcsec", radius)
    
    #5 Measure flux in Jy from Jy/beam map

    pix_as = abs(map_hdr['CDELT1']) * 3600.0
    beam_area = bmaj_as * bmin_as * (np.pi / (4 * np.log(2)))
    pix_area = pix_as**2
    pix_per_beam = beam_area / pix_area
    flux_reg = (np.nansum(image[seg_mask]) /pix_per_beam)
    flux_cyl = np.nansum(image[ys_sel[inside], xs_sel[inside]]) / pix_per_beam

    print("flux full:", flux_reg)
    print("flux cyl :", flux_cyl)


    #6. Internal pressure via equipartition in pysynch
 
    print('radius=', radius)
    print('height=',height)
    
    # 6. Internal pressure via equipartition

    P_int_by_kappa = {}
    
    for kappa in KAPPA_VALUES:
    
        try:
            synch_out = SynchSource(
                type='cylinder',
                alcyl=height,
                arcyl=radius,
                gmin=10,
                gmax=1e5,
                injection=2.0,
                cosmology=cosmo,
                z=z,
                spectrum='powerlaw'
            )
    
            synch_out.normalize(
                144e6,
                flux_cyl,
                method='equipartition',
                brange=(1e-11, 1e-7),
                zeta=1 + kappa
            )
    
            P_int_by_kappa[kappa] = (
                synch_out.total_energy_density / 3.0
            )
    
        except Exception as e:
            print(f"SynchSource failed for kappa={kappa}:", e)
            P_int_by_kappa[kappa] = np.nan
    #7.  External/ram pressure locally
    print('r_kpc=', r_kpc)
    
    P_ext, rho_local, v_local, P_ram,R_over_R200 = compute_external_at_rkpc(row, r_kpc, z)
    print('P_int=',P_int_by_kappa)
    print('P_ext=',P_ext)
    
        
    return (P_int_by_kappa, P_ext, P_ram, flux_cyl,x_c, y_c, px, py, width_pix, height_pix,height, radius, width_as_deconv, height_as_deconv, R_over_R200)


def host_index_on_ridge(ra_host, dec_host, x_ridge, y_ridge):
    
    """
    works out where on ridge spline host galaxy is (in terms on index)
    """
    
    host = SkyCoord(ra_host * u.deg, dec_host * u.deg)
    ridge = SkyCoord(x_ridge * u.deg, y_ridge * u.deg)

    
    return np.argmin(host.separation(ridge))



def process_source(row, source_type, cache_dir, out_dir):
    """
    Main function that calls others
    """
    source = row["Source_Name"]
    print('on source', source)
    z = row["z_best"]
    ra=row['optRA'] #host galaxy ra in deg
    dec=row['optDec'] #host galaxy dec in deg
    dist = cosmo.angular_diameter_distance(z)

    phys = compute_forces(row)
    row['v_lobe ms-1'] = phys['v_lobe']
    row['rho_ICM'] = phys['rho_ICM']
    rho_ICM = phys['rho_ICM']
    g = phys['g']

    # Geometry: ridgeline & spline
    ridgeline = load_ridgeline(source)
    xnew_u, ynew_u, *_ = get_spline_cached(source, ridgeline, cache_dir)

    # Floodfill cutout and (cache)
    map_file, xnew_u, ynew_u = get_floodfill_cutout_cached(row, source, xnew_u, ynew_u, cache_dir)
   
    with fits.open(map_file) as imhdu:
        image = imhdu[0].data
        map_hdr = imhdu[0].header

    bmaj = map_hdr['BMAJ']
    beamsize = np.deg2rad(bmaj)            # rad
    beam_kpc = beamsize * dist.value * 1e3 # kpc

    wcs_cutout = WCS(map_hdr)
    height, width = image.shape

    # Convert ridge world coords to pixel coords
    coords_world = np.vstack([xnew_u, ynew_u]).T
    coords_pix = wcs_cutout.all_world2pix(coords_world, 0)
    x_pix = coords_pix[:, 0]
    y_pix = coords_pix[:, 1]
    ridge_pixels = np.vstack([x_pix, y_pix]).T
    idx_95 = int(len(x_pix) * 0.95)
    idx_5 = int(len(x_pix) * 0.05)

    pt95 = np.array([x_pix[idx_95], y_pix[idx_95]])
    prev_pt = np.array([x_pix[max(idx_95 - 1, 0)], y_pix[max(idx_95 - 1, 0)]])
    dx = pt95[0] - prev_pt[0]
    dy = pt95[1] - prev_pt[1]
    m_perp = -dx / dy if dy != 0 else np.inf
    x0, y0 = pt95

    pt5 = np.array([x_pix[idx_5], y_pix[idx_5]])
    prev_pt5 = np.array([x_pix[max(idx_5 - 1, 0)], y_pix[max(idx_5 - 1, 0)]])
    dx5 = pt5[0] - prev_pt5[0]
    dy5 = pt5[1] - prev_pt5[1]
    m_perp5 = -dx5 / dy5 if dy5 != 0 else np.inf
    x05, y05 = pt5

    ys = np.arange(height)
    xs = np.arange(width)
    X, Y = np.meshgrid(xs, ys)

    if np.isfinite(m_perp):
        perp_vals = m_perp * (X - x0) + y0
        mask95_above = (Y >= perp_vals)
        mask95_below = (Y <= perp_vals)
    else:
        mask95_above = (X >= x0)
        mask95_below = (X <= x0)
    mask95 = smaller_side_mask(mask95_above, mask95_below)

    if np.isfinite(m_perp5):
        perp_vals5 = m_perp5 * (X - x05) + y05
        mask5_above = (Y >= perp_vals5)
        mask5_below = (Y <= perp_vals5)
    else:
        mask5_above = (X >= x05)
        mask5_below = (X <= x05)
    mask5 = smaller_side_mask(mask5_above, mask5_below)

    area5 = np.sum(mask5)
    area95 = np.sum(mask95)
    if area5 < area95:
        mask = mask5
        chosen_end = "start 5%"
    else:
        mask = mask95
        chosen_end = "end 5%"
    print('using', chosen_end)

    finite_mask = np.isfinite(image)
    main_mask = mask & finite_mask

    ys_sel, xs_sel = np.nonzero(main_mask)
    selected_pixels = image[main_mask]

    if len(selected_pixels) <= 2:
        print('too few selected pixels for area, trying other end')
        if chosen_end == "end 5%":
            chosen_end = "start 5%"
            mask = mask5 & finite_mask
        else:
            chosen_end = "end 5%"
            mask = mask95 & finite_mask
        ys_sel, xs_sel = np.nonzero(mask)
        selected_pixels = image[mask]
        print('new end=', chosen_end)
    
    x_host, y_host = wcs_cutout.all_world2pix(ra, dec, 0)
    x_host = float(x_host)
    y_host = float(y_host)
    
    # Use ridge to define direction
    host_idx = host_index_on_ridge(ra, dec, xnew_u, ynew_u)
    
    def format_wcs_axes(ax):
        ax.coords[0].set_axislabel('RA (J2000)')
        ax.coords[1].set_axislabel('Dec (J2000)')
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')

    if chosen_end == "end 5%":
        end_idx = len(xnew_u) - 1
    else:
        end_idx = 0
    
    x_end, y_end = wcs_cutout.all_world2pix(
        xnew_u[end_idx], ynew_u[end_idx], 0
    )
        
   
   

#     # projection along jet axis
#     proj = (X - x_host)*vx + (Y - y_host)*vy #dot product to get distance along jet axis (neg = behind host)


# half_mask = finite_mask & (proj >= 0)

    width_kpc, best_pair = get_width(xs_sel, ys_sel, wcs_cutout, dist, image)

    # Try alternative end if invalid width
    if (not np.isfinite(width_kpc)) or width_kpc <= 0:
        print("Width invalid for first end, trying other end…")
        alt_mask = (mask5 if chosen_end == "end 5%" else mask95) & finite_mask
        ys_sel_alt, xs_sel_alt = np.nonzero(alt_mask)
        if len(xs_sel_alt) > 2:
            width_kpc_alt, best_pair_alt = get_width(xs_sel_alt, ys_sel_alt, wcs_cutout, dist, image)
            if np.isfinite(width_kpc_alt) and width_kpc_alt > 0:
                print("Using alternative end")
                width_kpc = width_kpc_alt
                best_pair = best_pair_alt
                xs_sel, ys_sel = xs_sel_alt, ys_sel_alt

    if (not np.isfinite(width_kpc)) or width_kpc <= 0:
        print("Both ends failed. Setting width and pressures to NaN.")
        row['width_kpc'] = np.nan
        row['width_kpc_deconvolved'] = np.nan
        row['width_m'] = np.nan
        row['Cross_sectional_A (m2)'] = np.nan
        row['F_expansion'] = np.nan
        row['P_int'] = np.nan
        return

    if width_kpc > beam_kpc:
        width_kpc_deconvolved = np.sqrt(width_kpc**2 - beam_kpc**2)
    else:
        width_kpc_deconvolved = width_kpc

    row['width_kpc'] = width_kpc
    row['width_kpc_deconvolved'] = width_kpc_deconvolved
    width_m = width_kpc_deconvolved * 3.086e19
    row['width_m'] = width_m
    area_m2 = estimate_cross_section(width_m)
    row['Cross_sectional_A (m2)'] = area_m2

    # Buoyancy and ram at ends
    V = (4 / 3) * np.pi * (width_m / 2)**3
    F_buoy = V * g * rho_ICM
    ram_pressure_force = phys["P_ram"] * area_m2
    row['P_ram'] = phys["P_ram"]
    row['F_ram'] = ram_pressure_force
    row['F_buoy'] = F_buoy
    
    #split jet into three regions after host galaxy on chosen side  for P_ratio calcs BASED ON RIDGE

   
    from scipy.spatial import cKDTree
    
    segmentation_mask = finite_mask.copy()
    
    ys, xs = np.nonzero(segmentation_mask)

    pixels = np.vstack([xs, ys]).T
    ridge  = np.vstack([x_pix, y_pix]).T
        
    pixel_step = np.sqrt(np.diff(x_pix)**2 + np.diff(y_pix)**2)
    total_length_pix = np.sum(pixel_step)

    tree = cKDTree(ridge)
    _, nearest_ridge_idx = tree.query(pixels)
    if end_idx > host_idx:
    # keep pixels AFTER host along ridge
        side_mask = nearest_ridge_idx >= host_idx
    else:
    # keep pixels BEFORE host along ridge
        side_mask = nearest_ridge_idx <= host_idx

    pixels = pixels[side_mask]
    nearest_ridge_idx = nearest_ridge_idx[side_mask]
    ys = ys[side_mask]
    xs = xs[side_mask]
        # renormalise between host and that end
    if end_idx > host_idx:
        u_raw = (nearest_ridge_idx - host_idx)
    else:
        u_raw = (host_idx - nearest_ridge_idx) 
        
    u_ridge = (u_raw - np.min(u_raw)) / (np.max(u_raw) - np.min(u_raw))
                                         
    seg_id = np.digitize(u_ridge, [1/3, 2/3], right=True) #digitize returns the indices of the bins to which each value in input array belongs, splits -Pnorm into thrids

    seg_masks = []

            
    for k in (0, 1, 2):
        seg_mask_k = np.zeros_like(image, dtype=bool)

        sel = (seg_id == k)
                
        seg_mask_k[ys[sel], xs[sel]] = True
        seg_masks.append(seg_mask_k)

   
    # Pixel to kpc scale 
    pixscale_rad = np.deg2rad(abs(map_hdr['CDELT1']))
    pix_to_kpc = pixscale_rad * dist.value * 1e3
    total_length_kpc = total_length_pix * pix_to_kpc
    
    centres = [None, None, None]
    widths = [None, None, None]
    dirs = [None, None, None]
    heights = [None, None, None] 
    
    widths_as_deconv = [None, None, None]
    heights_as_deconv = [None, None, None]
    

    
        
    seg_flux   = [np.nan]*3
    seg_h_as   = [np.nan]*3
    seg_r_as   = [np.nan]*3
    seg_r_kpc  = [np.nan]*3
    seg_Rr200  = [np.nan]*3
    
    seg_Pext   = [np.nan]*3
  
    # Compute Pint/Pext per segment
    for i, seg_mask in enumerate(seg_masks, start=1):
        k=i-1
        if np.sum(seg_mask) < 10:
            
           
            row[f'Pext{i}'] = np.nan

            continue

        # Representative distance from host = median projection (kpc)
       
        ys_reg, xs_reg = np.nonzero(seg_mask)
        
        x_med = np.nanmedian(xs_reg)
        y_med = np.nanmedian(ys_reg)
        
        r_kpc = np.nanmedian(u_ridge[seg_id == k])* total_length_kpc
        idxs = nearest_ridge_idx[seg_id == k]
    
        if len(idxs) < 2:
            continue
        
        # get local ridge positions
        x_r = x_pix[idxs]
        y_r = y_pix[idxs]
        
        # sort along ridge index
        order = np.argsort(idxs)
        x_r = x_r[order]
        y_r = y_r[order]
        
        # local direction from endpoints
        vx= x_r[-1] - x_r[0]
        vy= y_r[-1] - y_r[0]
        
        vnorm = np.hypot(vx, vy)
        vx/= vnorm
        vy /= vnorm
         
        P_int_by_kappa, P_ext, P_ram, flux,x_c, y_c, px, py, width_pix, height_pix,height_as, radius_as, width_as_deconv, height_as_deconv, R_over_R200= measure_region(seg_mask, wcs_cutout, image, dist, map_hdr,
               beam_kpc, row, r_kpc, z,
               x_pix, y_pix,
               xs, ys, nearest_ridge_idx, seg_id, k)

        
        seg_flux[k]  = flux
        seg_h_as[k]  = height_as
        seg_r_as[k]  = radius_as
        seg_r_kpc[k] = r_kpc
        seg_Rr200[k] = R_over_R200
       
        seg_Pext[k]  = P_ext
                    
        row[f'Pext{i}'] = P_ext
        
        widths_as_deconv[k] = width_as_deconv
        heights_as_deconv[k] = height_as_deconv

        centres[i-1] = (x_c, y_c)
        widths[i-1] = width_pix
        dirs[i-1] = (px, py)
        heights[i-1] = height_pix




        # dist is the angular-diameter distance in Mpc.
        arcsec_to_m = (
            dist.to(u.m).value / 206265.0 #(arcec to rad 180/pi x 3600)
        )

        radius_i_m = radius_as * arcsec_to_m
        height_i_m = height_as * arcsec_to_m
        width_kpc_i = 2.0 * radius_i_m / 3.0856776e19
        length_kpc_i = height_i_m / 3.0856776e19
        area_i_m2 = np.pi * radius_i_m**2

        # Volume of the cylindrical segment.
        volume_i_m3 = area_i_m2 * height_i_m

        P_ext_i, rho_local_i, v_local_i, P_ram_i, R200_ratio_i = (
            compute_external_at_rkpc(row, r_kpc, z)
        )

        M500_raw, _ = get_M500(row)
        
        for kappa, P_int_k in P_int_by_kappa.items():

            row[f'Pint{i}_k{kappa}'] = P_int_k
        
            row[f'Fexp{i}_k{kappa}'] = (
                P_int_k * area_i_m2
                if np.isfinite(P_int_k)
                else np.nan
            )
            
        if (
            np.isfinite(M500_raw)
            and np.isfinite(r_kpc)
            and r_kpc > 0
            and np.isfinite(rho_local_i)
            and rho_local_i > 0
        ):
            M500_i = M500_raw / h

            prof_i = profile_nfw.NFWProfile(
                M=M500_i,
                c=conc,
                z=z,
                mdef="vir"
            )

            # Colossus radius is in kpc/h.
            r_i_kpc_h = r_kpc * h

            # Enclosed mass in solar masses/h.
            m_enc_i = prof_i.enclosedMass(r_i_kpc_h)

            # Convert enclosed mass to kg.
            m_enc_i_kg = m_enc_i * M_solar / h

            # Physical radius in metres.
            r_i_m = r_kpc * 3.0856776e19

            # Local gravitational acceleration (m/s^2).
            g_i = G * m_enc_i_kg / r_i_m**2

           
            F_buoy_i = (
                volume_i_m3 * g_i * rho_local_i
            )

        
            F_ram_i = P_ram_i * area_i_m2
          
        else:
            g_i = np.nan
            F_buoy_i = np.nan
            F_ram_i = np.nan
           
       
        row[f'F_buoy_{i}'] = F_buoy_i
        row[f'P_ram_{i}'] = P_ram_i
        row[f'F_ram_{i}'] = F_ram_i
        
   
        row[f'g_{i}'] = g_i
        row[f'rho_ICM_{i}'] = rho_local_i
      
        row[f'V_segment_{i}'] = volume_i_m3
        row[f'A_segment_{i}'] = area_i_m2

        row[f'width_kpc_deconv_{i}'] = width_kpc_i
        row[f'length_kpc_{i}'] = length_kpc_i
                
        # MOMENTUM FLUX in outer SEGMENT 3, for p balance and using Pint/Pext
       
        
    try:

        V3 = row['V_segment_3']
        Pext3 = row['Pext3']
    
        if (
            np.isfinite(V3)
            and np.isfinite(Pext3)
            and V3 > 0
        ):
    
            v_adv = 0.015 * const.c.value
    
            r3_kpc = seg_r_kpc[2]
    
            source_length_m = r3_kpc * 3.0856776e19
    
            t_age = source_length_m / v_adv
    
            row['t_age'] = t_age
    
            E_lobe_bal = 4.0 * Pext3 * V3
    
            P_lobe_bal = E_lobe_bal / t_age
    
            Q_bal = row['Jet_power_Cav [W]'] - P_lobe_bal
    
            row['E_lobe_bal'] = E_lobe_bal
            row['P_lobe_bal'] = P_lobe_bal
    
            row['F_m_bal'] = (
                Q_bal / v_adv
                if Q_bal > 0
                else np.nan
            )
    
            Pint3 = row['Pint3_k0']
    
            if (
                np.isfinite(Pint3)
                and Pext3 > 0.
            ):
    
                Pi3 = Pint3 / Pext3
    
                row['PintPext3'] = Pi3
    
                E_lobe_meas = (
                    (3.0 * Pi3 + 1.0)
                    * Pext3
                    * V3
                )
    
                P_lobe_meas = E_lobe_meas / t_age
    
                Q_meas = (
                    row['Jet_power_Cav [W]']
                    - P_lobe_meas
                )
    
                row['E_lobe_meas'] = E_lobe_meas
                row['P_lobe_meas'] = P_lobe_meas
    
                row['F_m_meas'] = (
                    Q_meas / v_adv
                    if Q_meas > 0
                    else np.nan
                )
            else:
                row['PintPext3'] = np.nan
                row['E_lobe_meas'] = np.nan
                row['P_lobe_meas'] = np.nan
                row['F_m_meas'] = np.nan
    except Exception as e:
    
        print(
            "Momentum-flux correction failed:",
            row['Source_Name'],
            e
        )
# make_geometry_diagnostics(row=row,  source=source, out_dir=out_dir, image=image, wcs_cutout=wcs_cutout,xnew_u=xnew_u,
    #                           ynew_u=ynew_u,x_pix=x_pix, y_pix=y_pix, x_host=x_host, y_host=y_host, host_idx=host_idx,
    #                           end_idx=end_idx, x0=x0,  y0=y0,  m_perp=m_perp, x05=x05, y05=y05, m_perp5=m_perp5,   mask=mask,
    #                           xs_sel=xs_sel, ys_sel=ys_sel,seg_masks=seg_masks,u_ridge=u_ridge,xs=xs,ys=ys,best_pair=best_pair,
    #                           width_kpc_deconvolved=width_kpc_deconvolved,centres=centres,dirs=dirs,widths_as_deconv=widths_as_deconv,
    #                           heights_as_deconv=heights_as_deconv,) #make diagnostic plot



    # Global internal expansion pressure (unchanged)
    selected_sum = np.nansum(selected_pixels)
    bmaj_as = map_hdr['BMAJ'] * 3600.0
    bmin_as = map_hdr['BMIN'] * 3600.0
    pixscale_as = abs(map_hdr['CDELT1']) * 3600.0
    beam_area = bmaj_as * bmin_as * (np.pi / (4 * math.log(2)))
    pixel_area = pixscale_as**2
    pix_per_beam = beam_area / pixel_area
    flux = selected_sum / pix_per_beam

    kappa = 10
    synch_out = SynchSource(type='sphere', rsph=width_m / 2,
                            gmin=10, gmax=100000, injection=2.0, cosmology=cosmo, z=z)
    synch_out.normalize(144e6, flux, method='equipartition', brange=(1e-11, 1e-6), zeta=(1 + kappa))
    P_int = synch_out.total_energy_density / 3.0

    row['F_expansion'] = P_int * area_m2
    row['P_int'] = P_int
    row['Flux (Jy)'] = flux


#main

if __name__ == "__main__":
    in_fits = ('/beegfs/lofar/lclews/FRI_evolution/subclassification/inspection_results/new_env_final_classes_with_forces_and_pressures_model3.fits')
    with fits.open(in_fits) as hdu:
        data = hdu[1].data
    table = Table(data)

    # Create output dirs
    cache_dir = "/beegfs/lofar/lclews/FRI_evolution/subclassification/inspection_results/buoyancy/cache"
    out_dir = "/beegfs/lofar/lclews/FRI_evolution/subclassification/inspection_results/buoyancy"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Ensure required columns exist (do not overwrite existing values)
    base_cols_with_nan = [
        'Cross_sectional_A (m2)', 'Jet_power_Hard [W]','Jet_power_Cav [W]', 't_age', 'E_lobe_bal', 'P_lobe_bal', 'F_m_bal', 
        'Flux (Jy)', 'F_buoy', 'F_ram',
        'F_momentum', 'F_m', 'F_expansion', 'P_ram', 'P_int',
        'width_kpc', 'width_kpc_deconvolved', 'width_m', 'rho_ICM', 'v_lobe ms-1'
    ]
    for col in base_cols_with_nan:
        if col not in table.colnames:
            table[col] = np.full(len(table), np.nan)

    
    for kappa in KAPPA_VALUES:
        for i in range(1, 4):
    
            for col in [
                f'Pint{i}_k{kappa}',
                f'Fexp{i}_k{kappa}'
            ]:
    
                if col not in table.colnames:
                    table[col] = np.full(len(table), np.nan)

    new_cols = [
       
        'P_ram_1','P_ram_2','P_ram_3',
        'Pext1', 'Pext2', 'Pext3',
        'E_lobe_bal',
        'F_m_bal',
        't_age',
        'PintPext3',
        'E_lobe_meas',
        'P_lobe_meas',
        'F_m_meas',

        'F_buoy_1',
        'F_buoy_2',
        'F_buoy_3',
        
        'F_ram_1',
        'F_ram_2',
        'F_ram_3',
        'P_lobe_bal',

        'g_1',
        'g_2',
        'g_3',
        
        'width_kpc_deconv_1',
        'width_kpc_deconv_2',
        'width_kpc_deconv_3',

        'length_kpc_1',
        'length_kpc_2',
        'length_kpc_3',

    
        'V_segment_1',
        'V_segment_2',
        'V_segment_3',

        'A_segment_1',
        'A_segment_2',
        'A_segment_3',
        
        'rho_ICM_1',
        'rho_ICM_2',
        'rho_ICM_3',
         ]



    for col in new_cols:
        if col not in table.colnames:
            table[col] = np.full(len(table), np.nan)

    # Main loop
    for i, row in enumerate(table):
        
     
        
        # Skip already-processed rows 
        # if ('Pint/Pext1' in row.colnames and
        #     np.isfinite(row['Pint/Pext1']) and (row['Pint/Pext1'] != 0)):
        #     print(f"Skipping {row['Source_Name']}  already processed.")
        #     continue

        # Determine source type (Morph takes precedence)
        if row['Morph'] not in (None, '', '--'):
            source_type = row['Morph']
            print('classified as:', row['Morph'])
        else:
            print('classified as:', row['subclass_JC'])
            source_type = row['subclass_JC']

        if source_type not in ['unclassifiable']:
            process_source(row, source_type, cache_dir, out_dir)


    # # Save updated table
    out_fits = ( "/beegfs/lofar/lclews/FRI_evolution/subclassification/inspection_results/FRIs_envs_forces_geom.fits")
    
    table.write(out_fits, overwrite=True)
    print(f"Saved table to {out_fits}")
