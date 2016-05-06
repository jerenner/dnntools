"""
dnnplot.py

Plots tracks and results from DNN analyses.

1.  plot individual tracks:

    python dnnplot.py tracks <evt_start> <evt_end> <si/bg>

    -- Example: python dnnplot.py tracks 0 5 si
                (Plots tracks 0 through 4 in the vox_dnn3d_NEXT100_Paolina222_v10x10x10_r200x200x200_si.h5 dataset)

2.  plot run summary:

    python dnnplot.py summary

    -- Example: python dnnplot.py summary
                (Plots summary for run testrun)

3.  plot signal vs. background curve for given epoch:

    python dnnplot.py svsb <epoch>

    -- Example: python dnnplot.py svsb 5
                (Plots signal vs. background curve for epoch 5 of run testrun)

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from mpl_toolkits.mplot3d import Axes3D
from math import *
from dnninputs import *

grdcol = 0.99

# -----------------------------------------------------------------------------
# Get the arguments
# -----------------------------------------------------------------------------
usage_str = "Usage:\n\n python dnnplot.py <type> (<start>) (<end>) (si/bg)"
args = sys.argv

# Must have at least 2 arguments.
if(len(args) < 2):
    print usage_str
    exit();

# Get the run name and type of plot.
ptype = args[1]

evt_start = -1; evt_end = -1
epoch = -1; si_bg = "bg"
# If we are plotting tracks, get the start and end events.
if(ptype == "tracks"):
    evt_start = int(args[2])
    evt_end = int(args[3])
    si_bg = args[4]
# If we are plotting signal vs. background, get the epoch.
elif(ptype == "svsb"):
    epoch = int(args[2])
# Otherwise we should be plotting the summary.
elif(ptype != "summary"):
    print usage_str
    exit()

# -----------------------------------------------------------------------------
# File names and directories
# -----------------------------------------------------------------------------
fn_summary = "{0}/{1}/acc/accuracy_{2}.dat".format(rdir,rname,rname)
fn_svsb = "{0}/{1}/acc/prob_{2}_test_ep{3}.dat".format(rdir,rname,rname,epoch)

if(not os.path.isdir("{0}/{1}/plt".format(rdir,rname))): os.mkdir("{0}/{1}/plt".format(rdir,rname))
if(not os.path.isdir("{0}/plt".format(datdir))): os.mkdir("{0}/plt".format(datdir))
if(not os.path.isdir("{0}/plt/{1}".format(datdir,dname))): os.mkdir("{0}/plt/{1}".format(datdir,dname))

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

# Summary plot
if(ptype == "summary"):

    print "Plotting summary..."

    # Read in the results.
    accmat = np.loadtxt(fn_summary)
    acc_trs = accmat[:,1]*100.
    acc_trb = accmat[:,2]*100.
    acc_vls = accmat[:,5]*100.
    acc_vlb = accmat[:,6]*100.
    acc_itr = [] 
    for iit in range(len(acc_trs)): acc_itr.append(iit)

    # Plot the results.
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(15.0);

    ax1 = fig.add_subplot(121);
    ax1.plot(acc_itr, acc_trs, '-', color='blue', lw=1, label='Training (si)')
    ax1.plot(acc_itr, acc_trb, '-', color='green', lw=1, label='Training (bg)')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title("")
    ax1.set_ylim([0, 100]);
    #ax1.set_xscale('log')

    lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

    ax2 = fig.add_subplot(122);
    ax2.plot(acc_itr, acc_vls, '-', color='blue', lw=1, label='Validation (si)')
    ax2.plot(acc_itr, acc_vlb, '-', color='green', lw=1, label='Validation (bg)')
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("")
    ax2.set_ylim([0, 100]);
    #ax2.set_xscale('log')

    lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

    # Show and/or print the plot.
    fn_plt = "{0}/{1}/plt/{2}_summary.png".format(rdir,rname,rname)
    plt.savefig(fn_plt, bbox_inches='tight')
    if(plt_show):
        plt.show()
    plt.close()

# Signal vs. background curve
if(ptype == "svsb"):

    accmat = np.loadtxt(fn_svsb)
    acc_etype = accmat[:,0]
    acc_psi = accmat[:,1]
    acc_pbg = accmat[:,2]

    # Plot the results.
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(15.0);

    ax1.plot(acc_ep, acc_trb, '-.', color='red', lw=1)
    ax1.set_xlabel("Background rejection")
    ax1.set_ylabel("Signal efficiency")
    ax1.set_title("")

    #lnd = plt.legend(loc=4,frameon=False,handletextpad=0)

    # Show and/or print the plot.
    fn_plt = "{0}/{1}/plt/{2}_svsb_ep{3}.png".format(rdir,rname,rname,epoch)
    plt.savefig(fn_plt, bbox_inches='tight')
    if(plt_show):
        plt.show()
    plt.close()

# Tracks
if(ptype == "tracks"):

    print "Plotting tracks {0} to {1} of type {2}...".format(evt_start,evt_end,si_bg)

    evt_plt = evt_start
    while(evt_plt < evt_end):

        # 
        # Read in the track.
        h5f = h5py.File("{0}/{1}_{2}.h5".format(datdir,dname,si_bg),'r');
        trkmat = h5f['trk{0}'.format(evt_plt)];
        varr_x = trkmat[0]*vox_sizeX;
        varr_y = trkmat[1]*vox_sizeY;
        varr_z = trkmat[2]*vox_sizeZ;
        varr_c = trkmat[3];

        # Plot the 3D voxelized track.
        fig = plt.figure(1);
        fig.set_figheight(5.0);
        fig.set_figwidth(8.0);
    
        ax1 = fig.add_subplot(111,projection='3d');
        s1 = ax1.scatter(varr_x,varr_y,varr_z,marker='s',linewidth=0.5,s=2*vox_size,c=varr_c,cmap=plt.get_cmap('rainbow'),vmin=0.0,vmax=max(varr_c));
        s1.set_edgecolors = s1.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
        min_x = min(varr_x); max_x = max(varr_x)
        min_y = min(varr_y); max_y = max(varr_y)
        min_z = min(varr_z); max_z = max(varr_z)
        ax1.set_xlim([0.8*min_x, 1.25*max_x])
        ax1.set_ylim([0.8*min_y, 1.25*max_y])
        ax1.set_zlim([0.8*min_z, 1.25*max_z])
    #    ax1.set_xlim([0, 2 * vox_ext]);
    #    ax1.set_ylim([0, 2 * vox_ext]);
    #    ax1.set_zlim([0, 2 * vox_ext]);
        ax1.set_xlabel("x (mm)");
        ax1.set_ylabel("y (mm)");
        ax1.set_zlabel("z (mm)");
        ax1.set_title("");

        lb_x = ax1.get_xticklabels();
        lb_y = ax1.get_yticklabels();
        lb_z = ax1.get_zticklabels();
        for lb in (lb_x + lb_y + lb_z):
            lb.set_fontsize(8);
    
        ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax1.w_xaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
        ax1.w_yaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
        ax1.w_zaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    
        cb1 = plt.colorbar(s1);
        cb1.set_label('Energy (keV)');

        if(not plt_show):
            fn_plt = "{0}/plt/{1}/plt3d_{2}_{3}_{4}.{5}".format(datdir,dname,dname,evt_plt,si_bg,plt_imgtype);
            print "-- Writing plot {0}".format(fn_plt)
            plt.savefig(fn_plt, bbox_inches='tight');
        if(plt_show):
            plt.show();

        plt.close()

        min_dim = min(min_x,min_y,min_z); max_dim = max(max_x,max_y,max_z)
        len_dim = (max_dim - min_dim)
        min_dim -= 0.1*len_dim; max_dim += 0.1*len_dim; len_dim *= 1.2   # widen the range by 20%
        pltsize = 14.0
        fwidth = 3.6*pltsize; fheight = pltsize
        vsize = 38.0*vox_size*pltsize/len_dim
        print "Voxel size is {0}; len_dim = {1}; min_dim = {2}, max_dim = {3}".format(vsize,len_dim,min_dim,max_dim) 

        # Plot the projections.
        fig = plt.figure(2);
        fig.set_figheight(fheight);
        fig.set_figwidth(fwidth);

        # Create the x-y projection.
        ax2 = fig.add_subplot(131);
        ax2.plot(varr_x,varr_y,marker='s',markersize=vsize,linewidth=0,color='black');
        ax2.set_xlabel("x (mm)");
        ax2.set_ylabel("y (mm)");
        ax2.set_xlim([0.8*min_dim, 1.25*max_dim]);
        ax2.set_ylim([0.8*min_dim, 1.25*max_dim]);

        # Create the y-z projection.
        ax3 = fig.add_subplot(132);
        ax3.plot(varr_y,varr_z,marker='s',markersize=vsize,linewidth=0,color='black');
        ax3.set_xlabel("y (mm)");
        ax3.set_ylabel("z (mm)");
        ax3.set_xlim([0.8*min_dim, 1.25*max_dim]);
        ax3.set_ylim([0.8*min_dim, 1.25*max_dim]);    

        # Create the x-z projection.
        ax4 = fig.add_subplot(133);
        ax4.plot(varr_x,varr_z,marker='s',markersize=vsize,linewidth=0,color='black');
        ax4.set_xlabel("x (mm)");
        ax4.set_ylabel("z (mm)");
        ax4.set_xlim([0.8*min_dim, 1.25*max_dim]);
        ax4.set_ylim([0.8*min_dim, 1.25*max_dim]);
    
        # Show and/or print the plot.
        if(not plt_show):
            fn_plt = "{0}/plt/{1}/plt_{2}_{3}_{4}.{5}".format(datdir,dname,dname,evt_plt,si_bg,plt_imgtype);
            print "-- Writing plot {0}".format(fn_plt)
            plt.savefig(fn_plt, bbox_inches='tight');
        if(plt_show):
            plt.show();
    
        plt.close();

        # 2D histogram
        fig = plt.figure(3);
        fig.set_figheight(5.0);
        fig.set_figwidth(20.0);

        # Create the x-y projection.
        ax1 = fig.add_subplot(131);        
        hxy, xxy, yxy = np.histogram2d(varr_y, varr_x, weights=varr_c, normed=False, bins=(len_dim/vox_sizeY, len_dim/vox_sizeX), range=[[min_dim,max_dim],[min_dim,max_dim]])
        extent1 = [yxy[0], yxy[-1], xxy[0], xxy[-1]]
        sp1 = ax1.imshow(hxy, extent=extent1, interpolation='none', aspect='auto', origin='lower')
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("y (mm)")
        cbp1 = plt.colorbar(sp1);
        cbp1.set_label('Energy (keV)');

        # Create the y-z projection.
        ax2 = fig.add_subplot(132);
        hyz, xyz, yyz = np.histogram2d(varr_z, varr_y, weights=varr_c, normed=False, bins=(len_dim/vox_sizeZ, len_dim/vox_sizeY), range=[[min_dim,max_dim],[min_dim,max_dim]])
        extent2 = [yyz[0], yyz[-1], xyz[0], xyz[-1]]
        sp2 = ax2.imshow(hyz, extent=extent2, interpolation='none', aspect='auto', origin='lower')
        ax2.set_xlabel("y (mm)")
        ax2.set_ylabel("z (mm)")
        cbp2 = plt.colorbar(sp2);
        cbp2.set_label('Energy (keV)');

        # Create the x-z projection.
        ax3 = fig.add_subplot(133);
        hxz, xxz, yxz = np.histogram2d(varr_z, varr_x, weights=varr_c, normed=False, bins=(len_dim/vox_sizeZ, len_dim/vox_sizeX), range=[[min_dim,max_dim],[min_dim,max_dim]])
        extent3 = [yxz[0], yxz[-1], xxz[0], xxz[-1]]
        sp3 = ax3.imshow(hxz, extent=extent3, interpolation='none', aspect='auto', origin='lower')
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("z (mm)")
        cbp3 = plt.colorbar(sp3);
        cbp3.set_label('Energy (keV)'); 

        # Show and/or print the plot.
        if(not plt_show):
            fn_plt = "{0}/plt/{1}/plt_h2D_{2}_{3}_{4}.{5}".format(datdir,dname,dname,evt_plt,si_bg,plt_imgtype);
            print "-- Writing plot {0}".format(fn_plt)
            plt.savefig(fn_plt, bbox_inches='tight');
        if(plt_show):
            plt.show();

        plt.close();

        evt_plt += 1
