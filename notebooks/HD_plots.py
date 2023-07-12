from __future__ import division

import os

from matplotlib.pyplot import *

from matplotlib.mlab import *

import pyPLUTO as pp

from pyPLUTO import pload

from pyPLUTO import Image

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

D = pload.pload(2638, datatype = 'hdf5' , level = 3)

D7 = pload.pload(2638, datatype = 'hdf5' , level = 7)

I = Image.Image()

gamma = 1 / np.sqrt(1 - (D.vx1**2 + D.vx2**2))

def pldisplay(self, D, var,**kwargs):
    """ This method allows the user to display a 2D data using the
         matplotlib's pcolormesh.

    **Inputs:**
        D   -- pyPLUTO pload object.\n
        var -- 2D array that needs to be displayed.
    *Required Keywords:*
        x1 -- The 'x' array\n
        x2 -- The 'y' array
    *Optional Keywords:*
        vmin -- The minimum value of the 2D array (Default : min(var))\n
        vmax -- The maximum value of the 2D array (Default : max(var))\n
        title -- Sets the title of the image.\n
        label1 -- Sets the X Label (Default: 'XLabel')\n
        label2 -- Sets the Y Label (Default: 'YLabel')\n
        polar -- A list to project Polar data on Cartesian Grid.\n
            polar = [True, True] -- Projects r-phi plane.\n
            polar = [True, False] -- Project r-theta plane.\n
            polar = [False, False] -- No polar plot (Default)\n
        cbar -- Its a tuple to set the colorbar on or off. \n
            cbar = (True,'vertical') -- Displays a vertical colorbar\n
            cbar = (True,'horizontal') -- Displays a horizontal colorbar\n
            cbar = (False,'') -- Displays no colorbar.
    **Usage:**
        ``import pyPLUTO as pp``\n
        ``wdir = '/path/to/the data files/'``\n
        ``D = pp.pload(1,w_dir=wdir)``\n
        ``I = pp.Image()``\n
        ``I.pldisplay(D, D.v2, x1=D.x1, x2=D.x2, cbar=(True,'vertical'),\
        title='Velocity',label1='Radius',label2='Height')``
    """

    x1 = kwargs.get('x1')
    x2 = kwargs.get('x2')
    var = var.T

    f1 = figure(kwargs.get('fignum',1), figsize=kwargs.get('figsize',[10,10]),
                                 dpi=80, facecolor='w', edgecolor='k')
    ax1 = f1.add_subplot(111)
    ax1.set_aspect('equal')

    if kwargs.get('polar',[False,False])[0]:
        xx, yy = self.getPolarData(D,kwargs.get('x2'),rphi=kwargs.get('polar')[1])
        pcolormesh(xx,yy,var,vmin=kwargs.get('vmin',np.min(var)),vmax=kwargs.get('vmax',np.max(var)))
    else:
        ax1.axis([np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
        pcolormesh(x1,x2,var,vmin=kwargs.get('vmin',np.min(var)),vmax=kwargs.get('vmax',np.max(var)))

    title(kwargs.get('title',"Title"),size=kwargs.get('size'))
    xlabel(kwargs.get('label1',"Xlabel"),size=kwargs.get('size'))
    ylabel(kwargs.get('label2',"Ylabel"),size=kwargs.get('size'))
    if kwargs.get('cbar',(False,''))[0] == True:
        colorbar(orientation=kwargs.get('cbar')[1])
        
def getPolarData(Data, ang_coord, rphi=False):
    """To get the Cartesian Co-ordinates from Polar.
    **Inputs:**
        Data -- pyPLUTO pload Object\n
        ang_coord -- The Angular co-ordinate (theta or Phi)
    *Optional Keywords:*
        rphi -- Default value FALSE is for R-THETA data,
        Set TRUE for R-PHI data.\n
    **Outputs**:
        2D Arrays of X, Y from the Radius and Angular co-ordinates.\n
        They are used in pcolormesh in the Image.pldisplay functions.
    """
    D = Data
    if ang_coord is D.x2:
        x2r = D.x2r
    elif ang_coord is D.x3:
        x2r = D.x3r
    else:
        print("Angular co-ordinate must be given")

    rcos = np.outer(np.cos(x2r), D.x1r)
    rsin = np.outer(np.sin(x2r), D.x1r)

    if rphi:
        xx = rcos
        yy = rsin
    else:
        xx = rsin
        yy = rcos

    return xx, yy

def oplotbox(AMRLevel, lrange=[0,0], cval=['b','r','g','m','w','k'],\
                                    islice=-1, jslice=-1, kslice=-1,geom='CARTESIAN'):
    """
    This method overplots the AMR boxes up to the specified level.
    **Input:**
        AMRLevel -- AMR object loaded during the reading and stored in the pload object
    *Optional Keywords:*
        lrange     -- [level_min,level_max] to be overplotted. By default it shows all the loaded levels\n
        cval       -- list of colors for the levels to be overplotted.\n
        [ijk]slice -- Index of the 2D slice to look for so that the adequate box limits are plotted.
                                    By default oplotbox considers you are plotting a 2D slice of the z=min(x3) plane.\n
        geom       -- Specified the geometry. Currently, CARTESIAN (default) and POLAR geometries are handled.
    """

    nlev = len(AMRLevel)
    lrange[1] = min(lrange[1],nlev-1)
    npl  = lrange[1]-lrange[0]+1
    lpls = [lrange[0]+v for v in range(npl)]
    cols = cval[0:nlev]
    # Get the offset and the type of slice
    Slice = 0 ; inds = 'k'
    xx = 'x' ; yy ='y'
    if (islice >= 0):
            Slice = islice + AMRLevel[0]['ibeg'] ; inds = 'i'
            xx = 'y' ; yy ='z'
    if (jslice >= 0):
            Slice = jslice + AMRLevel[0]['jbeg'] ; inds = 'j'
            xx = 'x' ; yy ='z'
    if (kslice >= 0):
            Slice = kslice + AMRLevel[0]['kbeg'] ; inds = 'k'
            xx = 'x' ; yy ='y'

    # Overplot the boxes
    for il in lpls:
        level = AMRLevel[il]
        for ib in range(level['nbox']):
            box = level['box'][ib]
            if ((Slice-box[inds+'b'])*(box[inds+'e']-Slice) >= 0):
                if (geom == 'CARTESIAN'):
                    x0 = box[xx+'0'] ; x1 = box[xx+'1']
                    y0 = box[yy+'0'] ; y1 = box[yy+'1']
                    plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color=cols[il])
                elif (geom == 'POLAR') or (geom == 'SPHERICAL'):
                    dn = np.pi/50.
                    x0 = box[xx+'0'] ; x1 = box[xx+'1']
                    y0 = box[yy+'0'] ; y1 = box[yy+'1']
                    if y0 == y1:
                        y1 = 2*np.pi+y0 - 1.e-3
                    xb = np.concatenate([
                                    [x0*np.cos(y0),x1*np.cos(y0)],\
                                    x1*np.cos(np.linspace(y0,y1,num=int(abs(y0-y1)/dn) )),\
                                    [x1*np.cos(y1),x0*np.cos(y1)],\
                                    x0*np.cos(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                    yb = np.concatenate([
                                    [x0*np.sin(y0),x1*np.sin(y0)],\
                                    x1*np.sin(np.linspace(y0,y1,num=int(abs(y0-y1)/dn))),\
                                    [x1*np.sin(y1),x0*np.sin(y1)],\
                                    x0*np.sin(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                    plot(xb,yb,color=cols[il])
                    
#plt.rcParams.update({'font.size': 50})

#font_size = 50

#mpl.rcParams.update(mpl.rcParamsDefault)

polar = [True,True]

x1 = D.x1

x2 = D.x2

AMRLevel = D7.AMRLevel

lrange = [0,6]

cval=['c','m','b','r','g','y','k']

geom = 'POLAR'

var_rho = np.log10(D.rho).T

var_gamma = np.log10(gamma).T

fig1 = plt.figure()

ax1 = fig.add_subplot(111)

fig2 = plt.figure()
                    
ax2 = fig.add_subplot(111)

fig3 = plt.figure()
                    
ax3 = fig.add_subplot(111)

# fig.set_figheight(13)

# fig.set_figwidth(18)

xx, yy = getPolarData(D,x2,rphi=polar[1])

ax1.pcolormesh(xx,yy,var_rho,vmin=np.min(var_rho),vmax=np.max(var_rho), cmap = 'magma')

mapp0 = ax1.pcolormesh(xx,yy,var_rho,vmin=np.min(var_rho),vmax=np.max(var_rho), cmap = 'magma')

cb0 = plt.colorbar(mappable = mapp0, ax = ax1)

#cb0.ax.tick_params(labelsize=font_size)
cb0.set_label(r'log$_{10}(\rho)$')

ax1.set_xlim(0,20000)

ax1.set_ylim(0,20000)

ax1.set_title('Density Profile')

ax1.set_xlabel(r'Jet Axis ($\times 10^9$ cm)')

ax1.set_ylabel(r'x-Axis ($\times 10^9$ cm)')

ax1.set_aspect('equal')

plt.savefig('HD_frame_rho_cb.pdf', dpi = 300) 

ax2.pcolormesh(xx,yy,var_gamma,vmin=np.min(var_gamma),vmax=np.max(var_gamma), cmap = 'magma')

mapp1 = ax2.pcolormesh(xx,yy,var_gamma,vmin=np.min(var_gamma),vmax=np.max(var_gamma), cmap = 'magma')

cb1 = plt.colorbar(mappable = mapp1, ax = ax2)

#cb1.ax.tick_params(labelsize=font_size)
cb1.set_label(r'log$_{10}(\Gamma)$')

ax2.set_xlim(0,20000)

ax2.set_ylim(0,20000)

ax2.set_title('Bulk Lorentz Factor Profile')

ax2.set_xlabel(r'Jet Axis ($\times 10^9$ cm)')

ax2.set_ylabel(r'x-Axis ($\times 10^9$ cm)')

ax2.set_aspect('equal')

plt.savefig('HD_frame_gamma_cb.pdf', dpi = 300) 

nlev = len(AMRLevel)
lrange[1] = min(lrange[1],nlev-1)
npl  = lrange[1]-lrange[0]+1
lpls = [lrange[0]+v for v in range(npl)]
cols = cval[0:nlev]
    # Get the offset and the type of slice
Slice = 0 ; inds = 'k'
xx = 'x' ; yy ='y'

    # Overplot the boxes
for il in lpls:
    level = AMRLevel[il]
    for ib in range(level['nbox']):
        box = level['box'][ib]
        if ((Slice-box[inds+'b'])*(box[inds+'e']-Slice) >= 0):
            if (geom == 'CARTESIAN'):
                x0 = box[xx+'0'] ; x1 = box[xx+'1']
                y0 = box[yy+'0'] ; y1 = box[yy+'1']
                plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color=cols[il])
            elif (geom == 'POLAR') or (geom == 'SPHERICAL'):
                dn = np.pi/50.
                x0 = box[xx+'0'] ; x1 = box[xx+'1']
                y0 = box[yy+'0'] ; y1 = box[yy+'1']
                if y0 == y1:
                    y1 = 2*np.pi+y0 - 1.e-3
                xb = np.concatenate([
                                [x0*np.cos(y0),x1*np.cos(y0)],\
                                x1*np.cos(np.linspace(y0,y1,num=int(abs(y0-y1)/dn) )),\
                                [x1*np.cos(y1),x0*np.cos(y1)],\
                                x0*np.cos(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                yb = np.concatenate([
                                [x0*np.sin(y0),x1*np.sin(y0)],\
                                x1*np.sin(np.linspace(y0,y1,num=int(abs(y0-y1)/dn))),\
                                [x1*np.sin(y1),x0*np.sin(y1)],\
                                x0*np.sin(np.linspace(y1,y0,num=int(abs(y0-y1)/dn)))])
                ax3.plot(xb,yb,color=cols[il])
                
ax3.set_xlim(0,20000)

ax3.set_ylim(0,20000)

ax3.set_title('AMR Grid')

ax3.set_xlabel(r'Jet Axis ($\times 10^9$ cm)')

ax3.set_ylabel(r'x-Axis ($\times 10^9$ cm)')

ax3.set_aspect('equal')

fig.tight_layout()



plt.savefig('HD_frame_AMR_grid_cb.pdf', dpi = 300)

