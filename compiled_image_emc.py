import os
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import atomap.api as am
import trackpy as tp
import seaborn as sns
import gc
import pandas as pd
import skimage.restoration
import random

from tqdm import tqdm
from glob import glob
from scipy.signal import wiener, medfilt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Rectangle
from skimage.transform import match_histograms, resize


import warnings
warnings.filterwarnings("ignore")

# %matplotlib

def nullfunc(a,b):
    return ''

def ax_off(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# rebuild filtered image from decomposition model
def template_rebuild(input_im, pos_int_pass, decomp_model, twidth, name, med_filt=False, gfilt=0):
    output_im = input_im.deepcopy()
    output_arr = np.zeros(input_im.data.shape, input_im.data.dtype)
    norm_arr = output_arr.copy()
    mask = gauss2Dmask(twidth, (twidth/2)**2)
    decomp_arr = decomp_model.data
    for i in tqdm(range(0, decomp_model.data.shape[0])):
        pos = pos_int_pass[i]
        output_arr[pos[1] - twidth: pos[1] + twidth + 1, pos[0] - twidth: pos[0] + twidth + 1] += decomp_arr[i]
        norm_arr[pos[1] - twidth: pos[1] + twidth + 1, pos[0] - twidth: pos[0] + twidth + 1] += mask
    output_arr = np.nan_to_num(output_arr / norm_arr)
    if med_filt == True:
        output_arr = medfilt(output_arr, 3)
    output_arr = gaussian_filter(output_arr, gfilt)
    output_im.metadata.General.title = name
    output_im.data = output_arr
    return output_im

def highpass(image, minrad):
    outim = image.deepcopy()
    fft = np.fft.fftshift(np.fft.fft2(image.data))
    X, Y = np.mgrid[0:image.data.shape[0], 0:image.data.shape[1]]
    dists = np.sqrt((X - image.data.shape[0]/2)**2 + (Y - image.data.shape[1]/2)**2)
    mask = dists > minrad
    cutfft = fft * mask
    plt.imshow(np.log(np.abs(cutfft)))
    rebuilt = np.abs(np.fft.ifft2(np.fft.ifftshift(cutfft)))
    outim.data = rebuilt
    return outim

def lopass(image, minrad):
    outim = image.deepcopy()
    fft = np.fft.fftshift(np.fft.fft2(image.data))
    X, Y = np.mgrid[0:image.data.shape[0], 0:image.data.shape[1]]
    dists = np.sqrt((X - image.data.shape[0]/2)**2 + (Y - image.data.shape[1]/2)**2)
    mask = dists < minrad
    cutfft = fft * mask
    plt.imshow(np.log(np.abs(cutfft)))
    rebuilt = np.abs(np.fft.ifft2(np.fft.ifftshift(cutfft)))
    outim.data = rebuilt
    return outim

# gaussian 2D mask
def gauss2Dmask(tempwidth, sigma, normed = False):
    import numpy as np
    prefac = 1 #/ np.sqrt(2 * np.pi * sigma)
    if normed == True:
        prefac = 1 / np.sqrt(2 * np.pi * sigma)
    X, Y =  np.meshgrid(range(0, tempwidth*2 + 1), range(0, tempwidth*2 + 1))
    distX, distY = X - tempwidth, Y - tempwidth
    dist2 = (distX**2) + (distY**2)
    exp = np.exp2(-dist2/(2*sigma))
    return prefac * exp

# create template image for decomposition
def round_templates(segment, tempwidth, pos_int):
    templates = []
    pos_int_pass = []
    mask = gauss2Dmask(tempwidth, (tempwidth/2)**2)
    for pos in tqdm(pos_int[:]):
        template = segment.data[pos[1] - tempwidth: pos[1] + tempwidth + 1, pos[0] - tempwidth: pos[0] + tempwidth + 1]
        if template.shape == (tempwidth * 2 + 1, tempwidth *2 + 1):
            templates.append(template * mask)
            pos_int_pass.append(pos)
    template_arr = np.array(templates)
    template_im = hs.signals.Signal2D(template_arr)
    return template_im, pos_int_pass

# create template image for decomposition
def round_fft_templates(segment, tempwidth, pos_int):
    templates = []
    pos_int_pass = []
    mask = gauss2Dmask(tempwidth, (tempwidth/2)**2)
    for pos in tqdm(pos_int[:]):
        template = segment.data[pos[1] - tempwidth: pos[1] + tempwidth + 1, pos[0] - tempwidth: pos[0] + tempwidth + 1]
        if template.shape == (tempwidth * 2 + 1, tempwidth *2 + 1):
            outpart = template * mask
            fft = np.fft.fftshift(np.fft.fft2(outpart))
            mag_spec = np.log(np.abs(fft))
            templates.append(mag_spec)
            pos_int_pass.append(pos)
    template_arr = np.array(templates)
    template_im = hs.signals.Signal2D(template_arr)
    return template_im, pos_int_pass

# save decomposition rebuilt image and output png
def rebuiltsave(folder, hsimage, framecut=30):
    hsimage.save(folder + os.sep + hsimage.metadata.General.title + '.hspy', overwrite=True)
    hsimage.save(folder + os.sep + hsimage.metadata.General.title + '.tiff', overwrite=True)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(hsimage.data[framecut:-framecut, framecut:-framecut], interpolation='hermite', cmap='inferno')
    scalebar = ScaleBar(hsimage.axes_manager[0].scale, hsimage.axes_manager[0].units, color='white', box_alpha=0)
    ax.add_artist(scalebar)
    fig.savefig(folder + os.sep + hsimage.metadata.General.title + '.png', overwrite=True, dpi=hsimage.axes_manager[0].size/10)
    fig.savefig(folder + os.sep + hsimage.metadata.General.title + '.svg', overwrite=True, dpi=hsimage.axes_manager[0].size/10)
    plt.close(fig=fig)
    
# save numpy image with scalebar
def imsave(folder, title ,npimage, scale_dx, scale_units):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(npimage, interpolation='hermite', cmap='inferno')
    if scale_dx != 0:
        scalebar = ScaleBar(scale_dx, scale_units, color='white', box_alpha=0)
        ax.add_artist(scalebar)
    fig.savefig(folder + os.sep + title + '.png', dpi=npimage.data.shape[0]/10, overwrite=True)
    plt.close(fig=fig)

def fakegpa(hsimage, savefolder, aprad, gpaname, distcut=180):
    imscale = hsimage.axes_manager[0].scale
    imscale_units = hsimage.axes_manager[0].units
    from scipy.signal import medfilt
    gpafolder = savefolder + os.sep + 'fakegpa_' + gpaname
    if not os.path.exists(gpafolder):
        os.mkdir(gpafolder)
    data = hsimage.data
    fft = np.fft.fftshift(np.fft.fft2(data))
    mag_spec = np.log(np.abs(fft))
    peaks = tp.locate(medfilt(mag_spec, 5), 81, topn=25)
    peaks['rot_angle'] = np.rad2deg(np.arctan2(peaks.x - data.shape[0]/2,peaks.y - data.shape[1]/2))
    peaks['dist'] = np.sqrt((peaks.x - data.shape[0]/2)**2 + (peaks.y - data.shape[1]/2)**2)
    peaks = peaks[peaks['rot_angle'] > 0]
    peaks = peaks[peaks['dist'] < distcut]
    peaks = peaks[peaks['dist'] > 2.0]
    peaks = peaks.reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mag_spec, cmap='inferno')
    ax.set_xlim(data.shape[0] / 4, 3 * data.shape[1] / 4)
    ax.set_ylim(data.shape[0] / 4, 3 * data.shape[1] / 4)
    for index, peak in peaks.iterrows():
        plotmarker = plt.Circle((peak.x, peak.y), radius=aprad, edgecolor='black', facecolor='None')
        ax.add_artist(plotmarker)
        plt.text(peak.x, peak.y + aprad + 10, index)
    fig.savefig(gpafolder + os.sep + 'fft_apertures.png')
    
    X, Y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    phases = []
    amps = []
    reals = []
    imags = []
    for index, peak in peaks.iterrows():
        dists = np.sqrt((X - peak.x)**2 + (Y - peak.y)**2)
        mask = dists < aprad
        rebuilt = np.fft.ifft2(np.fft.ifftshift(fft * mask))
        phase = np.angle(rebuilt)
        amp = np.abs(rebuilt)
        imsave(gpafolder, 'phase_' + str(index), phase, scale_dx=imscale, scale_units=imscale_units)
        imsave(gpafolder, 'amplitude_' + str(index), amp, scale_dx=imscale, scale_units=imscale_units)
        imsave(gpafolder, 'real_' +str(index), rebuilt.real, scale_dx=imscale, scale_units=imscale_units) 
        phases.append(phase)
        amps.append(amp)
        reals.append(rebuilt.real)
        imags.append(rebuilt.imag)
        
    phase_sum = np.sum(phases, axis=0)
    amps_sum = np.sum(amps, axis=0)
    reals_sum = np.sum(reals, axis=0)
    imsave(gpafolder, 'phase_sum', phase_sum, scale_dx=imscale, scale_units=imscale_units)
    imsave(gpafolder, 'amps_sum', amps_sum, scale_dx=imscale, scale_units=imscale_units) 
    imsave(gpafolder, 'reals_sum', reals_sum, scale_dx=imscale, scale_units=imscale_units) 

def templateICA_bss(template_im, N, pos_int_pass, savefolder, icaname='bss template ICA'):
    icafolder = savefolder + os.sep + icaname
    if not os.path.exists(icafolder):
        os.mkdir(icafolder)
    template_im.decomposition()
    template_im.blind_source_separation(N)
    loadings = template_im.get_bss_loadings()
    factors = template_im.get_bss_factors()
    pos_list = np.array(pos_int_pass)
    for i in range(0, N):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0, 0, 1, 1], aspect='equal')
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 2048)
        loading = loadings.inav[i]
        factor = factors.inav[i]
        ax.scatter(pos_list[:,0], pos_list[:,1], c=loading.data, cmap='bwr')
        ax_inset = fig.add_axes([0.86, 0.86, 0.13, 0.13])
        ax_inset.imshow(factor)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        fig.savefig(icafolder + os.sep + '%d.png' %i)

def slplot(sublattice, name, scale):
    imsize = sublattice.image.data.shape[0]
    screensize = 12
    fig = plt.figure(figsize=(screensize, screensize))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(sublattice.image, interpolation='hermite')
    ax.scatter(sublattice.atom_positions[:, 0], sublattice.atom_positions[:, 1], s=3, c='red')
    fig.savefig(savefolder + name + '.png', dpi = imsize * scale / screensize)
    fig.savefig(savefolder + name + '.svg', dpi = imsize * scale / screensize)

def templateICA_nmf(template_im, N, pos_int_pass, savefolder, icaname='bss template ICA'):
    icafolder = savefolder + os.sep + icaname
    if not os.path.exists(icafolder):
        os.mkdir(icafolder)
    template_im.decomposition(algorithm='nmf', output_dimension=N)
    loadings = template_im.get_decomposition_loadings()
    factors = template_im.get_decomposition_factors()
    pos_list = np.array(pos_int_pass)
    for i in range(0, N):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0, 0, 1, 1], aspect='equal')
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 2048)
        loading = loadings.inav[i]
        factor = factors.inav[i]
        ax.scatter(pos_list[:,0], pos_list[:,1], c=loading.data, cmap='bwr')
        ax_inset = fig.add_axes([0.86, 0.86, 0.13, 0.13])
        ax_inset.imshow(factor)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        fig.savefig(icafolder + os.sep + '%d.png' %i)

def export_decomposition_loading_ims(image, comp_ids, folder, scfactor=10):
    if not os.path.exists(folder):
        os.mkdir(folder)
    factors = image.get_decomposition_factors().inav[:comp_ids]
    facdata = factors.data
    for i in range(factors.data.shape[0]):
        image = facdata[i]
        plt.imsave(folder + '%s.png' % str(i).zfill(4), imresize(image, (scfactor*image.shape[0], scfactor*image.shape[1]), 'nearest'), cmap='inferno')

def export_bss_loading_ims(image, comp_ids, folder, scfactor=10):
    if not os.path.exists(folder):
        os.mkdir(folder)
    factors = image.get_bss_factors().inav[:comp_ids]
    facdata = factors.data
    for i in range(factors.data.shape[0]):
        image = facdata[i]
        plt.imsave(folder + '%s.png' % str(i).zfill(4), imresize(image, (scfactor*image.shape[0], scfactor*image.shape[1]), 'nearest'), cmap='inferno')

# load data and create savefolder

basefolder = r"C:\Users\Nick\Dropbox (The University of Manchester)\samples\astrid\TwistTMD Data\Nick Analysis\images to fit\EMC_PCA_highpass" + os.sep
settingslist = pd.read_csv(basefolder + 'settings_emc.csv', converters={'filename': lambda x: str(x)})

names = settingslist.filename.values

for fullname in names:
    
    name = fullname[:-4]
    
    print(name)

    settings = settingslist[settingslist.filename == name + '.dm3']
    filename = basefolder + name + '.dm3'

    filepath = filename.split(os.sep)
    filepath[0] += '\\'
    savefolder = os.path.join(*filepath[:-1], 'fig_outputs', name + '_output') + os.sep
    loadfolder = os.path.join(*filepath[:-1], 'outputs_v1', name + '_output') + os.sep

    if os.path.exists(loadfolder):

        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        input_im = hs.load(filename)
        input_im.plot()
        input_im.change_dtype('float64')

        # pixel size
        imscale = input_im.axes_manager[0].scale
        imscale_units = input_im.axes_manager[0].units

        segment = input_im.isig[:,:].deepcopy()

        segment_smooth = segment.deepcopy()
        segment_smooth.data = gaussian_filter(segment.data, settings.smoothrad.values[0])










        svd2_hi = hs.load(loadfolder + 'SVD_HI_2.hspy')
        svd2_med = hs.load(loadfolder + 'SVD_MED_2.hspy')
        svd2_low = hs.load(loadfolder + 'SVD_LO_2.hspy')


        bss2_hi = hs.load(loadfolder + 'BSS_2_HI.hspy')
        bss2_med = hs.load(loadfolder + 'BSS_2_MED.hspy')
        bss2_low = hs.load(loadfolder + 'BSS_2_LO.hspy')



        xlim_out = [600, 1000]
        ylim_out = [1000, 600]

        xlim_in = [800, 900]
        ylim_in = [900, 800]




        fig, ax = plt.subplots(2,4, figsize=(18,10))

        ax_off(ax[0][0])
        ax_off(ax[0][1])
        ax_off(ax[0][2])
        ax_off(ax[0][3])
        ax_off(ax[1][0])
        ax_off(ax[1][3])
        ax_off(ax[1][1])
        ax_off(ax[1][2])

        ax[1][0].spines['right'].set_visible(False)
        ax[1][0].spines['left'].set_visible(False)
        ax[1][0].spines['top'].set_visible(False)
        ax[1][0].spines['bottom'].set_visible(False)


        ax00 = ax[0][0].inset_axes([0.6, 0.6, 0.37, 0.37])
        ax01 = ax[0][1].inset_axes([0.6, 0.6, 0.37, 0.37])
        ax02 = ax[0][2].inset_axes([0.6, 0.6, 0.37, 0.37])
        ax03 = ax[0][3].inset_axes([0.6, 0.6, 0.37, 0.37])
        # ax10 = ax[1][0].inset_axes([0.6, 0.6, 0.37, 0.37])
        ax11 = ax[1][1].inset_axes([0.6, 0.6, 0.37, 0.37])
        ax12 = ax[1][2].inset_axes([0.6, 0.6, 0.37, 0.37])
        ax13 = ax[1][3].inset_axes([0.6, 0.6, 0.37, 0.37])


        ax_off(ax00)
        ax_off(ax01)
        ax_off(ax02)
        ax_off(ax03)
        ax_off(ax11)
        ax_off(ax12)
        ax_off(ax13)


        ax[0][0].imshow(input_im.data, interpolation='Hermite', cmap='inferno')
        ax[0][0].set_title('(a) Original Image')
        ax[0][0].set_xlim(xlim_out)
        ax[0][0].set_ylim(ylim_out)
        ax00.imshow(input_im.data, interpolation='Hermite', cmap='inferno')
        ax00.set_xlim(xlim_in)
        ax00.set_ylim(ylim_in)
        ax[0][0].indicate_inset_zoom(ax00, alpha=1)

        ax[0][1].imshow(match_histograms(svd2_hi.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[0][1].set_title('(b) SVD %d components' % (settings.svd2_hi.values[0]))
        ax[0][1].set_xlim(xlim_out)
        ax[0][1].set_ylim(ylim_out)
        ax01.imshow(match_histograms(svd2_hi.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax01.set_xlim(xlim_in)
        ax01.set_ylim(ylim_in)
        ax[0][1].indicate_inset_zoom(ax01, alpha=1)

        ax[0][2].imshow(match_histograms(svd2_med.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[0][2].set_title('(c) SVD %d components' % (settings.svd2_med.values[0]))
        ax[0][2].set_xlim(xlim_out)
        ax[0][2].set_ylim(ylim_out)
        ax02.imshow(match_histograms(svd2_med.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax02.set_xlim(xlim_in)
        ax02.set_ylim(ylim_in)
        ax[0][2].indicate_inset_zoom(ax02, alpha=1)

        ax[0][3].imshow(match_histograms(svd2_low.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[0][3].set_title('(d) SVD %d components' % (settings.svd2_lo.values[0]))
        ax[0][3].set_xlim(xlim_out)
        ax[0][3].set_ylim(ylim_out)
        ax03.imshow(match_histograms(svd2_low.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax03.set_xlim(xlim_in)
        ax03.set_ylim(ylim_in)
        ax[0][3].indicate_inset_zoom(ax03, alpha=1)

        ax[1][1].imshow(match_histograms(bss2_hi.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[1][1].set_title('(e) FastICA %d components' % (settings.bss2_hi.values[0]))
        ax[1][1].set_xlim(xlim_out)
        ax[1][1].set_ylim(ylim_out)
        ax11.imshow(match_histograms(bss2_hi.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax11.set_xlim(xlim_in)
        ax11.set_ylim(ylim_in)
        ax[1][1].indicate_inset_zoom(ax11, alpha=1)

        ax[1][2].imshow(match_histograms(bss2_med.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[1][2].set_title('(f) FastICA %d components' % (settings.bss2_med.values[0]))
        ax[1][2].set_xlim(xlim_out)
        ax[1][2].set_ylim(ylim_out)
        ax12.imshow(match_histograms(bss2_med.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax12.set_xlim(xlim_in)
        ax12.set_ylim(ylim_in)
        ax[1][2].indicate_inset_zoom(ax12, alpha=1)

        ax[1][3].imshow(match_histograms(bss2_low.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[1][3].set_title('(g) FastICA %d components' % (settings.bss2_lo.values[0]))
        ax[1][3].set_xlim(xlim_out)
        ax[1][3].set_ylim(ylim_out)
        ax13.imshow(match_histograms(bss2_low.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax13.set_xlim(xlim_in)
        ax13.set_ylim(ylim_in)
        ax[1][3].indicate_inset_zoom(ax13, alpha=1)

        fig.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0.02, hspace=0)

        fig.savefig(savefolder + 'component_compare.png', dpi=300)
        fig.savefig(savefolder + 'component_compare.svg', dpi=300)























        atom_lattice2_load = am.load_atom_lattice_from_hdf5(loadfolder + 'atom_lattice2_save.hdf5', construct_zone_axes=False)
        sublattice_2 = atom_lattice2_load.sublattice_list[0]

        pos_int_2 = np.round(sublattice_2.atom_positions).astype(int)
        tempwidth2 = settings.tempwidth2.values[0] # template size = 2 x tempwidth + 1
        template_im_2, pos_int_pass_2 = round_templates(segment, tempwidth2, pos_int_2)

        SVDfactors = hs.load(loadfolder + 'svd_2_eigenvalues' + os.sep + 'factors.hspy')

        BSSfactors = hs.load(loadfolder + 'bss_2_lo_eigenvalues' + os.sep + 'bss_factors.hspy')

        template_rebuilt_svd_lo_2 = hs.load(loadfolder + 'SVD_LO_2.hspy')
           
        template_rebuilt_bss_lo_2 = hs.load(loadfolder + 'BSS_2_lo.hspy')

        yoff = input_im.axes_manager[1].offset
        xoff = input_im.axes_manager[0].offset

        cropx = 846
        cropy = 808
        cropwidth = 150

        roi_x_1 = 722
        roi_y_1 = 839

        roi_x_2 = 923
        roi_y_2 = 812

        roi_x_1_real = xoff + imscale * roi_x_1
        roi_y_1_real = yoff + imscale * roi_y_1

        roi_x_2_real = xoff + imscale * roi_x_2
        roi_y_2_real = yoff + imscale * roi_y_2

        line_roi = hs.roi.Line2DROI(roi_x_1_real, roi_y_1_real, roi_x_2_real, roi_y_2_real, linewidth=1)

        origprofile = line_roi(input_im)
        bssprofile = line_roi(template_rebuilt_bss_lo_2)
        svdprofile = line_roi(template_rebuilt_svd_lo_2)

        bssprofile_diff = bssprofile - origprofile
        svdprofile_diff = svdprofile - origprofile

        profile_X = np.linspace(0, origprofile.axes_manager[0].size - 1, origprofile.axes_manager[0].size) * origprofile.axes_manager[0].scale

        crop_px = 16

        bss = template_rebuilt_bss_lo_2.data
        svd = template_rebuilt_svd_lo_2.data

        bss_diff = template_rebuilt_bss_lo_2.data - input_im.data
        svd_diff = template_rebuilt_svd_lo_2.data - input_im.data



        orig = input_im.deepcopy()

        fig = plt.figure(figsize=(12,6))

        ax1 = plt.subplot2grid((4,9), (0,0), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid((4,9), (0,3), rowspan=3, colspan=3)
        ax3 = plt.subplot2grid((4,9), (0,6), rowspan=3, colspan=3)

        profile1 = plt.subplot2grid((4,9), (3,0), colspan=3)
        profile2 = plt.subplot2grid((4,9), (3,3), colspan=3)
        profile3 = plt.subplot2grid((4,9), (3,6), colspan=3)

        ax1.imshow(input_im.data, cmap='inferno', interpolation='hermite', origin='lower')
        inset1 = ax1.inset_axes([0.6, 0.6, 0.37, 0.37])
        inset1.imshow(input_im.data, cmap='inferno', interpolation='hermite', origin='lower')
        lims = ax1.get_xlim()
        inset1.set_xlim(cropx-cropwidth, cropx+cropwidth)
        inset1.set_ylim(cropy-cropwidth, cropy+cropwidth)
        ax_off(ax1)
        ax_off(inset1)
        scalebar1 = ScaleBar(imscale, imscale_units, location='lower right', color='white', box_alpha=0)
        ax1.add_artist(scalebar1)
        scalebar1_ins = ScaleBar(imscale, imscale_units, location='lower right', color='white', box_alpha=0)
        inset1.add_artist(scalebar1_ins)
        ax1.indicate_inset_zoom(inset1, edgecolor='black')
        ax1.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        inset1.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_title(' (a) original', loc='left')

        profile1.plot(profile_X, origprofile.data, color='blue')
        profile1.set_xlabel('length (nm)')
        profile1.set_ylabel('Intensity (arb. units)')
        profile1.set_yticks([])
        profile1.set_xlim(profile_X[0], profile_X[-1])

        ax2.imshow(match_histograms(bss, input_im.data), cmap='inferno', interpolation='hermite', origin='lower')
        inset2 = ax2.inset_axes([0.6, 0.6, 0.37, 0.37])
        inset2.imshow(match_histograms(bss, input_im.data), cmap='inferno', interpolation='hermite', origin='lower')
        inset2.set_xlim(cropx-cropwidth, cropx+cropwidth)
        inset2.set_ylim(cropy-cropwidth, cropy+cropwidth)
        ax2.set_xticks([])
        ax2.set_yticks([])
        inset2.set_xticks([])
        inset2.set_yticks([])
        scalebar2 = ScaleBar(input_im.axes_manager[0].scale, input_im.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        ax2.add_artist(scalebar2)
        scalebar2_ins = ScaleBar(input_im.axes_manager[0].scale, input_im.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        inset2.add_artist(scalebar2_ins)
        ax2.indicate_inset_zoom(inset2, edgecolor='black')
        ax2.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        inset2.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_title(' (b) filtered', loc='left')

        profile2.plot(profile_X, bssprofile.data, color='blue')
        profile2.set_xlabel('length (nm)')
        profile2.set_yticks([])
        profile2.set_xlim(profile_X[0], profile_X[-1])

        ax3.imshow(match_histograms(bss_diff, input_im.data), cmap='inferno', origin='lower', interpolation='None')
        inset3 = ax3.inset_axes([0.6, 0.6, 0.37, 0.37])
        inset3.imshow(match_histograms(bss_diff, input_im.data), cmap='inferno', origin='lower', interpolation='None')
        inset3.set_xlim(cropx-cropwidth, cropx+cropwidth)
        inset3.set_ylim(cropy-cropwidth, cropy+cropwidth)
        ax3.set_xticks([])
        ax3.set_yticks([])
        inset3.set_xticks([])
        inset3.set_yticks([])
        scalebar3 = ScaleBar(orig.axes_manager[0].scale, orig.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        ax3.add_artist(scalebar3)
        scalebar3_ins = ScaleBar(orig.axes_manager[0].scale, orig.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        inset3.add_artist(scalebar3_ins)
        ax3.indicate_inset_zoom(inset3, edgecolor='black')
        ax3.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        inset3.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        ax3.set_xlim(lims)
        ax3.set_ylim(lims)
        ax3.set_title(' (c) difference', loc='left')

        profile3.plot(profile_X, bssprofile_diff, color='blue')
        profile3.set_xlabel('length (nm)')
        profile3.set_yticks([])
        profile3.set_xlim(profile_X[0], profile_X[-1])

        fig.subplots_adjust(top=0.98,
                            bottom=0.07,
                            left=0.02,
                            right=0.995,
                            hspace=0.0,
                            wspace=0.13)

        fig.savefig(savefolder + 'bss_filter_illustrative.png', dpi=400)
        fig.savefig(savefolder + 'bss_filter_illustrative.svg', dpi=400)









        fig = plt.figure(figsize=(12,6))

        ax1 = plt.subplot2grid((4,9), (0,0), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid((4,9), (0,3), rowspan=3, colspan=3)
        ax3 = plt.subplot2grid((4,9), (0,6), rowspan=3, colspan=3)

        profile1 = plt.subplot2grid((4,9), (3,0), colspan=3)
        profile2 = plt.subplot2grid((4,9), (3,3), colspan=3)
        profile3 = plt.subplot2grid((4,9), (3,6), colspan=3)

        ax1.imshow(input_im.data, cmap='inferno', interpolation='hermite', origin='lower')
        inset1 = ax1.inset_axes([0.6, 0.6, 0.37, 0.37])
        inset1.imshow(input_im.data, cmap='inferno', interpolation='hermite', origin='lower')
        lims = ax1.get_xlim()
        inset1.set_xlim(cropx-cropwidth, cropx+cropwidth)
        inset1.set_ylim(cropy-cropwidth, cropy+cropwidth)
        ax_off(ax1)
        ax_off(inset1)
        scalebar1 = ScaleBar(imscale, imscale_units, location='lower right', color='white', box_alpha=0)
        ax1.add_artist(scalebar1)
        scalebar1_ins = ScaleBar(imscale, imscale_units, location='lower right', color='white', box_alpha=0)
        inset1.add_artist(scalebar1_ins)
        ax1.indicate_inset_zoom(inset1, edgecolor='black')
        ax1.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        inset1.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_title(' (a) original', loc='left')

        profile1.plot(profile_X, origprofile.data, color='blue')
        profile1.set_xlabel('length (nm)')
        profile1.set_ylabel('Intensity (arb. units)')
        profile1.set_yticks([])
        profile1.set_xlim(profile_X[0], profile_X[-1])

        ax2.imshow(match_histograms(svd, input_im.data), cmap='inferno', interpolation='hermite', origin='lower')
        inset2 = ax2.inset_axes([0.6, 0.6, 0.37, 0.37])
        inset2.imshow(match_histograms(svd, input_im.data), cmap='inferno', interpolation='hermite', origin='lower')
        inset2.set_xlim(cropx-cropwidth, cropx+cropwidth)
        inset2.set_ylim(cropy-cropwidth, cropy+cropwidth)
        ax2.set_xticks([])
        ax2.set_yticks([])
        inset2.set_xticks([])
        inset2.set_yticks([])
        scalebar2 = ScaleBar(input_im.axes_manager[0].scale, input_im.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        ax2.add_artist(scalebar2)
        scalebar2_ins = ScaleBar(input_im.axes_manager[0].scale, input_im.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        inset2.add_artist(scalebar2_ins)
        ax2.indicate_inset_zoom(inset2, edgecolor='black')
        ax2.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        inset2.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_title(' (b) filtered', loc='left')

        profile2.plot(profile_X, svdprofile.data, color='blue')
        profile2.set_xlabel('length (nm)')
        profile2.set_yticks([])
        profile2.set_xlim(profile_X[0], profile_X[-1])

        ax3.imshow(match_histograms(svd_diff, input_im.data), cmap='inferno', origin='lower', interpolation='None')
        inset3 = ax3.inset_axes([0.6, 0.6, 0.37, 0.37])
        inset3.imshow(match_histograms(svd_diff, input_im.data), cmap='inferno', origin='lower', interpolation='None')
        inset3.set_xlim(cropx-cropwidth, cropx+cropwidth)
        inset3.set_ylim(cropy-cropwidth, cropy+cropwidth)
        ax3.set_xticks([])
        ax3.set_yticks([])
        inset3.set_xticks([])
        inset3.set_yticks([])
        scalebar3 = ScaleBar(orig.axes_manager[0].scale, orig.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        ax3.add_artist(scalebar3)
        scalebar3_ins = ScaleBar(orig.axes_manager[0].scale, orig.axes_manager[0].units, location='lower right', color='white', box_alpha=0)
        inset3.add_artist(scalebar3_ins)
        ax3.indicate_inset_zoom(inset3, edgecolor='black')
        ax3.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        inset3.plot([roi_x_1, roi_x_2],[roi_y_1, roi_y_2], color='blue')
        ax3.set_xlim(lims)
        ax3.set_ylim(lims)
        ax3.set_title(' (c) difference', loc='left')

        profile3.plot(profile_X, svdprofile_diff, color='blue')
        profile3.set_xlabel('length (nm)')
        profile3.set_yticks([])
        profile3.set_xlim(profile_X[0], profile_X[-1])

        fig.subplots_adjust(top=0.98,
                            bottom=0.07,
                            left=0.02,
                            right=0.995,
                            hspace=0.0,
                            wspace=0.13)

        fig.savefig(savefolder + 'svd_filter_illustrative.png', dpi=400)
        fig.savefig(savefolder + 'svd_filter_illustrative.svg', dpi=400)




        wienerfilt = wiener(input_im.data, settings.wfilt.values[0])
        gaussfilt = gaussian_filter(input_im.data, settings.gfilt.values[0])
        mefilt = medfilt(input_im.data, settings.mfilt.values[0])


        cutsize=16
        oldim = template_rebuilt_bss_lo_2.isig[cutsize: - cutsize, cutsize: - cutsize].deepcopy()
        lp = lopass(oldim, settings.lprad.values[0])
        highpass = oldim - lp
        hpim = np.pad(highpass.data, cutsize, mode='mean')







        centre = (350, 1300)
        halfsize = 200



        xlim = (centre[0] - halfsize, centre[0] + halfsize)
        ylim = (centre[1] + halfsize, centre[1] - halfsize)


        fig, ax = plt.subplots(2,3, figsize=(16,12))

        ax_off(ax[0][0])
        ax_off(ax[0][1])
        ax_off(ax[0][2])
        ax_off(ax[1][0])
        ax_off(ax[1][1])
        ax_off(ax[1][2])

        ax00 = ax[0][0].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax01 = ax[0][1].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax02 = ax[0][2].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax10 = ax[1][0].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax11 = ax[1][1].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax12 = ax[1][2].inset_axes([0.5, 0.5, 0.47, 0.47])

        ax_off(ax00)
        ax_off(ax01)
        ax_off(ax02)
        ax_off(ax10)
        ax_off(ax11)
        ax_off(ax12)

        ax[0][0].imshow(input_im.data, interpolation='Hermite', cmap='inferno')
        ax[0][0].set_title('(a) Original Image')
        ax[0][1].imshow(match_histograms(template_rebuilt_bss_lo_2.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[0][1].set_title('(b) PCA Filtered')
        ax[0][2].imshow(match_histograms(hpim, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[0][2].set_title('(c) PCA Filtered + High Pass')


        ax[1][0].imshow(match_histograms(gaussfilt, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[1][0].set_title('(d) Gaussian Filtered')
        ax[1][1].imshow(match_histograms(mefilt, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[1][1].set_title('(e) Median Filtered')
        ax[1][2].imshow(match_histograms(wienerfilt, input_im.data), interpolation='Hermite', cmap='inferno')
        ax[1][2].set_title('(f) Wiener Filtered')

        ax00.imshow(input_im.data, interpolation='Hermite', cmap='inferno')
        ax01.imshow(match_histograms(template_rebuilt_bss_lo_2.data, input_im.data), interpolation='Hermite', cmap='inferno')
        ax02.imshow(match_histograms(hpim, input_im.data), interpolation='Hermite', cmap='inferno')

        ax10.imshow(match_histograms(gaussfilt, input_im.data), interpolation='Hermite', cmap='inferno')
        ax11.imshow(match_histograms(mefilt, input_im.data), interpolation='Hermite', cmap='inferno')
        ax12.imshow(match_histograms(wienerfilt, input_im.data), interpolation='Hermite', cmap='inferno')

        for axis in [ax00, ax01, ax02, ax10, ax11, ax12]:
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)

        ax[0][0].indicate_inset_zoom(ax00)
        ax[0][1].indicate_inset_zoom(ax01)
        ax[0][2].indicate_inset_zoom(ax02)

        ax[1][0].indicate_inset_zoom(ax10)
        ax[1][1].indicate_inset_zoom(ax11)
        ax[1][2].indicate_inset_zoom(ax12)

        ax[0][0].indicate_inset_zoom(ax00)
        ax[0][1].indicate_inset_zoom(ax01)
        ax[0][2].indicate_inset_zoom(ax02)

        ax[1][0].indicate_inset_zoom(ax10)
        ax[1][1].indicate_inset_zoom(ax11)
        ax[1][2].indicate_inset_zoom(ax12)

        fig.tight_layout()

        fig.savefig(savefolder + 'filtercompare.png')
        fig.savefig(savefolder + 'filtercompare.svg')






        centre = (350, 1300)
        halfsize = 200

        xlim = (centre[0] - halfsize, centre[0] + halfsize)
        ylim = (centre[1] + halfsize, centre[1] - halfsize)

        
        pxscale = input_im.axes_manager[0].scale
        pxunits = input_im.axes_manager[1].scale
        
        fig, ax = plt.subplots(2,3, figsize=(14,10))

        ax_off(ax[0][0])
        ax_off(ax[0][1])
        ax_off(ax[0][2])
        ax_off(ax[1][0])
        ax_off(ax[1][1])
        ax_off(ax[1][2])

        ax00 = ax[0][0].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax01 = ax[0][1].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax02 = ax[0][2].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax10 = ax[1][0].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax11 = ax[1][1].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax12 = ax[1][2].inset_axes([0.5, 0.5, 0.47, 0.47])

        ax_off(ax00)
        ax_off(ax01)
        ax_off(ax02)
        ax_off(ax10)
        ax_off(ax11)
        ax_off(ax12)
        
        framecut = 30
        imgcrop = input_im.data[framecut:-framecut, framecut:-framecut]
        pcacrop = template_rebuilt_bss_lo_2.data[framecut:-framecut, framecut:-framecut]
        hpimcrop = hpim[framecut:-framecut, framecut:-framecut]
        gausscrop = gaussfilt[framecut:-framecut, framecut:-framecut]
        mecrop = mefilt[framecut:-framecut, framecut:-framecut]
        wienercrop = wienerfilt[framecut:-framecut, framecut:-framecut]
        

        ax[0][0].imshow(imgcrop, interpolation='Hermite', cmap='inferno')
        ax[0][0].set_title('(a) Original Image')
        ax[0][1].imshow(pcacrop, interpolation='Hermite', cmap='inferno')
        ax[0][1].set_title('(b) PCA Filtered')
        ax[0][2].imshow(hpimcrop, interpolation='Hermite', cmap='inferno')
        ax[0][2].set_title('(c) PCA Filtered + High Pass')


        ax[1][0].imshow(gausscrop, interpolation='Hermite', cmap='inferno')
        ax[1][0].set_title('(d) Gaussian Filtered')
        ax[1][1].imshow(mecrop, interpolation='Hermite', cmap='inferno')
        ax[1][1].set_title('(e) Median Filtered')
        ax[1][2].imshow(wienercrop, interpolation='Hermite', cmap='inferno')
        ax[1][2].set_title('(f) Wiener Filtered')

        ax00.imshow(imgcrop, interpolation='Hermite', cmap='inferno')
        ax01.imshow(pcacrop, interpolation='Hermite', cmap='inferno')
        ax02.imshow(hpimcrop, interpolation='Hermite', cmap='inferno')

        ax10.imshow(gausscrop, interpolation='Hermite', cmap='inferno')
        ax11.imshow(mecrop, interpolation='Hermite', cmap='inferno')
        ax12.imshow(wienercrop, interpolation='Hermite', cmap='inferno')

        for axis in [ax00, ax01, ax02, ax10, ax11, ax12]:
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)

        ax[0][0].indicate_inset_zoom(ax00)
        ax[0][1].indicate_inset_zoom(ax01)
        ax[0][2].indicate_inset_zoom(ax02)

        ax[1][0].indicate_inset_zoom(ax10)
        ax[1][1].indicate_inset_zoom(ax11)
        ax[1][2].indicate_inset_zoom(ax12)

        ax[0][0].indicate_inset_zoom(ax00)
        ax[0][1].indicate_inset_zoom(ax01)
        ax[0][2].indicate_inset_zoom(ax02)

        ax[1][0].indicate_inset_zoom(ax10)
        ax[1][1].indicate_inset_zoom(ax11)
        ax[1][2].indicate_inset_zoom(ax12)

        fig.tight_layout()

        fig.savefig(savefolder + 'filtercompare_nomatch.png')
        fig.savefig(savefolder + 'filtercompare_nomatch.svg')








        centre = (350, 1300)
        halfsize = 200

        xlim = (centre[0] - halfsize, centre[0] + halfsize)
        ylim = (centre[1] + halfsize, centre[1] - halfsize)

        
        pxscale = input_im.axes_manager[0].scale
        pxunits = input_im.axes_manager[1].units
        
        fig, ax = plt.subplots(1,3, figsize=(14,6))

        ax_off(ax[0])
        ax_off(ax[1])
        ax_off(ax[2])


        ax0 = ax[0].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax1 = ax[1].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax2 = ax[2].inset_axes([0.5, 0.5, 0.47, 0.47])


        ax_off(ax0)
        ax_off(ax1)
        ax_off(ax2)

        
        framecut = 30
        imgcrop = input_im.data[framecut:-framecut, framecut:-framecut]
        pcacrop = template_rebuilt_bss_lo_2.data[framecut:-framecut, framecut:-framecut]
        hpimcrop = hpim[framecut:-framecut, framecut:-framecut]
        

        ax[0].imshow(imgcrop, interpolation='Hermite', cmap='inferno')
        ax[0].set_title('(a) Original Image')
        ax[1].imshow(pcacrop, interpolation='Hermite', cmap='inferno')
        ax[1].set_title('(b) PCA Filtered')
        ax[2].imshow(hpimcrop, interpolation='Hermite', cmap='inferno')
        ax[2].set_title('(c) PCA Filtered + High Pass')


        ax0.imshow(imgcrop, interpolation='Hermite', cmap='inferno')
        ax1.imshow(pcacrop, interpolation='Hermite', cmap='inferno')
        ax2.imshow(hpimcrop, interpolation='Hermite', cmap='inferno')


        for axis in [ax0, ax1, ax2]:
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)

        ax[0].indicate_inset_zoom(ax0)
        ax[1].indicate_inset_zoom(ax1)
        ax[2].indicate_inset_zoom(ax2)

        fig.tight_layout()
        


        fig.savefig(savefolder + 'filtercompare_3.png', dpi=300)
        fig.savefig(savefolder + 'filtercompare_3.svg', dpi=300)
        
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax[0].add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax[1].add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax[2].add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax0.add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax1.add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax2.add_artist(sb)
        
        fig.savefig(savefolder + 'filtercompare_3_sb.png', dpi=300)
        fig.savefig(savefolder + 'filtercompare_3_sb.svg', dpi=300)
        
        
        
        
        
        
        
        
        
        
        
        
        centre = (350, 1300)
        halfsize = 120

        xlim = (centre[0] - halfsize, centre[0] + halfsize)
        ylim = (centre[1] + halfsize, centre[1] - halfsize)

        
        pxscale = input_im.axes_manager[0].scale
        pxunits = input_im.axes_manager[1].units
        
        fig, ax = plt.subplots(1,2, figsize=(11,6))

        ax_off(ax[0])
        ax_off(ax[1])



        ax0 = ax[0].inset_axes([0.5, 0.5, 0.47, 0.47])
        ax1 = ax[1].inset_axes([0.5, 0.5, 0.47, 0.47])



        ax_off(ax0)
        ax_off(ax1)
        ax_off(ax2)

        
        framecut = 30
        imgcrop = input_im.data[framecut:-framecut, framecut:-framecut]
        pcacrop = template_rebuilt_bss_lo_2.data[framecut:-framecut, framecut:-framecut]
        hpimcrop = hpim[framecut:-framecut, framecut:-framecut]
        

        ax[0].imshow(imgcrop, interpolation='Hermite', cmap='inferno')
        ax[0].set_title('(a) Original Image')

        ax[1].imshow(hpimcrop, interpolation='Hermite', cmap='inferno')
        ax[1].set_title('(b) PCA Filtered + High Pass')


        ax0.imshow(imgcrop, interpolation='Hermite', cmap='inferno')

        ax1.imshow(hpimcrop, interpolation='Hermite', cmap='inferno')


        for axis in [ax0, ax1]:
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)

        ax[0].indicate_inset_zoom(ax0)
        ax[1].indicate_inset_zoom(ax1)


        fig.tight_layout()
        


        fig.savefig(savefolder + 'filtercompare_2.png', dpi=300)
        fig.savefig(savefolder + 'filtercompare_2.svg', dpi=300)
        
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax[0].add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax[1].add_artist(sb)

        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax0.add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax1.add_artist(sb)

        
        fig.savefig(savefolder + 'filtercompare_2_sb.png', dpi=300)
        fig.savefig(savefolder + 'filtercompare_2_sb.svg', dpi=300)
        
        
        
        
        
        
        
        
        
        
        centre = (350, 1300)
        halfsize = 120

        xlim = (centre[0] - halfsize, centre[0] + halfsize)
        ylim = (centre[1] + halfsize, centre[1] - halfsize)

        
        pxscale = input_im.axes_manager[0].scale
        pxunits = input_im.axes_manager[1].units
        
        fig, ax = plt.subplots(1,2, figsize=(11,6))

        ax_off(ax[0])
        ax_off(ax[1])


        ax_off(ax0)
        ax_off(ax1)
        ax_off(ax2)

        
        framecut = 30
        imgcrop = input_im.data[framecut:-framecut, framecut:-framecut]
        pcacrop = template_rebuilt_bss_lo_2.data[framecut:-framecut, framecut:-framecut]
        hpimcrop = hpim[framecut:-framecut, framecut:-framecut]
        

        ax[0].imshow(imgcrop, interpolation='Hermite', cmap='inferno')
        ax[0].set_title('(a) Original Image')

        ax[1].imshow(hpimcrop, interpolation='Hermite', cmap='inferno')
        ax[1].set_title('(b) PCA Filtered + High Pass')



        fig.tight_layout()
    

        fig.savefig(savefolder + 'filtercompare_2_nozoom.png', dpi=600)
        fig.savefig(savefolder + 'filtercompare_2_nozoom.svg', dpi=600)
        
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax[0].add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax[1].add_artist(sb)

        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax0.add_artist(sb)
        sb = ScaleBar(pxscale, pxunits, box_alpha=0, color='white', location='lower right')
        ax1.add_artist(sb)

        
        fig.savefig(savefolder + 'filtercompare_2_sb_nozoom.png', dpi=600)
        fig.savefig(savefolder + 'filtercompare_2_sb_nozoom.svg', dpi=600)
        
        
        
        plt.imsave(savefolder + 'original_raw.png', imgcrop, cmap='inferno')
        plt.imsave(savefolder + 'pca_raw.png', pcacrop, cmap='inferno')
        plt.imsave(savefolder + 'pca_lopass.png', hpimcrop, cmap='inferno')
        
        plt.imsave(savefolder + 'original_raw_grey.png', imgcrop)
        plt.imsave(savefolder + 'pca_raw_grey.png', pcacrop)
        plt.imsave(savefolder + 'pca_lopass_grey.png', hpimcrop)
        
        
        






        template_im_2.decomposition()
        template_dmodel_svd_lo_2 = template_im_2.get_decomposition_model(settings.svd2_lo.values[0])
        template_im_2.blind_source_separation(settings.bss2_lo.values[0])         
        template_dmodel_bss_lo_2 = template_im_2.get_bss_model(settings.bss2_lo.values[0])   
        









        nreps = 5

        index_sets = []
        for i in range(nreps):
            indexes = [random.randint(0, template_im_2.data.shape[0]) for i in range(10)]
            index_sets.append(indexes)

        for rep in range(nreps):

            indexes = index_sets[rep]

            template_images = []
            template_positions = []
            svd_template_images = []
            bss_template_images = []

            for i in range(0,10):
                index = indexes[i]

                image = template_im_2.inav[index].data
                template_images.append(image)

                pos = pos_int_pass_2[index]
                template_positions.append(pos)

                svd_image = template_dmodel_svd_lo_2.inav[index].data
                svd_template_images.append(svd_image)

                bss_image = template_dmodel_bss_lo_2.inav[index].data
                bss_template_images.append(bss_image)


            figsize = (16,9)

            gridsize = [8, 14]

            zoomcentre = [951, 826]
            zoomsize = 200

            fig = plt.figure(figsize=figsize)

            ax_input_out = plt.subplot2grid(gridsize, (0,0), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_input_out)
            ax_input_in = plt.subplot2grid(gridsize, (int(gridsize[0]/2),0), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_input_in)

            ax_input_out.imshow(input_im.data, cmap='inferno', interpolation='None')
            ax_input_in.imshow(input_im.data, cmap='inferno', interpolation='None')

            ax_input_in.set_xlim(zoomcentre[0] - zoomsize, zoomcentre[0] + zoomsize)
            ax_input_in.set_ylim(zoomcentre[1] + zoomsize, zoomcentre[1] - zoomsize)

            ax_input_out.indicate_inset_zoom(ax_input_in)



            ax_output_out = plt.subplot2grid(gridsize, (0,10), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_output_out)
            ax_output_in = plt.subplot2grid(gridsize, (int(gridsize[0]/2),10), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_output_in)

            outimage = match_histograms(svd2_low.data, input_im.data)

            ax_output_out.imshow(outimage, cmap='inferno', interpolation='Hermite')
            ax_output_in.imshow(outimage, cmap='inferno', interpolation='Hermite')

            ax_output_in.set_xlim(zoomcentre[0] - zoomsize, zoomcentre[0] + zoomsize)
            ax_output_in.set_ylim(zoomcentre[1] + zoomsize, zoomcentre[1] - zoomsize)

            ax_output_out.indicate_inset_zoom(ax_output_in)

            template_in_axes = []
            for i in range(0,gridsize[0]):
                index = indexes[i]
                ax_template_in = plt.subplot2grid(gridsize, (i, int(gridsize[0]/2)), rowspan=1, colspan=1, aspect='equal')
                ax_off(ax_template_in)
                ax_template_in.imshow(template_images[i], cmap='inferno', interpolation='None')
                sb = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                ax_template_in.add_artist(sb)
                template_in_axes.append(ax_template_in)

                ax_template_in.annotate(str(i+1), (-0.2, 0.7), xycoords='axes fraction', size=16, color='blue')
                pos = template_positions[i]
                bl = [pos[0] - tempwidth2, pos[1] - tempwidth2]
                rect = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                rect2 = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                ax_input_in.add_artist(rect)
                ax_input_out.add_artist(rect2)
                ax_input_out.annotate(str(i+1), (pos[0]-(4*tempwidth2), pos[1] + tempwidth2), xycoords='data', size=16, color='blue', annotation_clip=False)


            eigenvector_axes = []
            for i in range(0, 4):
                index1 = 2*i
                index2 = index1 + 1
                ax_l = plt.subplot2grid(gridsize, (2*i, int(gridsize[0]/2) + 1), rowspan=2, colspan=2, aspect='equal')
                ax_r = plt.subplot2grid(gridsize, (2*i, int(gridsize[0]/2) + 3), rowspan=2, colspan=2, aspect='equal')
                ax_off(ax_l)
                ax_off(ax_r)
                ax_l.imshow(SVDfactors.inav[index1].data, cmap='inferno', interpolation='Hermite')
                ax_r.imshow(SVDfactors.inav[index2].data, cmap='inferno', interpolation='Hermite')
                sb_l = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                sb_r = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                ax_l.add_artist(sb_l)
                ax_r.add_artist(sb_r)
                eigenvector_axes.append(ax_l)
                eigenvector_axes.append(ax_r)

            template_out_axes = []
            for i in range(0,gridsize[0]):
                index = indexes[i]
                ax_template_out = plt.subplot2grid(gridsize, (i, int(gridsize[0]/2) + 5), rowspan=1, colspan=1, aspect='equal')
                ax_off(ax_template_out)
                ax_template_out.imshow(svd_template_images[i], cmap='inferno', interpolation='Hermite')
                sb = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                ax_template_out.add_artist(sb)
                template_out_axes.append(ax_template_out)


                ax_template_out.annotate(str(i+1), (-0.2, 0.7), xycoords='axes fraction', size=16, color='blue')

                pos = template_positions[i]
                bl = [pos[0] - tempwidth2, pos[1] - tempwidth2]
                rect = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                rect2 = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                ax_output_in.add_artist(rect)
                ax_output_out.add_artist(rect2)
                ax_output_out.annotate(str(i+1), (pos[0]-(4*tempwidth2), pos[1] + tempwidth2), xycoords='data', size=16, color='blue', annotation_clip=False)


            sb1 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)
            sb2 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)
            sb3 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)
            sb4 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)

            ax_input_out.add_artist(sb1)
            ax_input_in.add_artist(sb2)
            ax_output_out.add_artist(sb3)
            ax_output_in.add_artist(sb4)

            fig.subplots_adjust(top=1.0,
                                bottom=0.0,
                                left=0.005,
                                right=0.99,
                                hspace=0.25,
                                wspace=0.25)

            fig.savefig(savefolder + 'SVD_%d.png' % rep)
            fig.savefig(savefolder + 'SVD_%d.svg' % rep)




            fig = plt.figure(figsize=figsize)

            ax_input_out = plt.subplot2grid(gridsize, (0,0), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_input_out)
            ax_input_in = plt.subplot2grid(gridsize, (int(gridsize[0]/2),0), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_input_in)

            ax_input_out.imshow(input_im.data, cmap='inferno', interpolation='None')
            ax_input_in.imshow(input_im.data, cmap='inferno', interpolation='None')

            ax_input_in.set_xlim(zoomcentre[0] - zoomsize, zoomcentre[0] + zoomsize)
            ax_input_in.set_ylim(zoomcentre[1] + zoomsize, zoomcentre[1] - zoomsize)

            ax_input_out.indicate_inset_zoom(ax_input_in)



            ax_output_out = plt.subplot2grid(gridsize, (0,10), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_output_out)
            ax_output_in = plt.subplot2grid(gridsize, (int(gridsize[0]/2),10), rowspan=int(gridsize[0]/2), colspan=int(gridsize[0]/2), fig=fig, aspect='equal')
            ax_off(ax_output_in)

            outimage = match_histograms(bss2_low.data, input_im.data)

            ax_output_out.imshow(outimage, cmap='inferno', interpolation='Hermite')
            ax_output_in.imshow(outimage, cmap='inferno', interpolation='Hermite')

            ax_output_in.set_xlim(zoomcentre[0] - zoomsize, zoomcentre[0] + zoomsize)
            ax_output_in.set_ylim(zoomcentre[1] + zoomsize, zoomcentre[1] - zoomsize)

            ax_output_out.indicate_inset_zoom(ax_output_in)

            template_in_axes = []
            for i in range(0,gridsize[0]):
                index = indexes[i]
                ax_template_in = plt.subplot2grid(gridsize, (i, int(gridsize[0]/2)), rowspan=1, colspan=1, aspect='equal')
                ax_off(ax_template_in)
                ax_template_in.imshow(template_images[i], cmap='inferno', interpolation='None')
                sb = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                ax_template_in.add_artist(sb)
                template_in_axes.append(ax_template_in)

                ax_template_in.annotate(str(i+1), (-0.2, 0.7), xycoords='axes fraction', size=16, color='blue')
                pos = template_positions[i]
                bl = [pos[0] - tempwidth2, pos[1] - tempwidth2]
                rect = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                rect2 = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                ax_input_in.add_artist(rect)
                ax_input_out.add_artist(rect2)
                ax_input_out.annotate(str(i+1), (pos[0]-(4*tempwidth2), pos[1] + tempwidth2), xycoords='data', size=16, color='blue', annotation_clip=False)


            eigenvector_axes = []
            for i in range(0, 4):
                index1 = 2*i
                index2 = index1 + 1
                ax_l = plt.subplot2grid(gridsize, (2*i, int(gridsize[0]/2) + 1), rowspan=2, colspan=2, aspect='equal')
                ax_r = plt.subplot2grid(gridsize, (2*i, int(gridsize[0]/2) + 3), rowspan=2, colspan=2, aspect='equal')
                ax_off(ax_l)
                ax_off(ax_r)
                ax_l.imshow(BSSfactors.inav[index1].data, cmap='inferno', interpolation='Hermite')
                ax_r.imshow(BSSfactors.inav[index2].data, cmap='inferno', interpolation='Hermite')
                sb_l = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                sb_r = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                ax_l.add_artist(sb_l)
                ax_r.add_artist(sb_r)
                eigenvector_axes.append(ax_l)
                eigenvector_axes.append(ax_r)

            template_out_axes = []
            for i in range(0,gridsize[0]):
                index = indexes[i]
                ax_template_out = plt.subplot2grid(gridsize, (i, int(gridsize[0]/2) + 5), rowspan=1, colspan=1, aspect='equal')
                ax_off(ax_template_out)
                ax_template_out.imshow(bss_template_images[i], cmap='inferno', interpolation='Hermite')
                sb = ScaleBar(imscale, imscale_units, color='white', box_alpha=0, label_formatter=nullfunc)
                ax_template_out.add_artist(sb)
                template_out_axes.append(ax_template_out)


                ax_template_out.annotate(str(i+1), (-0.2, 0.7), xycoords='axes fraction', size=16, color='blue')

                pos = template_positions[i]
                bl = [pos[0] - tempwidth2, pos[1] - tempwidth2]
                rect = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                rect2 = Rectangle(bl, (2*tempwidth2)+1, (2*tempwidth2)+1, fill=False, color='blue')
                ax_output_in.add_artist(rect)
                ax_output_out.add_artist(rect2)
                ax_output_out.annotate(str(i+1), (pos[0]-(4*tempwidth2), pos[1] + tempwidth2), xycoords='data', size=16, color='blue', annotation_clip=False)


            sb1 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)
            sb2 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)
            sb3 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)
            sb4 = ScaleBar(imscale, imscale_units, color='white', box_alpha=0)

            ax_input_out.add_artist(sb1)
            ax_input_in.add_artist(sb2)
            ax_output_out.add_artist(sb3)
            ax_output_in.add_artist(sb4)

            fig.subplots_adjust(top=1.0,
                                bottom=0.0,
                                left=0.005,
                                right=0.99,
                                hspace=0.25,
                                wspace=0.25)

            fig.savefig(savefolder + 'BSS_%d.png' % rep)
            fig.savefig(savefolder + 'BSS_%d.svg' % rep)

            plt.close('all')
            gc.collect()

