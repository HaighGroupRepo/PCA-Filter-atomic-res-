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

from tqdm import tqdm
from glob import glob
from scipy.signal import wiener, medfilt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.transform import resize


import warnings
warnings.filterwarnings("ignore")

# %matplotlib

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
    ax.imshow(hsimage.data[framecut:-framecut, framecut:-framecut], interpolation='hermite', cmap='afmhot')
    scalebar = ScaleBar(hsimage.axes_manager[0].scale, hsimage.axes_manager[0].units, color='white', box_alpha=0)
    ax.add_artist(scalebar)
    fig.savefig(folder + os.sep + hsimage.metadata.General.title + '.png', overwrite=True, dpi=hsimage.axes_manager[0].size/10)
    fig.savefig(folder + os.sep + hsimage.metadata.General.title + '.svg', overwrite=True, dpi=hsimage.axes_manager[0].size/10)
    plt.close(fig=fig)
    
# save numpy image with scalebar
def imsave(folder, title ,npimage, scale_dx, scale_units):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(npimage, interpolation='hermite', cmap='afmhot')
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
    ax.imshow(mag_spec, cmap='afmhot')
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
        plt.imsave(folder + '%s.png' % str(i).zfill(4), resize(image, (scfactor*image.shape[0], scfactor*image.shape[1]), order=0), cmap='afmhot')

def export_bss_loading_ims(image, comp_ids, folder, scfactor=10):
    if not os.path.exists(folder):
        os.mkdir(folder)
    factors = image.get_bss_factors().inav[:comp_ids]
    facdata = factors.data
    for i in range(factors.data.shape[0]):
        image = facdata[i]
        plt.imsave(folder + '%s.png' % str(i).zfill(4), resize(image, (scfactor*image.shape[0], scfactor*image.shape[1]), order=0), cmap='afmhot')

# load data and create savefolder

basefolder = r"C:\Users\Nick\Dropbox (The University of Manchester)\samples\astrid\TwistTMD Data\Nick Analysis\images to fit\EMC_PCA_highpass" + os.sep

filenamelist = glob(basefolder + '*.dm3')

settingslist = pd.read_csv(basefolder + 'settings_emc.csv', converters={'filename': lambda x: str(x)})

for filename in tqdm(filenamelist[:]):

    filepath = filename.split(os.sep)
    name = filepath[-1][:-4]
    filepath[0] += '\\'
    savefolder = os.path.join(*filepath[:-1], 'outputs_v1', name + '_output') + os.sep
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    settings = settingslist[settingslist.filename == name + '.dm3']

    input_im = hs.load(filename)
    input_im.plot(cmap='afmhot')
    input_im.change_dtype('float64')

    plt.imsave(savefolder + 'orig.png', input_im.data, cmap='afmhot')

    # pixel size
    imscale = input_im.axes_manager[0].scale
    imscale_units = input_im.axes_manager[0].units

    segment = input_im.isig[:,:].deepcopy()

    #clear up

    plt.close('all')
    gc.collect()

    if os.path.exists(savefolder + 'BSS_1_HI.hspy'):
        template_rebuilt_bss_hi = hs.load(savefolder + 'BSS_1_HI.hspy')
    else:

        # smoothing for peakfinding

        segment_smooth = segment.deepcopy()
        segment_smooth.data = gaussian_filter(segment.data, settings.smoothrad.values[0])
        plt.imsave(savefolder + 'smoothed_image_peakfinding_1.png', segment_smooth.data, cmap='afmhot')

        if os.path.exists(savefolder + '/atom_lattice_save.hdf5'):
            atom_lattice_load = am.load_atom_lattice_from_hdf5(savefolder + '/atom_lattice_save.hdf5', construct_zone_axes=False)
            sublattice = atom_lattice_load.sublattice_list[0]

        # find and refine atom positions

        else:

            atom_pos = am.get_atom_positions(segment_smooth, separation=settings.sep1_px.values[0])
            sublattice = am.Sublattice(atom_pos, image=segment.data)
            slplot(sublattice, 'markers1_initial', scale=2)

            plt.close('all')
            gc.collect()

            sublattice.find_nearest_neighbors()
            sublattice.refine_atom_positions_using_center_of_mass()
            slplot(sublattice, 'markers1_refined', scale=2)

            plt.close('all')
            gc.collect()

            # save (and/ or load) sublattice

            atom_lattice = am.Atom_Lattice(image=sublattice.image, sublattice_list=[sublattice])
            atom_lattice.save(savefolder + '/atom_lattice_save.hdf5', overwrite=True)

        # set up template image for decomposition

        pos_int = np.round(sublattice.atom_positions).astype(int)
        tempwidth = settings.tempwidth1.values[0]# template size = 2 x tempwidth + 1
        template_im, pos_int_pass = round_templates(segment, tempwidth, pos_int)

        # SVD decomp and rebuild

        template_im.decomposition()

        plt.close('all')
        gc.collect()

        export_decomposition_loading_ims(template_im, comp_ids=60, folder = savefolder + 'svd_1_eigenvalues' + os.sep)

        template_im.plot_explained_variance_ratio(n=120, log=True)
        plt.gcf().savefig(savefolder + 'svd1_scree_out.png', dpi=300)
        ylim = plt.gca().get_ylim()
        plt.gca().set_ylim(ylim[0], ylim[1]/1e4)
        plt.gcf().savefig(savefolder + 'svd1_scree_in.png', dpi=300)

        template_dmodel_svd_hi = template_im.get_decomposition_model(settings.svd1_hi.values[0])
        template_dmodel_svd_lo = template_im.get_decomposition_model(settings.svd1_lo.values[0])
        template_rebuilt_svd_hi = template_rebuild(segment, pos_int_pass, template_dmodel_svd_hi, tempwidth, 'SVD_HI_1')
        template_rebuilt_svd_lo = template_rebuild(segment, pos_int_pass, template_dmodel_svd_lo, tempwidth, 'SVD_LO_1')
        rebuiltsave(savefolder, template_rebuilt_svd_hi)
        rebuiltsave(savefolder, template_rebuilt_svd_lo)

        del template_dmodel_svd_hi, template_dmodel_svd_lo, template_rebuilt_svd_hi, template_rebuilt_svd_lo

        # BSS decomp and rebuild
                    
        template_im.blind_source_separation(settings.bss1_lo.values[0])
        export_bss_loading_ims(template_im, comp_ids=settings.bss1_lo.values[0], folder = savefolder + 'bss_1_lo_eigenvectors' + os.sep)

        template_dmodel_bss_lo = template_im.get_bss_model(settings.bss1_lo.values[0])        
        template_rebuilt_bss_lo = template_rebuild(segment, pos_int_pass, template_dmodel_bss_lo, tempwidth, 'BSS_1_LO')
        bss_1_lo_diff = template_rebuilt_bss_lo - input_im
        bss_1_lo_diff.metadata.General.title = 'BSS_1_lo residuals'
        rebuiltsave(savefolder, template_rebuilt_bss_lo)
        rebuiltsave(savefolder, bss_1_lo_diff)

        template_im.decomposition()
        template_im.blind_source_separation(settings.bss1_hi.values[0])
        export_bss_loading_ims(template_im, comp_ids=settings.bss1_hi.values[0], folder = savefolder + 'bss_1_hi_eigenvectors' + os.sep)

        template_dmodel_bss_hi = template_im.get_bss_model(settings.bss1_hi.values[0])        
        template_rebuilt_bss_hi = template_rebuild(segment, pos_int_pass, template_dmodel_bss_hi, tempwidth, 'BSS_1_hi')
        bss_1_hi_diff = template_rebuilt_bss_hi - input_im
        bss_1_hi_diff.metadata.General.title = 'BSS_1_hi residuals'
        rebuiltsave(savefolder, template_rebuilt_bss_hi)
        rebuiltsave(savefolder, bss_1_hi_diff)

        plt.close('all')
        gc.collect()

        del template_rebuilt_bss_hi, template_rebuilt_bss_lo, template_dmodel_bss_hi, template_dmodel_bss_lo
        del template_im
        gc.collect()

    if not os.path.exists(savefolder + 'atom_lattice2_save.hdf5'):

        # find and refine atom positions
        template_rebuilt_bss_hi = hs.load(savefolder + 'BSS_1_HI.hspy') 
        atom_pos_2 = am.get_atom_positions(template_rebuilt_bss_hi, separation=settings.sep2_px.values[0])
        sublattice_2 = am.Sublattice(atom_pos_2, image=template_rebuilt_bss_hi.data)
        slplot(sublattice_2, 'markers2_initial', scale=2)

        plt.close('all')
        gc.collect()

        sublattice_2.find_nearest_neighbors()
        sublattice_2.refine_atom_positions_using_center_of_mass()
        slplot(sublattice_2, 'markers2_comrefined', scale=2)

        plt.close('all')
        gc.collect()

        # save (and/ or load) sublattice

        atom_lattice_2 = am.Atom_Lattice(image=sublattice_2.image, sublattice_list=[sublattice_2])
        atom_lattice_2.save(savefolder + 'atom_lattice2_save.hdf5', overwrite=True)

    if not os.path.exists(savefolder + 'BSS_2_HI.hspy'):


        atom_lattice2_load = am.load_atom_lattice_from_hdf5(savefolder + 'atom_lattice2_save.hdf5', construct_zone_axes=False)
        sublattice_2 = atom_lattice2_load.sublattice_list[0]


        # set up template image for decomposition

        pos_int_2 = np.round(sublattice_2.atom_positions).astype(int)
        tempwidth2 = settings.tempwidth2.values[0] # template size = 2 x tempwidth + 1
        template_im_2, pos_int_pass_2 = round_templates(segment, tempwidth2, pos_int_2)

        # # SVD decomp and rebuild

        template_im_2.decomposition()

        plt.close('all')
        gc.collect()

        export_decomposition_loading_ims(template_im_2, comp_ids=60, folder = savefolder + 'svd_2_eigenvalues' + os.sep)

        template_im_2.export_decomposition_results(comp_ids=100, folder=savefolder + 'svd_2_eigenvalues', multiple_files=False)

        template_im_2.plot_explained_variance_ratio(n=120, log=True)
        plt.gcf().savefig(savefolder + 'svd2_scree_out.png', dpi=300)
        ylim = plt.gca().get_ylim()
        plt.gca().set_ylim(ylim[0], ylim[1]/1e4)
        plt.gcf().savefig(savefolder + 'svd2_scree_in.png', dpi=300)

        template_dmodel_svd_hi_2 = template_im_2.get_decomposition_model(settings.svd2_hi.values[0])
        template_dmodel_svd_med_2 = template_im_2.get_decomposition_model(settings.svd2_med.values[0])
        template_dmodel_svd_lo_2 = template_im_2.get_decomposition_model(settings.svd2_lo.values[0])
        template_rebuilt_svd_hi_2 = template_rebuild(segment, pos_int_pass_2, template_dmodel_svd_hi_2, tempwidth2, 'SVD_HI_2')
        template_rebuilt_svd_med_2 = template_rebuild(segment, pos_int_pass_2, template_dmodel_svd_med_2, tempwidth2, 'SVD_MED_2')
        template_rebuilt_svd_lo_2 = template_rebuild(segment, pos_int_pass_2, template_dmodel_svd_lo_2, tempwidth2, 'SVD_LO_2')
        rebuiltsave(savefolder, template_rebuilt_svd_hi_2)
        rebuiltsave(savefolder, template_rebuilt_svd_med_2)
        rebuiltsave(savefolder, template_rebuilt_svd_lo_2)

        del template_dmodel_svd_hi_2, template_dmodel_svd_med_2, template_dmodel_svd_lo_2, template_rebuilt_svd_hi_2, template_rebuilt_svd_med_2, template_rebuilt_svd_lo_2

        plt.close('all')
        gc.collect()

        template_im_2.blind_source_separation(settings.bss2_hi.values[0])
        export_bss_loading_ims(template_im_2, comp_ids=settings.bss2_hi.values[0], folder = savefolder + 'bss_2_hi_eigenvectors' + os.sep)

        plt.close('all')
        gc.collect()
        
        template_dmodel_bss_hi_2 = template_im_2.get_bss_model(settings.bss2_hi.values[0])        
        template_rebuilt_bss_hi_2 = template_rebuild(segment, pos_int_pass_2, template_dmodel_bss_hi_2, tempwidth2, 'BSS_2_hi')
        bss_1_hi_diff_2 = template_rebuilt_bss_hi_2 - input_im
        bss_1_hi_diff_2.metadata.General.title = 'BSS_2_hi residuals'
        rebuiltsave(savefolder, template_rebuilt_bss_hi_2)
        rebuiltsave(savefolder, bss_1_hi_diff_2)

        plt.close('all')
        gc.collect()

        template_im_2.decomposition()
        template_im_2.blind_source_separation(settings.bss2_med.values[0])
        export_bss_loading_ims(template_im_2, comp_ids=settings.bss2_med.values[0], folder = savefolder + 'bss_2_med_eigenvectors' + os.sep)

        plt.close('all')
        gc.collect()

        template_dmodel_bss_med_2 = template_im_2.get_bss_model(settings.bss2_med.values[0])        
        template_rebuilt_bss_med_2 = template_rebuild(segment, pos_int_pass_2, template_dmodel_bss_med_2, tempwidth2, 'BSS_2_med')
        bss_1_med_diff_2 = template_rebuilt_bss_med_2 - input_im
        bss_1_med_diff_2.metadata.General.title = 'BSS_2_med residuals'
        rebuiltsave(savefolder, template_rebuilt_bss_med_2)
        rebuiltsave(savefolder, bss_1_med_diff_2)

        plt.close('all')
        gc.collect()

        template_im_2.decomposition()
        template_im_2.blind_source_separation(settings.bss2_lo.values[0])
        export_bss_loading_ims(template_im_2, comp_ids=settings.bss2_lo.values[0], folder = savefolder + 'bss_2_lo_eigenvectors' + os.sep)
        template_im_2.export_bss_results(folder=savefolder + 'bss_2_lo_eigenvalues', multiple_files=False)
        plt.close('all')
        gc.collect()

        template_dmodel_bss_lo_2 = template_im_2.get_bss_model(settings.bss2_lo.values[0])        
        template_rebuilt_bss_lo_2 = template_rebuild(segment, pos_int_pass_2, template_dmodel_bss_lo_2, tempwidth2, 'BSS_2_lo')
        bss_1_lo_diff_2 = template_rebuilt_bss_lo_2 - input_im
        bss_1_lo_diff_2.metadata.General.title = 'BSS_2_lo residuals'
        rebuiltsave(savefolder, template_rebuilt_bss_lo_2)
        rebuiltsave(savefolder, bss_1_lo_diff_2)


        plt.close('all')
        gc.collect()

        del template_dmodel_bss_hi_2, template_dmodel_bss_med_2, template_dmodel_bss_lo_2, template_rebuilt_bss_hi_2, template_rebuilt_bss_med_2, template_rebuilt_bss_lo_2
        del template_im_2

    if not os.path.exists(savefolder + os.sep + 'lopass_bss2_hi' + os.sep):


        template_rebuilt_bss_hi_2 = hs.load(savefolder + 'BSS_2_HI.hspy')
        template_rebuilt_bss_med_2 = hs.load(savefolder + 'BSS_2_MED.hspy')
        template_rebuilt_bss_lo_2 = hs.load(savefolder + 'BSS_2_LO.hspy')


        cutsize = 16

        lopasshifolder = savefolder + os.sep + 'lopass_bss2_hi' + os.sep
        if not os.path.exists(lopasshifolder):
            os.mkdir(lopasshifolder)

        for i in tqdm(range(0,64,2), desc='Outputting Lopass'):
            oldim = template_rebuilt_bss_hi_2.isig[cutsize: - cutsize, cutsize: - cutsize].deepcopy()
            lp = lopass(oldim, i)
            imsave(lopasshifolder, '%d' % (i), (oldim - lp).data, input_im.axes_manager[0].scale, input_im.axes_manager[0].units)
            plt.close('all')
            gc.collect()

        lopassmedfolder = savefolder + os.sep + 'lopass_bss2_med' + os.sep
        if not os.path.exists(lopassmedfolder):
            os.mkdir(lopassmedfolder)

        for i in tqdm(range(0,64,2), desc='Outputting Lopass'):
            oldim = template_rebuilt_bss_med_2.isig[cutsize: - cutsize, cutsize: - cutsize].deepcopy()
            lp = lopass(oldim, i)
            imsave(lopassmedfolder, '%d' % (i), (oldim - lp).data, input_im.axes_manager[0].scale, input_im.axes_manager[0].units)
            plt.close('all')
            gc.collect()

        lopasslofolder = savefolder + os.sep + 'lopass_bss2_lo' + os.sep
        if not os.path.exists(lopasslofolder):
            os.mkdir(lopasslofolder)

        for i in tqdm(range(0,64,2), desc='Outputting Lopass'):
            oldim = template_rebuilt_bss_lo_2.isig[cutsize: - cutsize, cutsize: - cutsize].deepcopy()
            lp = lopass(oldim, i)
            imsave(lopasslofolder, '%d' % (i), (oldim - lp).data, input_im.axes_manager[0].scale, input_im.axes_manager[0].units)
            plt.close('all')
            gc.collect()

        atom_lattice2_load = am.load_atom_lattice_from_hdf5(savefolder + 'atom_lattice2_save.hdf5', construct_zone_axes=False)
        sublattice_2 = atom_lattice2_load.sublattice_list[0]


        # set up template image for decomposition

        pos_int_2 = np.round(sublattice_2.atom_positions).astype(int)
        tempwidth2 = settings.tempwidth2.values[0] # template size = 2 x tempwidth + 1
        template_im_2, pos_int_pass_2 = round_templates(segment, tempwidth2, pos_int_2)


        templateICA_bss(template_im_2, settings.bss2_lo.values[0], pos_int_pass_2, savefolder, 'bssICA template2_bss_lo_2')
        plt.close('all')
        gc.collect()

        templateICA_nmf(template_im_2, settings.bss2_lo.values[0], pos_int_pass_2, savefolder, 'nmfICA template2_bss_lo_2')
        plt.close('all')
        gc.collect()
            
        image_arr_2 = template_rebuilt_bss_lo_2.deepcopy().data
        image_arr_2 -= (image_arr_2[image_arr_2 != 0.0]).min()
        image_arr_2[image_arr_2 < 0] = 0.0
        image_arr_2 = image_arr_2 / image_arr_2.max()
        #sns.distplot(image_arr_2.flatten())

        unblurringfolder = savefolder + 'wiener_unblur' + os.sep
        if not os.path.exists(unblurringfolder):
            os.mkdir(unblurringfolder)

        for rad in range(1,3):
            for bal in [0.1, 0.5, 1, 2, 5, 10, 100]:
                unblurred_2 = skimage.restoration.wiener(image_arr_2, gauss2Dmask(5, rad), balance=bal)
                unblurred_im_2 = segment.deepcopy()
                unblurred_im_2.data = unblurred_2
                unblurred_im_2.metadata.General.title = 'Wiener Unblurred BSS 2 LO rad%d balance %d' %(rad, bal)
                rebuiltsave(unblurringfolder, unblurred_im_2)
                    
        wienerfolder = savefolder + 'wiener_filter' + os.sep
        if not os.path.exists(wienerfolder):
            os.mkdir(wienerfolder)

        est_sigmas = skimage.restoration.estimate_sigma(input_im.data)

        np.savetxt(wienerfolder + 'est_sigma.txt', [est_sigmas,])

        for filsize in [3,5,7,9,11]:
            filtered = wiener(input_im.data, mysize=filsize)
            imsave(wienerfolder, 'filsize %d' % (filsize), filtered, input_im.axes_manager[0].scale, input_im.axes_manager[0].units)

        gaussianfolder = savefolder + 'gaussian_filter' + os.sep
        if not os.path.exists(gaussianfolder):
            os.mkdir(gaussianfolder)

        for filsize in [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8, 3]:
            filtered = gaussian_filter(input_im.data, filsize)
            imsave(gaussianfolder, 'filsize %f' % (filsize), filtered, input_im.axes_manager[0].scale, input_im.axes_manager[0].units)


