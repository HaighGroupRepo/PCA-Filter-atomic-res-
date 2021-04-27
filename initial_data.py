# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:29:15 2020

@author: Nick
"""
import os
import pandas as pd
import hyperspy.api as hs
from glob import glob

basefolder = r"C:\Users\Nick\Dropbox (The University of Manchester)\samples\astrid\TwistTMD Data\Nick Analysis\images to fit\EMC_PCA_highpass" + os.sep

filenamelist = glob(basefolder + '*.dm3')

fnamelist = []
pxscalelist = []
sizelist = []
numpxlist = []
for filename in filenamelist:
    if filename[-3:] == 'dm3':
        fname = filename.split(os.sep)[-1]
        fnamelist.append(fname)
        im = hs.load(filename)
        im_ax = im.axes_manager[0]
        pxscale = im_ax.scale
        pxscalelist.append(pxscale)
        num_px = im_ax.size
        numpxlist.append(num_px)
        size = num_px * pxscale
        sizestr = ('%.2f ' % size + im_ax.units)
        sizelist.append(sizestr)
        
fnum = len(fnamelist)
zeroarr = numpy.zeros(fnum)

settings_df = pd.DataFrame()

settings_df['filename'] = fnamelist
settings_df['pxscale'] = pxscalelist
settings_df['imsize'] = sizelist
settings_df['pxnum'] = numpxlist

settings_df.to_csv(basefolder + 'file_list.csv')

settings_df['smoothrad'] = zeroarr
settings_df['sep1_px'] = zeroarr
settings_df['tempwidth1'] = zeroarr
settings_df['svd1_hi'] = zeroarr
settings_df['svd1_lo'] = zeroarr
settings_df['bss1_lo'] = zeroarr
settings_df['bss1_hi'] = zeroarr
settings_df['sep2_px'] = zeroarr
settings_df['tempwidth2'] = zeroarr
settings_df['svd2_hi'] = zeroarr
settings_df['svd2_med'] = zeroarr
settings_df['svd2_lo'] = zeroarr
settings_df['svd2_hi'] = zeroarr
settings_df['svd2_med'] = zeroarr
settings_df['svd2_lo'] = zeroarr
settings_df['bss2_hi'] = zeroarr
settings_df['bss2_med'] = zeroarr
settings_df['bss2_lo'] = zeroarr
settings_df['gfilt'] = zeroarr
settings_df['wfilt'] = zeroarr
settings_df['mfilt'] = zeroarr
settings_df['mrad'] = zeroarr

settings_df.to_csv(basefolder + 'settings_base.csv')