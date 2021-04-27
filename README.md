# PCA-Filter-atomic-res-
Example of patch based PCA filtering of atomic resolution imaging


Contains absolute filelinks so will need editing to refelct your filestructure (or conversion to relative). Example data included - do not share outside group.

PCA patch based filtering (in filterloop_emc) works by:

1. finding estimated atomic locations by finding bright spots
2. Extract regions of image centred on atomic locations into stack
3. Perform PCA on stack of image patches to denoise
4. Tiling denoised image patches into denoised image
5. Finding improved atomic locations from denoised image and repeating

Notes: This is a non local averaging technique, so may (settings dependent) remove defects/ distortions if they arent repeated in your image (can be extended to large series of images)
Works particularly well for images where you have several different local structures distributed across FOV (eg twisted 2DM).

Can also do ICA in same way - ask me for details (nick)
