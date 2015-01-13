Source
------
Data from [LAMBDA WMAP archive](http://lambda.gsfc.nasa.gov/product/map/dr5/m_products.cfm).

`wmap_ilc_9yr_v5.fits` - Final ILC foreground-subtracted map. 1 deg resolution,
    nside = 512
`wmap_ilc_rgn_defn_9yr_v5.fits` - ILC map region definitions. Refer to header.
`wmap_band_smth_imap_r9_9yr_<BAND>_v5.fits` - The smoothed intensity map for the
    specified band (<BAND> in [K, Ka, Q, V, W]). 1 deg resolution. nside = 512.
    These 5 maps are the input to the WMAP ILC routine. 



Importing data into a Python environment
----------------------------------------

Use `pyfits`, available in `astropy.io.fits`, e.g.

    from astropy.io import fits
    
    ilcdata = fits.open('data/wmap_ilc_9yr_v5.fits')
    ilc = ilcdata[1]
    # Header: ilc.header
    ilc_nside = ilc.header['NSIDE']
    ilcTs = ilc.data['TEMPERATURE']  # in default ordering (nested for WMAP)
    ilcTs = hp.pixelfunc.reorder(ilcTs, n2r=True)  # Nested -> Ring
    
    
Plotting data
-------------
    
A plotting example:

    import healpy as hp
    
    # See above section for definition of `ilcTs`.
    hp.mollview(ilcTs, nest=True)
    
    
Smoothing and downsampling data
-------------------------------

The `healpy` smoothing routine only works on ring-ordered data (hence the 
transformation above). Then smooth as so:

    import numpy as np
    
    sigma = 1.*np.pi/180  # Smooth with a 1 degree std dev Gaussian
    ilcTs_smooth = hp.sphtfunc.smoothing(ilcTs, sigma=sigma)
    
and downsample:

    ilcTs_smooth_128 = hp.pixelfunc.ud_grade(ilcTs_smooth, 
                                             nside_out=128)  # 512 -> 128
    
