# -*- coding: utf-8 -*-
"""
@author: Robert A. McLeod
@email: Robert.McLeod@hitachi-hhtc.ca
"""
from typing import List, Sequence, Dict, Tuple
import logging
log = logging.getLogger(__name__)

# MKL FFT does apparently respect the 'MKL_NUM_THREADS' environment variable if 
# we populate it prior to import.
import os
import cpufeature
n_thread = cpufeature.CPUFeature['num_physical_cores']
os.environ['MKL_NUM_THREADS'] = str(n_thread)

import numpy as np
# import numexpr3 as ne3
# ne3.set_nthreads(n_thread)
import matplotlib.pyplot as plt
import scipy.ndimage as ni
from mkl_fft import rfftn_numpy as rfftn
from mkl_fft import irfftn_numpy as irfftn
from mkl_fft import fft2, ifft2

# Required for polar fft:
# import hihi.util as hutil



def show(mage, title='', block=True):
    plt.figure()
    plt.imshow(mage)
    plt.title(title)
    plt.show(block=block)

_valid_dims = np.sort(np.concatenate((2**np.arange(1,14), 3**np.arange(1,10)))).astype(int)
def _find_valid_fft_dim(shape: List[int]) -> List[int]:
    """
    Finds the nearest power-of-2 or power-of-3 shape that an image can be 
    padded to for Fast Fourier Transforms.
    """
    global _valid_dims
    return [_valid_dims[np.argwhere(N <= _valid_dims)[0][0]] for N in shape]

def edge_mask(image: np.ndarray, edge: Sequence[int]=(32, 32, 32, 32)):
    """
    Returns a mask where the edges are masked, up to `[left, top, right, bottom]`.
    """
    if isinstance(edge, int):
        edge = (edge, edge, edge, edge)

    mask = np.ones(image.shape, dtype=np.bool)
    if edge[0] > 0:
        mask[:, :edge[0]]  = False
    if edge[1] > 0:
        mask[:edge[1], :]  = False
    if edge[2] > 0:
        mask[:, -edge[2]:] = False
    if edge[3] > 0:
        mask[-edge[3]:, :] = False
    return mask

# Slow version
def mncc(fixed: np.ndarray, moving: np.ndarray, 
             mask_fixed: np.ndarray=None, mask_moving: np.ndarray=None,
             subpix: int=8, low_pass: float=0.0, overlap_ratio: float=0.2) -> Tuple[List[int], np.ndarray]:
    """
    Computes the masked, normalized cross-correlation (NCC) on real- or complex-valued images. 
    Generally `realmncc` is substantially faster for real (floating-point) 
    images.

    Parameters
    ----------
    fixed
        a fixed (i.e. template) image
    moving
        a moving (i.e. base) image whose translation relative to `fixed` is to 
        be computed.
    masked_fixed
        the mask for the fixed image. If ``None`` a standard `edge_mask` will 
        be constructed.
    masked_moving:
        the mask for the moving image. If ``None`` is assumed to be the same as 
        `masked_fixed`.

    subpix
        The fraction of a pixel precision to estimate the shift to. Disable by 
        setting to `1`.
    low_pass
        The sigma value to Gaussian low-pass the NCC by. Generally low-pass 
        filtering reduces the image of shot noise in NCC, but it can also reduce
        the correlation on high-frequency features (which may also desireable).
        Typical values are in the range `[0.5, 2.0]`. Disable by setting to `0.0`.
    overlap_ratio
        The minimum overlap between `masked_fixed` and `masked_moving` to accept
        as valid, in terms of fraction of the total number of valid pixels in 
        the mask. Generally the user should not have to change this value.

    Returns
    -------
    shift
        The estimated shift between `fixed` and `moving`.
    ncc
        The calculated normalized, masked cross-correlation.

    Citation
    --------
    D. Padfield. Masked object registration in the Fourier domain. 
    IEEE Transactions on Image Processing, 21(5):2706–2718, 2012.
    """
    
    # We also want a faster version that just takes a stack of images and either a single 
    # mask or stack of masks and cross-correlates each image to the first one   
    shapeFixed = np.array(fixed.shape)
    shapeMoving = np.array(moving.shape)
    shapeCombined = shapeFixed + shapeMoving - 1

    eps = np.finfo(np.float64).resolution
    if fixed.dtype == np.float32:
        eps = np.finfo(np.float32).resolution
    
    shapeOptimized = _find_valid_fft_dim(shapeCombined)
    
    # Verify positivity in images
    # minFixed = np.min(fixed)
    # if minFixed < 0.0:
    #     fixed -= minFixed
    # minMoving = np.min(moving)
    # if minMoving < 0.0:
    #     moving -= minFixed
    
    # Verify mask shapes and emptiness
    if mask_fixed is None:
        mask_fixed = edge_mask(fixed)
    if mask_moving is None:
        mask_moving = mask_fixed
    
    # Ensure masks are doubles 
    mask_fixed = np.array(mask_fixed > 0, dtype=fixed.dtype)
    mask_moving = np.array(mask_moving > 0, dtype=fixed.dtype)

    # Apply masks
    fixed = fixed * mask_fixed
    moving = moving * mask_moving
    
    # Pre-flip (RAM: This may screw up your plans to accelerate the code)
    moving = np.rot90(moving, k=2)
    mask_moving = np.rot90(mask_moving, k=2)
    
    # F_s
    # Note that we can speed things up considerably by passing in F_s of the masks and images, if they are re-used.
    F_fixed_sqr = fft2( fixed * fixed, shapeOptimized)
    F_moving_sqr = fft2( moving * moving, shapeOptimized)
    F_fixed = fft2(fixed, shapeOptimized)
    F_moving = fft2(moving, shapeOptimized)
    F_mask_fixed = fft2(mask_fixed, shapeOptimized)
    F_mask_moving = fft2(mask_moving, shapeOptimized)
    
    # Limit the overlap to reasonable values
    overlap_mask = np.real(ifft2(F_mask_fixed * F_mask_moving))
    # overlap_mask = np.maximum(np.round(overlap_mask), eps)
    overlap_mask = np.maximum(overlap_mask, eps)
    
    # Masks correlations
    F_corr_mask_fixed = np.real(ifft2(F_mask_moving * F_fixed))
    F_corr_mask_moving = np.real(ifft2(F_mask_fixed * F_moving))

    numerator = np.real(ifft2(F_moving * F_fixed)) - F_corr_mask_fixed*F_corr_mask_moving / overlap_mask

    denom_fixed = np.real(ifft2(F_mask_moving * F_fixed_sqr)) - F_corr_mask_fixed*F_corr_mask_fixed / overlap_mask
    denom_moving = np.real(ifft2(F_mask_fixed * F_moving_sqr)) - F_corr_mask_moving*F_corr_mask_moving / overlap_mask
    
    denom = np.sqrt(np.maximum(denom_fixed * denom_moving, 0.0))
    # show(denom, 'ncc denom', block=False)
    
    tolerance = 1e3 * np.spacing(np.max(np.abs(denom)))
    # print(f'Tolerance: {tolerance}')

    mask_not_zero = denom > tolerance
    denom = np.maximum(denom, tolerance)  
    C = (numerator / denom) * mask_not_zero
    # show(C, 'C for NCC', block=False)
    # Need to further apply a mask based on overlap_mask and the desired minimum overlap_ratio
    C *= (overlap_mask >= overlap_ratio * np.max(overlap_mask))
    # show(C, 'C2 for NCC', block=False)
    C = np.clip(C, -1, 1)

    # Crop back to combinedSize
    diffShape = shapeCombined - shapeOptimized
    if diffShape.any() > 0:
        C = C[:diffShape[0], :diffShape[1]]

    # Low-pass filter the cross-correlation to suppress noise
    # (also suppresses influence of very high frequency features in specimen)
    if low_pass > 0.0:
        # print(f'Low pass filtering with kernel {low_pass}')
        C = ni.gaussian_filter(C, low_pass)

    # Find maximum in NCC
    max_pos = np.unravel_index(np.argmax(C), shapeCombined)
    translate = max_pos - (shapeCombined-1)/2.0

    # Fourier subpixel interpolation of NCC
    if subpix > 1:
        F__C_sub = fft2(C[max_pos[0]-8: max_pos[0]+8, 
                                  max_pos[1]-8: max_pos[1]+8])
        F__C_pad = np.zeros([8*subpix, 8*subpix], dtype=F__C_sub.dtype)
        # Copy the quadrants into the zero-padded oversampled Fourier transform
        # of the NCC
        F__C_pad[:9, :9] = F__C_sub[:9,:9]
        F__C_pad[-8:, :9] = F__C_sub[-8:,:9]
        F__C_pad[:9, -8:] = F__C_sub[:9,-8:]
        F__C_pad[-8:, -8:] = F__C_sub[-8:,-8:]
        # Calculate the oversampled NCC
        C_pad = ifft2(F__C_pad).real
        # show(C_pad)
        max_pos_pad = np.array(np.unravel_index(np.argmax(C_pad), C_pad.shape))

        fractional = (max_pos_pad) - np.array(C_pad.shape)/2.0
        fractional = (np.array(fractional) - 1) / subpix
        # print("Fractional position: ", fractional)
        translate += fractional

    return -translate, C

# Optimized fast version
def realmncc(fixed: np.ndarray, moving: np.ndarray, 
             mask_fixed: np.ndarray=None, mask_moving: np.ndarray=None,
             subpix: int=8, low_pass: float=0.0, overlap_ratio: float=0.2) -> Tuple[List[int], np.ndarray]:
    """
    Computes the masked, normalized cross-correlation (NCC) on real-valued 
    images only.

    Parameters
    ----------
    fixed
        a fixed (i.e. template) image
    moving
        a moving (i.e. base) image whose translation relative to `fixed` is to 
        be computed.
    masked_fixed
        the mask for the fixed image. If ``None`` a standard `edge_mask` will 
        be constructed.
    masked_moving:
        the mask for the moving image. If ``None`` is assumed to be the same as 
        `masked_fixed`.
    subpix
        The fraction of a pixel precision to estimate the shift to. Disable by 
        setting to `1`.
    low_pass
        The sigma value to Gaussian low-pass the NCC by. Generally low-pass 
        filtering reduces the image of shot noise in NCC, but it can also reduce
        the correlation on high-frequency features (which may also desireable).
        Typical values are in the range `[0.5, 2.0]`. Disable by setting to `0.0`.
    overlap_ratio
        The minimum overlap between `masked_fixed` and `masked_moving` to accept
        as valid, in terms of fraction of the total number of valid pixels in 
        the mask. Generally the user should not have to change this value.

    Returns
    -------
    shift
        The estimated shift between `fixed` and `moving`.
    ncc
        The calculated normalized, masked cross-correlation.

    Citation
    --------
    D. Padfield. Masked object registration in the Fourier domain. 
    IEEE Transactions on Image Processing, 21(5):2706-2718, 2012.
    """
    
    # We also want a faster version that just takes a stack of images and either a single 
    # mask or stack of masks and cross-correlates each image to the first one   
    shapeFixed = np.array(fixed.shape)
    shapeMoving = np.array(moving.shape)
    shapeCombined = shapeFixed + shapeMoving - 1

    eps = np.finfo(np.float64).resolution
    if fixed.dtype == np.float32:
        eps = np.finfo(np.float32).resolution
    
    shapeOptimized = _find_valid_fft_dim(shapeCombined)
    
    # Verify mask shapes and emptiness
    if mask_fixed is None:
        mask_fixed = edge_mask(fixed)
    if mask_moving is None:
        mask_moving = mask_fixed
    
    # Ensure masks are doubles 
    mask_fixed = np.array(mask_fixed > 0, dtype=fixed.dtype)
    mask_moving = np.array(mask_moving > 0, dtype=fixed.dtype)

    # Apply masks
    # print(f'fixed.shape: {fixed.shape}, {mask_fixed.shape}')
    # ne3.NumExpr('fixed = fixed * mask_fixed')()
    # ne3.NumExpr('moving = moving * mask_moving')()
    # Doing *= is in-place.
    fixed = fixed * mask_fixed
    moving = moving * mask_moving
    
    # Pre-flip (RAM: This may screw up your plans to accelerate the code)
    moving = np.rot90(moving, k=2)
    mask_moving = np.rot90(mask_moving, k=2)
    
    # Note that we can speed things up considerably by passing in FFTs of the masks and images, if they are re-used.
    # F_fixed_sqr = rfftn(ne3.NumExpr('fixed * fixed')(), shapeOptimized)
    # F_moving_sqr = rfftn(ne3.NumExpr('moving * moving')(), shapeOptimized)
    F_fixed_sqr = rfftn(fixed * fixed, shapeOptimized)
    F_moving_sqr = rfftn(moving * moving, shapeOptimized)

    # t0 = pc()
    F_fixed = rfftn(fixed, shapeOptimized)
    F_moving = rfftn(moving, shapeOptimized)
    F_mask_fixed = rfftn(mask_fixed, shapeOptimized)
    F_mask_moving = rfftn(mask_moving, shapeOptimized)
    # t1 = pc()
    # print(f'Estimated rFFT time: {(t1-t0)*3e3} ms')
    # About 65 ms, or 80 % of computation time is for the rFFTs, so the 
    # arithmetic is the other 20 %.
    # inv_phasecorr = irfftn(ne3.NumExpr('F_moving * F_fixed')())
    # inv_sqr_fixed = irfftn(ne3.NumExpr('F_mask_moving * F_fixed_sqr')())
    # inv_sqr_moving = irfftn(ne3.NumExpr('F_mask_fixed * F_moving_sqr')())
    inv_phasecorr = irfftn(F_moving * F_fixed)
    inv_sqr_fixed = irfftn(F_mask_moving * F_fixed_sqr)
    inv_sqr_moving = irfftn(F_mask_fixed * F_moving_sqr)
    
    # M1 * conj(M2)
    overlap_mask = irfftn(F_mask_fixed * F_mask_moving)
    # Limit the overlap to reasonable values
    overlap_mask = np.fmax(overlap_mask, eps)
    
    # Masks correlations
    # F_corr_mask_fixed = irfftn(ne3.NumExpr('F_mask_moving * F_fixed')())
    # F_corr_mask_moving = irfftn(ne3.NumExpr('F_mask_fixed * F_moving')())
    F_corr_mask_fixed = irfftn(F_mask_moving * F_fixed)
    F_corr_mask_moving = irfftn(F_mask_fixed * F_moving)

    # numerator = ne3.NumExpr('inv_phasecorr - F_corr_mask_fixed * F_corr_mask_moving / overlap_mask')()
    #
    # denom_fixed = ne3.NumExpr('inv_sqr_fixed - F_corr_mask_fixed * F_corr_mask_fixed / overlap_mask')()
    # denom_moving = ne3.NumExpr('inv_sqr_moving - F_corr_mask_moving * F_corr_mask_moving / overlap_mask')()

    numerator = inv_phasecorr - F_corr_mask_fixed * F_corr_mask_moving / overlap_mask

    denom_fixed = inv_sqr_fixed - F_corr_mask_fixed * F_corr_mask_fixed / overlap_mask
    denom_moving = inv_sqr_moving - F_corr_mask_moving * F_corr_mask_moving / overlap_mask

    # Crop denominator to be >= 0.0 for sqrt
    zero = np.float32(0.0)
    # denom = ne3.NumExpr('sqrt(fmax(denom_fixed * denom_moving, zero))')()
    denom = np.sqrt(np.fmax(denom_fixed * denom_moving, zero))
    # show(denom, 'realmncc denom', block=False)

    # TODO: any short-cut to estimate tolerance ahead of time? Could save an
    # extra fmax call.
    tolerance = 1e3 * np.spacing(denom.max())
    if fixed.dtype == np.float32:
        tolerance = np.float32(tolerance)
    else:
        tolerance = np.float64(tolerance)

    mask_not_zero = (denom > tolerance).astype(numerator.dtype)
    # C = ne3.NumExpr('(numerator / fmax(denom, tolerance)) * mask_not_zero')()
    C = (numerator / np.fmax(denom, tolerance)) * mask_not_zero

    # show(C, 'C for realmncc', block=False)

    # Need to further apply a mask based on overlap_mask and the desired minimum overlap_ratio
    C *= (overlap_mask >= overlap_ratio * np.max(overlap_mask))

    # Crop back to combinedSize
    diffShape = shapeCombined - shapeOptimized
    if diffShape.any() > 0:
        C = C[:diffShape[0], :diffShape[1]]

    # Low-pass filter the cross-correlation to suppress noise
    # (also suppresses influence of very high frequency features in specimen)
    if low_pass > 0.0:
        # print(f'Low pass filtering with kernel {low_pass}')
        C = ni.gaussian_filter(C, low_pass)

    # Find maximum in NCC
    max_pos = np.unravel_index(np.argmax(C), C.shape)
    translate = max_pos - (np.array(C.shape) - 1) / 2.0
    print(f'Found max position at {max_pos} px with value {C[max_pos[0], max_pos[1]]:.4f} yielding translation {translate}')

    # Fourier subpixel interpolation of NCC
    if subpix > 1:
        # try:
        F__C_sub = fft2(C[max_pos[0]-8: max_pos[0]+8, 
                        max_pos[1]-8: max_pos[1]+8])
        # except ValueError:
        #     # Here we sometimes, randomly, get round-off error with float32 
        #     # causing incorrect maximums
        #     # plt.figure()
        #     # plt.imshow(C)
        #     # plt.title(f'Error in max position: {max_pos}')
        #     # plt.show()
        #     print(f'Error in max position: {max_pos}')
        #     return -translate, C
            
        F__C_pad = np.zeros([8*subpix, 8*subpix], dtype=F__C_sub.dtype)
        # Copy the quadrants into the zero-padded oversampled Fourier transform
        # of the NCC
        F__C_pad[:9,  :9] = F__C_sub[:9,  :9]
        F__C_pad[-8:, :9] = F__C_sub[-8:, :9]
        F__C_pad[:9, -8:] = F__C_sub[:9, -8:]
        F__C_pad[-8:,-8:] = F__C_sub[-8:,-8:]

        # Calculate the oversampled NCC
        C_pad = ifft2(F__C_pad).real
        # show(C_pad)
        max_pos_pad = np.array(np.unravel_index(np.argmax(C_pad), C_pad.shape))

        fractional = (max_pos_pad) - np.array(C_pad.shape)/2.0
        fractional = (np.array(fractional) - 1) / subpix
        # print("Fractional position: ", fractional)
        translate += fractional

    return -translate, C

def polarncc(fixed: np.ndarray, moving: np.ndarray, 
             mask_fixed: np.ndarray=None, mask_moving: np.ndarray=None,
             subpix: int=8, low_pass: float=0.0, overlap_ratio: float=0.2,
             plot: bool=False) -> (List[int], np.ndarray):
    """
    As per `realmncc` but calculates the rotation and scale difference between 
    two images. Note that the masks are applied after `hihi.util.polar_transform` 
    is applied to `fixed` and `moving`. 

    Parameters
    ----------
    fixed
        a fixed (i.e. template) image
    moving
        a moving (i.e. base) image whose translation relative to `fixed` is to 
        be computed.
    masked_fixed
        the mask for the fixed image. If ``None`` a standard `edge_mask` will 
        be constructed.
    masked_moving:
        the mask for the moving image. If ``None`` is assumed to be the same as 
        `masked_fixed`.
    subpix
        The fraction of a pixel precision to estimate the shift to. Disable by 
        setting to `1`.
    low_pass
        The sigma value to Gaussian low-pass the NCC by. Generally low-pass 
        filtering reduces the image of shot noise in NCC, but it can also reduce
        the correlation on high-frequency features (which may also desireable).
        Typical values are in the range `[0.5, 2.0]`. Disable by setting to `0.0`.
    overlap_ratio
        The minimum overlap between `masked_fixed` and `masked_moving` to accept
        as valid, in terms of fraction of the total number of valid pixels in 
        the mask. Generally the user should not have to change this value.

    Returns
    -------
    rotation
        The estimated rotation between `fixed` and `moving` in radians.
    scale
        The estimated scale between `fixed` and `moving`. Do not expect high 
        accuracy from this value in this implementation.
    ncc
        The calculated normalized, masked cross-correlation.
    """
    N, M = fixed.shape

    # Generate a von Hann window
    xmesh, ymesh = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
    rmesh = np.sqrt(xmesh*xmesh + ymesh*ymesh).astype(fixed.dtype)

    von_hann = 0.5 + 0.5 * np.cos(2.0 * np.pi * rmesh / N)
    von_hann *= rmesh < N//2
    von_hann = von_hann.astype(fixed.dtype)

    filt_fixed = von_hann * fixed
    filt_moving = von_hann * moving

    F_fixed = np.abs(np.fft.fftshift(fft2(filt_fixed)))
    F_moving = np.abs(np.fft.fftshift(fft2(filt_moving)))

    # `linear` seems to be considerably better, but how does it change the 
    # constant `c` used to compute the scale?
    mode = 'linear'
    # May need to work more on the interpolation
    interp = 'bilinear'
    polar_fixed = hutil.polar_transform(F_fixed, phase_width=fixed.shape[0],
                mode=mode, interpolate=interp)
    polar_moving = hutil.polar_transform(F_moving, phase_width=fixed.shape[0],
                mode=mode, interpolate=interp)

    # Trying various masks here:
    # mask_fixed = mask_moving = edge_mask(polar_fixed, (0, 12, 0, 12))

    # Padfield algorithm
    # ==================
    shift, NCC = realmncc(polar_fixed, polar_moving, 
                         mask_fixed=mask_fixed, mask_moving=mask_moving,
                         subpix=subpix, low_pass=low_pass)
    rotation = -shift[1] / N * 2.0 * np.pi
    c = 5 * np.log10(N) / N
    scale = 1.0 + c * (shift[0]-1)
    scale = 1.0 / scale


    if plot:
        plt.figure()
        plt.subplot(231)
        plt.imshow(fixed, cmap='gray')
        plt.title('Fixed')

        plt.subplot(232)
        plt.imshow(moving, cmap='gray')
        plt.title('Moving')

        plt.subplot(234)
        plt.imshow(np.log1p(polar_fixed))
        plt.title('Fixed PolarFFT')

        plt.subplot(235)
        plt.imshow(np.log1p(polar_moving))
        plt.title('Moving PolarFFT')

        plt.subplot(236)
        plt.imshow(NCC)
        plt.title(f'NCC: {np.rad2deg(rotation):.1f}°, {scale*100:.1f} %')
        plt.show(block=False)

    return rotation, scale, NCC

'''
def polarpcc(fixed: np.ndarray, moving: np.ndarray, 
             subpix: int=8, low_pass: float=0.0, 
             plot: bool=False) -> (List[int], np.ndarray):
    """
    Calculates the rotation and scale difference between two images using a 
    conventional phase cross-correlation. 

    Parameters
    ----------
    fixed
        a fixed (i.e. template) image
    moving
        a moving (i.e. base) image whose translation relative to `fixed` is to 
        be computed.

    subpix
        The fraction of a pixel precision to estimate the shift to. Disable by 
        setting to `1`.
    low_pass
        The sigma value to Gaussian low-pass the NCC by. Generally low-pass 
        filtering reduces the image of shot noise in NCC, but it can also reduce
        the correlation on high-frequency features (which may also desireable).
        Typical values are in the range `[0.5, 2.0]`. Disable by setting to `0.0`.

    Returns
    -------
    rotation
        The estimated rotation between `fixed` and `moving` in radians.
    scale
        The estimated scale between `fixed` and `moving`. Do not expect high 
        accuracy from this value in this implementation.
    pcc
        The calculated phase cross-correlation.
    """
    N, M = fixed.shape

    # Generate a von Hann window
    xmesh, ymesh = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
    rmesh = np.sqrt(xmesh*xmesh + ymesh*ymesh).astype(fixed.dtype)

    von_hann = 0.5 + 0.5 * np.cos(2.0 * np.pi * rmesh / N)
    von_hann *= rmesh < N//2
    von_hann = von_hann.astype(fixed.dtype)

    filt_fixed = von_hann * fixed
    filt_moving = von_hann * moving

    F_fixed = np.abs(np.fft.fftshift(fft2(filt_fixed)))
    F_moving = np.abs(np.fft.fftshift(fft2(filt_moving)))

    # `linear` seems to be considerably better, but how does it change the 
    # constant `c` used to compute the scale?
    mode = 'linear'
    # May need to work more on the interpolation
    interp = 'bilinear'
    polar_fixed = hutil.polar_transform(F_fixed, phase_width=fixed.shape[0],
                mode=mode, interpolate=interp)
    polar_moving = hutil.polar_transform(F_moving, phase_width=fixed.shape[0],
                mode=mode, interpolate=interp)
    
    # Conventional Phase Correlation
    # ==============================
    # PCC may be better because we are rotationally periodic in the theta (x) axis.
    #
    # Low-pass seems to be ok for the theta-axis but bad for the scale-axis.
    shift, pcc, sub_pcc = hutil.phase_corr(polar_fixed, polar_moving, subpix=subpix, lowpass=low_pass, view=True)

    # print(f'Shift in polarpcc of {shift}')

    rotation = shift[1] / N * 2.0 * np.pi
    # Really not sure if I have the appropriate scale factor applied 
    c = np.sqrt(5) * np.log10(N) / N
    scale = 1.0 + c * (shift[0])

    if plot:
        plt.figure()
        plt.subplot(221)
        plt.imshow(fixed, cmap='gray')
        plt.title('Fixed')

        plt.subplot(222)
        plt.imshow(moving, cmap='gray')
        plt.title('Moving')

        plt.subplot(223)
        plt.imshow(np.log1p(polar_fixed))
        plt.title('Fixed PolarFFT')

        plt.subplot(224)
        plt.imshow(np.log1p(polar_moving))
        plt.title('Moving PolarFFT')

        plt.figure()
        plt.imshow(pcc)
        plt.title(f'PCC: {np.rad2deg(rotation):.1f}°, {scale*100:.1f} %')
        plt.show(block=False)

    return rotation, scale, pcc
'''