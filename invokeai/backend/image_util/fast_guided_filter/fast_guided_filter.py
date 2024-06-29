# ruff: noqa: E741
# -*- coding: utf-8 -*-
## @package guided_filter.core.filters
#
#  Implementation of guided filter.
#  * GuidedFilter: Original guided filter.
#  * FastGuidedFilter: Fast version of the guided filter.
#  @author      tody
#  @date        2015/08/26


import cv2
import numpy as np


## Convert image into float32 type.
def to32F(img):
    if img.dtype == np.float32:
        return img
    return (1.0 / 255.0) * np.float32(img)


## Convert image into uint8 type.
def to8U(img):
    if img.dtype == np.uint8:
        return img
    return np.clip(np.uint8(255.0 * img), 0, 255)


## Return if the input image is gray or not.
def _isGray(I):
    return len(I.shape) == 2


## Return down sampled image.
#  @param scale (w/s, h/s) image will be created.
#  @param shape I.shape[:2]=(h, w). numpy friendly size parameter.
def _downSample(I, scale=4, shape=None):
    if shape is not None:
        h, w = shape
        return cv2.resize(I, (w, h), interpolation=cv2.INTER_NEAREST)

    h, w = I.shape[:2]
    return cv2.resize(I, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_NEAREST)


## Return up sampled image.
#  @param scale (w*s, h*s) image will be created.
#  @param shape I.shape[:2]=(h, w). numpy friendly size parameter.
def _upSample(I, scale=2, shape=None):
    if shape is not None:
        h, w = shape
        return cv2.resize(I, (w, h), interpolation=cv2.INTER_LINEAR)

    h, w = I.shape[:2]
    return cv2.resize(I, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


## Fast guide filter.
class FastGuidedFilter:
    ## Constructor.
    #  @param I Input guidance image. Color or gray.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    #  @param scale Down sampled scale.
    def __init__(self, I, radius=5, epsilon=0.4, scale=4):
        I_32F = to32F(I)
        self._I = I_32F
        h, w = I.shape[:2]

        I_sub = _downSample(I_32F, scale)

        self._I_sub = I_sub
        radius = int(radius / scale)

        if _isGray(I):
            self._guided_filter = GuidedFilterGray(I_sub, radius, epsilon)
        else:
            self._guided_filter = GuidedFilterColor(I_sub, radius, epsilon)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        p_32F = to32F(p)
        shape_original = p.shape[:2]

        p_sub = _downSample(p_32F, shape=self._I_sub.shape[:2])

        if _isGray(p_sub):
            return self._filterGray(p_sub, shape_original)

        cs = p.shape[2]
        q = np.array(p_32F)

        for ci in range(cs):
            q[:, :, ci] = self._filterGray(p_sub[:, :, ci], shape_original)
        return to8U(q)

    def _filterGray(self, p_sub, shape_original):
        ab_sub = self._guided_filter._computeCoefficients(p_sub)
        ab = [_upSample(abi, shape=shape_original) for abi in ab_sub]
        return self._guided_filter._computeOutput(ab, self._I)


## Guide filter.
class GuidedFilter:
    ## Constructor.
    #  @param I Input guidance image. Color or gray.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.4):
        I_32F = to32F(I)

        if _isGray(I):
            self._guided_filter = GuidedFilterGray(I_32F, radius, epsilon)
        else:
            self._guided_filter = GuidedFilterColor(I_32F, radius, epsilon)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return to8U(self._guided_filter.filter(p))


## Common parts of guided filter.
#
#  This class is used by guided_filter class. GuidedFilterGray and GuidedFilterColor.
#  Based on guided_filter._computeCoefficients, guided_filter._computeOutput,
#  GuidedFilterCommon.filter computes filtered image for color and gray.
class GuidedFilterCommon:
    def __init__(self, guided_filter):
        self._guided_filter = guided_filter

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        p_32F = to32F(p)
        if _isGray(p_32F):
            return self._filterGray(p_32F)

        cs = p.shape[2]
        q = np.array(p_32F)

        for ci in range(cs):
            q[:, :, ci] = self._filterGray(p_32F[:, :, ci])
        return q

    def _filterGray(self, p):
        ab = self._guided_filter._computeCoefficients(p)
        return self._guided_filter._computeOutput(ab, self._guided_filter._I)


## Guided filter for gray guidance image.
class GuidedFilterGray:
    #  @param I Input gray guidance image.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.4):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = to32F(I)
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        I = self._I
        r = self._radius
        self._I_mean = cv2.blur(I, (r, r))
        I_mean_sq = cv2.blur(I**2, (r, r))
        self._I_var = I_mean_sq - self._I_mean**2

    def _computeCoefficients(self, p):
        r = self._radius
        p_mean = cv2.blur(p, (r, r))
        p_cov = p_mean - self._I_mean * p_mean
        a = p_cov / (self._I_var + self._epsilon)
        b = p_mean - a * self._I_mean
        a_mean = cv2.blur(a, (r, r))
        b_mean = cv2.blur(b, (r, r))
        return a_mean, b_mean

    def _computeOutput(self, ab, I):
        a_mean, b_mean = ab
        return a_mean * I + b_mean


## Guided filter for color guidance image.
class GuidedFilterColor:
    #  @param I Input color guidance image.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.2):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = to32F(I)
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir**2, (r, r)) - self._Ir_mean**2 + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        Irr_inv /= I_cov
        Irg_inv /= I_cov
        Irb_inv /= I_cov
        Igg_inv /= I_cov
        Igb_inv /= I_cov
        Ibb_inv /= I_cov

        self._Irr_inv = Irr_inv
        self._Irg_inv = Irg_inv
        self._Irb_inv = Irb_inv
        self._Igg_inv = Igg_inv
        self._Igb_inv = Igb_inv
        self._Ibb_inv = Ibb_inv

    def _computeCoefficients(self, p):
        r = self._radius
        I = self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        p_mean = cv2.blur(p, (r, r))

        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov
        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar_mean = cv2.blur(ar, (r, r))
        ag_mean = cv2.blur(ag, (r, r))
        ab_mean = cv2.blur(ab, (r, r))
        b_mean = cv2.blur(b, (r, r))

        return ar_mean, ag_mean, ab_mean, b_mean

    def _computeOutput(self, ab, I):
        ar_mean, ag_mean, ab_mean, b_mean = ab

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean

        return q
