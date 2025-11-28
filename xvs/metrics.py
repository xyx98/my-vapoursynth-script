from .utils import *

def _IQA_downsample(clip: vs.VideoNode) -> vs.VideoNode:
    """Downsampler for image quality assessment model.

    The “clip” is first filtered by a 2x2 average filter, and then down-sampled by a factor of 2.
    """

    return core.std.Convolution(clip, [1, 1, 0, 1, 1, 0, 0, 0, 0]).resize.Point(clip.width // 2, clip.height // 2, src_left=-1, src_top=-1)

#modified from muvsfunc
def SSIM(clip1: vs.VideoNode, clip2: vs.VideoNode, plane: Optional[int] = None,
         downsample: bool = True, k1: float = 0.01, k2: float = 0.03,
         fun: Optional[VSFuncType] = None, dynamic_range: int = 1,
         show_map: bool = False) -> vs.VideoNode:
    """Structural SIMilarity Index Calculator

    The Structural SIMilarity (SSIM) index is a method for measuring the similarity between two images.
    It is based on the hypothesis that the HVS is highly adapted for extracting structural information,
    which compares local patterns of pixel intensities that have been normalized for luminance and contrast.

    The mean SSIM (MSSIM) index value of the distorted image will be stored as frame property 'PlaneSSIM' in the output clip.

    The value of SSIM measures the structural similarity in an image.
    The higher the SSIM score, the higher the image perceptual quality.
    If "clip1" == "clip2", SSIM = 1.

    All the internal calculations are done at 32-bit float, only one channel of the image will be processed.

    Args:
        clip1: The distorted clip, will be copied to output if "show_map" is False.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        plane: (int) Specify which plane to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        k1, k2: (float) Constants in the SSIM index formula.
            According to the paper, the performance of the SSIM index algorithm is fairly insensitive to variations of these values.
            Default are 0.01 and 0.03.

        fun: (function or float) The function of how the clips are filtered.
            If it is None, it will be set to a gaussian filter whose standard deviation is 1.5.
            Note that the size of gaussian kernel is different from the one in MATLAB.
            If it is a float, it specifies the standard deviation of the gaussian filter. (sigma in core.tcanny.TCanny)
            According to the paper, the quality map calculated from gaussian filter exhibits a locally isotropic property,
            which prevents the present of undesirable “blocking” artifacts in the resulting SSIM index map.
            Default is None.

        dynamic_range: (float) Dynamic range of the internal float point clip. Default is 1.

        show_map: (bool) Whether to return SSIM index map. If not, "clip1" will be returned. Default is False.

    Ref:
        [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), 600-612.
        [2] https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    """

    funcName = 'SSIM'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')
    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if clip1.format.id != clip2.format.id:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same format!')
    if clip1.width != clip2.width or clip1.height != clip2.height:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same width and height!')

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    if fun is None:
        fun = functools.partial(core.tcanny.TCanny, sigma=1.5, mode=-1)
    elif isinstance(fun, (int, float)):
        fun = functools.partial(core.tcanny.TCanny, sigma=fun, mode=-1)
    elif not callable(fun):
        raise TypeError(funcName + ': \"fun\" must be a function or a float!')

    # Store the "clip1"
    clip1_src = clip1

    # Convert to float type grayscale image
    clip1 = getplane(clip1, plane)
    clip2 = getplane(clip2, plane)
    clip1 = core.fmtc.bitdepth(clip1,bits=32)
    clip2 = core.fmtc.bitdepth(clip2,bits=32)

    # Filtered by a 2x2 average filter and then down-sampled by a factor of 2
    if downsample:
        clip1 = _IQA_downsample(clip1)
        clip2 = _IQA_downsample(clip2)

    # Core algorithm
    mu1 = fun(clip1)
    mu2 = fun(clip2)
    mu1_sq = Expr([mu1], ['x dup *'])
    mu2_sq = Expr([mu2], ['x dup *'])
    mu1_mu2 = Expr([mu1, mu2], ['x y *'])
    sigma1_sq_pls_mu1_sq = fun(Expr([clip1], ['x dup *']))
    sigma2_sq_pls_mu2_sq = fun(Expr([clip2], ['x dup *']))
    sigma12_pls_mu1_mu2 = fun(Expr([clip1, clip2], ['x y *']))

    if c1 > 0 and c2 > 0:
        expr = '2 x * {c1} + 2 y x - * {c2} + * z a + {c1} + b c - d e - + {c2} + * /'.format(c1=c1, c2=c2)
        expr_clips = [mu1_mu2, sigma12_pls_mu1_mu2, mu1_sq, mu2_sq, sigma1_sq_pls_mu1_sq, mu1_sq, sigma2_sq_pls_mu2_sq, mu2_sq]
        ssim_map = Expr(expr_clips, [expr])
    else:
        denominator1 = Expr([mu1_sq, mu2_sq], ['x y + {c1} +'.format(c1=c1)])
        denominator2 = Expr([sigma1_sq_pls_mu1_sq, mu1_sq, sigma2_sq_pls_mu2_sq, mu2_sq], ['x y - z a - + {c2} +'.format(c2=c2)])

        numerator1_expr = '2 z * {c1} +'.format(c1=c1)
        numerator2_expr = '2 a z - * {c2} +'.format(c2=c2)
        expr = 'x y * 0 > {numerator1} {numerator2} * x y * / x 0 = not y 0 = and {numerator1} x / {i} ? ?'.format(numerator1=numerator1_expr,
            numerator2=numerator2_expr, i=1)
        ssim_map = Expr([denominator1, denominator2, mu1_mu2, sigma12_pls_mu1_mu2], [expr])

    # The following code is modified from mvf.PlaneStatistics(), which is used to compute the mean of the SSIM index map as MSSIM
    map_mean = core.std.PlaneStats(ssim_map, plane=0, prop='PlaneStats')

    def _PlaneSSIMTransfer(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['PlaneSSIM'] = f[1].props['PlaneStatsAverage']
        return fout

    output_clip = ssim_map if show_map else clip1_src
    output_clip = core.std.ModifyFrame(output_clip, [output_clip, map_mean], selector=_PlaneSSIMTransfer)

    return output_clip

#modified from muvsfunc
def GMSD(clip1: vs.VideoNode, clip2: vs.VideoNode, plane: Optional[int] = None,
         downsample: bool = True, c: float = 0.0026, show_map: bool = False
         ) -> vs.VideoNode:
    """Gradient Magnitude Similarity Deviation Calculator

    GMSD is a new effective and efficient image quality assessment (IQA) model, which utilizes the pixel-wise gradient magnitude similarity (GMS)
    between the reference and distorted images combined with standard deviation of the GMS map to predict perceptual image quality.

    The distortion degree of the distorted image will be stored as frame property 'PlaneGMSD' in the output clip.

    The value of GMSD reflects the range of distortion severities in an image.
    The lowerer the GMSD score, the higher the image perceptual quality.
    If "clip1" == "clip2", GMSD = 0.

    All the internal calculations are done at 32-bit float, only one channel of the image will be processed.

    Args:
        clip1: The distorted clip, will be copied to output if "show_map" is False.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        plane: (int) Specify which plane to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        c: (float) A positive constant that supplies numerical stability.
            According to the paper, for all the test databases, GMSD shows similar preference to the value of c.
            Default is 0.0026.

        show_map: (bool) Whether to return GMS map. If not, "clip1" will be returned. Default is False.

    Ref:
        [1] Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014). Gradient magnitude similarity deviation:
            A highly efficient perceptual image quality index. IEEE Transactions on Image Processing, 23(2), 684-695.
        [2] http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm.

    """

    funcName = 'GMSD'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')
    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if clip1.format.id != clip2.format.id:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same format!')
    if clip1.width != clip2.width or clip1.height != clip2.height:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same width and height!')

    # Store the "clip1"
    clip1_src = clip1

    # Convert to float type grayscale image
    clip1 = getplane(clip1, plane)
    clip2 = getplane(clip2, plane)
    clip1 = core.fmtc.bitdepth(clip1,bits=32)
    clip2 = core.fmtc.bitdepth(clip2,bits=32)

    # Filtered by a 2x2 average filter and then down-sampled by a factor of 2, as in the implementation of SSIM
    if downsample:
        clip1 = _IQA_downsample(clip1)
        clip2 = _IQA_downsample(clip2)

    # Calculate gradients based on Prewitt filter
    clip1_dx = core.std.Convolution(clip1, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
    clip1_dy = core.std.Convolution(clip1, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
    clip1_grad_squared = Expr([clip1_dx, clip1_dy], ['x dup * y dup * +'])

    clip2_dx = core.std.Convolution(clip2, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
    clip2_dy = core.std.Convolution(clip2, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
    clip2_grad_squared = Expr([clip2_dx, clip2_dy], ['x dup * y dup * +'])

    # Compute the gradient magnitude similarity (GMS) map
    quality_map = Expr([clip1_grad_squared, clip2_grad_squared], ['2 x y * sqrt * {c} + x y + {c} + /'.format(c=c)])

    # The following code is modified from mvf.PlaneStatistics(), which is used to compute the standard deviation of the GMS map as GMSD
    map_mean = core.std.PlaneStats(quality_map, plane=0, prop='PlaneStats')

    def _PlaneSDFrame(n: int, f: vs.VideoFrame, clip: vs.VideoNode, core: vs.Core) -> vs.VideoNode:
        mean = f.props['PlaneStatsAverage']
        expr = "x {mean} - dup *".format(mean=mean)
        return Expr(clip, expr)
    if hasattr(core,"akarin"):
        mean = "y.PlaneStatsAverage"
        SDclip = core.akarin.Expr([quality_map, map_mean], "x {mean} - dup *".format(mean=mean))
    else:
        SDclip = core.std.FrameEval(quality_map, functools.partial(_PlaneSDFrame, clip=quality_map, core=core), map_mean)

    SDclip = core.std.PlaneStats(SDclip, plane=0, prop='PlaneStats')

    def _PlaneGMSDTransfer(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['PlaneGMSD'] = math.sqrt(f[1].props['PlaneStatsAverage']) # type: ignore
        return fout
    output_clip = quality_map if show_map else clip1_src
    output_clip = core.std.ModifyFrame(output_clip, [output_clip, SDclip], selector=_PlaneGMSDTransfer)

    return output_clip


def getsharpness(clip,show=False,usePropExpr=False):
    luma=getY(clip).fmtc.bitdepth(bits=16)
    blur=core.rgvs.RemoveGrain(luma, 20)
    dif=Expr([luma,blur],[f"x y - 65535 / 2 pow 65535 *"])
    dif=core.std.PlaneStats(dif)

    if hasattr(core,"akarin") and hasattr(core.akarin,"PropExpr") and usePropExpr:
        last=core.akarin.PropExpr([clip,dif],lambda: dict(sharpness='y.PlaneStatsAverage 65535 *'))
    else:
        def calc(n,f): 
            fout=f[1].copy()
            fout.props["sharpness"]=f[0].props["PlaneStatsAverage"]*65535
            return fout

        last=core.std.ModifyFrame(clip,[dif,clip],calc)
    return core.text.FrameProps(last,"sharpness",scale=2) if show else last


