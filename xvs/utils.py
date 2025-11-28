import vapoursynth as vs
import math
import functools
from typing import Callable, Optional, Sequence, Union, Any, List
from warnings import deprecated


core=vs.core
from .config import nnedi3_resample,Expr

PlanesType = Optional[Union[int, Sequence[int]]]
VSFuncType = Union[vs.Func, Callable[..., vs.VideoNode]]

#converting the values in one depth to another
def scale(i,depth_out=16,depth_in=8):
    return i*2**(depth_out-depth_in)

#getplane in YUV
def getplane(clip,i):
    return clip.std.ShufflePlanes(i, colorfamily=vs.GRAY)

def getY(clip):
    return clip.std.ShufflePlanes(0, colorfamily=vs.GRAY)

def getU(clip):
    return clip.std.ShufflePlanes(1, colorfamily=vs.GRAY)

def getV(clip):
    return clip.std.ShufflePlanes(2, colorfamily=vs.GRAY)

#extract all planes
def extractPlanes(clip):
    return tuple(clip.std.ShufflePlanes(x, colorfamily=vs.GRAY) for x in range(clip.format.num_planes))

#show plane in YUV(interger only)
def showY(clip):
    return Expr(clip,["",str(scale(128,clip.format.bits_per_sample))])

def showU(clip):
    return Expr(clip,["0","",str(scale(128,clip.format.bits_per_sample))])

def showV(clip):
    return Expr(clip,["0",str(scale(128,clip.format.bits_per_sample)),""])

def showUV(clip):
    return Expr(clip,["0",""])

#inpand/expand
def inpand(clip:vs.VideoNode,planes=0,thr=None,mode="square",cycle:int=1):
    modemap={
        "square":[1,1,1,1,1,1,1,1],
        "horizontal":[0,0,0,1,1,0,0,0],
        "vertical":[0,1,0,0,0,0,1,0],
        "both":[0,1,0,1,1,0,1,0],
    }

    if not (cd:=modemap.get(mode)):
        raise TypeError("unknown mode")
    
    for i in range(cycle):
        clip = core.std.Minimum(clip,planes,thr,cd)
    return clip

def expand(clip:vs.VideoNode,planes=0,thr=None,mode="square",cycle=1):
    modemap={
        "square":[1,1,1,1,1,1,1,1],
        "horizontal":[0,0,0,1,1,0,0,0],
        "vertical":[0,1,0,0,0,0,1,0],
        "both":[0,1,0,1,1,0,1,0],
    }

    if not (cd:=modemap.get(mode)):
        raise TypeError("unknown mode")
    
    for i in range(cycle):
        clip = core.std.Maximum(clip,planes,thr,cd)
    return clip

def getCSS(w,h):
    css={
        (1,1):"420",
        (1,0):"422",
        (0,0):"444",
        (2,2):"410",
        (2,0):"411",
        (0,1):"440"}
    sub=(w,h)
    if css.get(sub) is None:
        raise ValueError('Unknown subsampling')
    else:
        return css[sub]

def clip2css(clip):
    return getCSS(clip.format.subsampling_w,clip.format.subsampling_h)

def nnedi3(clip,field,dh=False,dw=False,planes=None,nsize=6,nns=1,qual=1,etype=0,pscrn=2,exp=0,
           mode="znedi3",device=-1,list_device=False,info=False,int16_predictor=True,int16_prescreener=True):
    mode=mode.lower()
    if mode=="nnedi3":
        return core.nnedi3.nnedi3(clip,field=field, dh=dh, planes=planes, nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn,exp=exp,int16_predictor=int16_predictor,int16_prescreener=int16_prescreener)
    elif mode=="znedi3":
        return core.znedi3.nnedi3(clip,field=field, dh=dh, planes=planes, nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn,exp=exp,int16_predictor=int16_predictor,int16_prescreener=int16_prescreener)
    elif mode=="nnedi3cl":
        return core.nnedi3cl.NNEDI3CL(clip,field=field, dh=dh, dw=dw, planes=planes, nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn,device=device,list_device=list_device,info=info)
    elif mode=="sneedif":
        return core.sneedif.NNEDI3(clip,field=field, dh=dh, dw=dw, planes=planes, nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn,device=device)
    else:
        raise ValueError("Unknown mode,mode must in ['nnedi3','nnedi3cl','znedi3','sneedif']")

def eedi3(clip, field, dh=None, planes=None, alpha=None, beta=None, gamma=None, nrad=None, mdis=None,
        hp=None, ucubic=None, cost3=None, vcheck=None, vthresh0=None, vthresh1=None, vthresh2=None, sclip=None,
        mode="eedi3m",device=None,list_device=None,info=None,opt=None):
    mode=mode.lower()
    if mode=="eedi3m":
        return clip.eedi3m.EEDI3(field, dh, planes, alpha, beta, gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2, sclip, opt)
    elif mode=="eedi3cl":
        return clip.eedi3m.EEDI3CL(field, dh, planes, alpha, beta, gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2, sclip, opt, device, list_device, info)
    elif mode=="eedi3":
        return clip.eedi3.eedi3(field, dh, planes, alpha, beta, gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2, sclip)
    else:
        raise ValueError("Unknown mode,mode must in ['eedi3m,eedi3,eedi3cl']")

def nlm(clip,d=1,a=2,s=4,h=12,channels='auto',wmode=0,wref=1.0,rclip=None,device_type="auto",device_id=0,ocl_x=0,ocl_y=0,ocl_r=0,info=False,num_streams=1,mode="nlm_ispc"):
    mode=mode.lower()
    if mode=="knl" or mode=="knlmeanscl":
        return core.knlm.KNLMeansCL(clip,d=d,a=a,s=s,h=h,channels=channels,wmode=wmode,wref=wref,rclip=rclip,device_type=device_type,device_id=device_id,ocl_x=ocl_x,ocl_y=ocl_y,ocl_r=ocl_r,info=info)
    elif mode=="nlm" or mode=="nlm_ispc":
        return core.nlm_ispc.NLMeans(clip,d=d,a=a,s=s,h=h,channels=channels,wmode=wmode,wref=wref,rclip=rclip)
    elif mode=="nlm_cuda":
        return core.nlm_cuda.NLMeans(clip,d=d,a=a,s=s,h=h,channels=channels,wmode=wmode,wref=wref,rclip=rclip,device_id=device_id,num_streams=num_streams)
    else:
        raise ValueError("Unknown mode,mode must in ['knlmeanscl','nlm_ispc','nlm_cuda']")
#copy from havsfunc
def mt_expand_multi(src, mode='rectangle', planes=None, sw=1, sh=1):
    """
    mt_expand_multi
    mt_inpand_multi

    Calls mt_expand or mt_inpand multiple times in order to grow or shrink
    the mask from the desired width and height.

    Parameters:
    - sw   : Growing/shrinking shape width. 0 is allowed. Default: 1
    - sh   : Growing/shrinking shape height. 0 is allowed. Default: 1
    - mode : "rectangle" (default), "ellipse" or "losange". Replaces the
        mt_xxpand mode. Ellipses are actually combinations of
        rectangles and losanges and look more like octogons.
        Losanges are truncated (not scaled) when sw and sh are not
        equal.
    Other parameters are the same as mt_xxpand.
    """
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_expand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_expand_multi(src.std.Maximum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src

#copy from havsfunc
def mt_inpand_multi(src, mode='rectangle', planes=None, sw=1, sh=1):
    """
    mt_expand_multi
    mt_inpand_multi

    Calls mt_expand or mt_inpand multiple times in order to grow or shrink
    the mask from the desired width and height.

    Parameters:
    - sw   : Growing/shrinking shape width. 0 is allowed. Default: 1
    - sh   : Growing/shrinking shape height. 0 is allowed. Default: 1
    - mode : "rectangle" (default), "ellipse" or "losange". Replaces the
        mt_xxpand mode. Ellipses are actually combinations of
        rectangles and losanges and look more like octogons.
        Losanges are truncated (not scaled) when sw and sh are not
        equal.
    Other parameters are the same as mt_xxpand.
    """
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inpand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_inpand_multi(src.std.Minimum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src

def mt_inflate_multi(src, planes=None, radius=1):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inflate_multi: this is not a clip')

    for i in range(radius):
        src = core.std.Inflate(src, planes=planes)
    return src

def mt_deflate_multi(src, planes=None, radius=1):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_deflate_multi: this is not a clip')

    for i in range(radius):
        src = core.std.Deflate(src, planes=planes)
    return src

def LimitFilter(flt, src, ref=None, thr=None, elast=None, brighten_thr=None, thrc=None, planes=None,use_vszip=None):
    """
    Similar to the AviSynth function Dither_limit_dif16() and HQDeringmod_limit_dif16().
    It acts as a post-processor, and is very useful to limit the difference of filtering while avoiding artifacts.
    Commonly used cases:
       de-banding
       de-ringing
       de-noising
       sharpening
       combining high precision source with low precision filtering: mvf.LimitFilter(src, flt, thr=1.0, elast=2.0)
    ############################################################################################################################
    From mvsfunc, and only Expr implementation left,if vszip installed,will use vszip.LimitFilter.
    ############################################################################################################################
    Algorithm for Y/R/G/B plane (for chroma, replace "thr" and "brighten_thr" with "thrc")
       dif = flt - src
       dif_ref = flt - ref
       dif_abs = abs(dif_ref)
       thr_1 = brighten_thr if (dif > 0) else thr
       thr_2 = thr_1 * elast

       if dif_abs <= thr_1:
           final = flt
       elif dif_abs >= thr_2:
           final = src
       else:
           final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
    ############################################################################################################################
    Basic parameters
       flt {clip}: filtered clip, to compute the filtering diff
           can be of YUV/RGB/Gray color family, can be of 8-16 bit integer or 16/32 bit float
       src {clip}: source clip, to apply the filtering diff
           must be of the same format and dimension as "flt"
       ref {clip} (optional): reference clip, to compute the weight to be applied on filtering diff
           must be of the same format and dimension as "flt"
           default: None (use "src")
       thr {float}: threshold (8-bit scale) to limit filtering diff
           default: 1.0
       elast {float}: elasticity of the soft threshold
           default: 2.0
       planes {int[]}: specify which planes to process
           unprocessed planes will be copied from "flt"
           default: all planes will be processed, [0,1,2] for YUV/RGB input, [0] for Gray input
    ############################################################################################################################
    Advanced parameters
       brighten_thr {float}: threshold (8-bit scale) for filtering diff that brightening the image (Y/R/G/B plane)
           set a value different from "thr" is useful to limit the overshoot/undershoot/blurring introduced in sharpening/de-ringing
           default is the same as "thr"
       thrc {float}: threshold (8-bit scale) for chroma (U/V/Co/Cg plane)
           default is the same as "thr"
    """
    # input clip
    if not isinstance(flt, vs.VideoNode):
        raise vs.Error('LimitFilter:"flt" must be a clip!')
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('LimitFilter:"src" must be a clip!')
    if ref is not None and not isinstance(ref, vs.VideoNode):
        raise vs.Error('LimitFilter:"ref" must be a clip!')

    # Get properties of input clip
    sFormat = flt.format
    if sFormat.id != src.format.id:
        raise vs.Error('LimitFilter:"flt" and "src" must be of the same format!')
    if flt.width != src.width or flt.height != src.height:
        raise vs.Error('LimitFilter:"flt" and "src" must be of the same width and height!')

    if ref is not None:
        if sFormat.id != ref.format.id:
            raise vs.Error('LimitFilter:"flt" and "ref" must be of the same format!')
        if flt.width != ref.width or flt.height != ref.height:
            raise vs.Error('LimitFilter:"flt" and "ref" must be of the same width and height!')

    sColorFamily = sFormat.color_family
    #CheckColorFamily(sColorFamily)
    sIsYUV = sColorFamily == vs.YUV

    sSType = sFormat.sample_type
    sbitPS = sFormat.bits_per_sample
    sNumPlanes = sFormat.num_planes

    # Parameters
    if thr is None:
        thr = 1.0
    elif isinstance(thr, int) or isinstance(thr, float):
        if thr < 0:
            raise vs.Error('valid range of "thr" is [0, +inf)')
    else:
        raise vs.Error('"thr" must be an int or a float!')

    if elast is None:
        elast = 2.0
    elif isinstance(elast, int) or isinstance(elast, float):
        if elast < 1:
            raise vs.Error('valid range of "elast" is [1, +inf)')
    else:
        raise vs.Error('"elast" must be an int or a float!')

    if brighten_thr is None:
        brighten_thr = thr
    elif isinstance(brighten_thr, int) or isinstance(brighten_thr, float):
        if brighten_thr < 0:
            raise vs.Error('valid range of "brighten_thr" is [0, +inf)')
    else:
        raise vs.Error('"brighten_thr" must be an int or a float!')

    if thrc is None:
        thrc = thr
    elif isinstance(thrc, int) or isinstance(thrc, float):
        if thrc < 0:
            raise vs.Error('valid range of "thrc" is [0, +inf)')
    else:
        raise vs.Error('"thrc" must be an int or a float!')

    if use_vszip is None and hasattr(core,"vszip") and hasattr(core.vszip,"LimitFilter"):
        use_vszip=True
    else:
        use_vszip=False

    # planes
    process = [0,0,0]

    if planes is None:
        process = [1,1,1]
    elif isinstance(planes, int):
        if planes < 0 or planes >= 3:
            raise vs.Error(f'valid range of "planes" is [0, 3)!')
        process[planes] = 1
    elif isinstance(planes, Sequence):
        for p in planes:
            if not isinstance(p, int):
                raise vs.Error('"planes" must be a (sequence of) int!')
            elif p < 0 or p >= 3:
                raise vs.Error(f'valid range of "planes" is [0, 3)!')
            process[p] = 1
    else:
        raise vs.Error('"planes" must be a (sequence of) int!')

    if use_vszip:
        return core.vszip.LimitFilter(flt=flt,src=src,ref=ref,dark_thr=[thr,thrc],bright_thr=[brighten_thr,thrc],elast=elast,planes=planes)
    else:
        # Process
        if thr <= 0 and brighten_thr <= 0:
            if sIsYUV:
                if thrc <= 0:
                    return src
            else:
                return src
        if thr >= 255 and brighten_thr >= 255:
            if sIsYUV:
                if thrc >= 255:
                    return flt
            else:
                return flt

        valueRange = (1 << sbitPS) - 1 if sSType == vs.INTEGER else 1
        limitExprY = _limit_filter_expr(ref is not None, thr, elast, brighten_thr, valueRange)
        limitExprC = _limit_filter_expr(ref is not None, thrc, elast, thrc, valueRange)
        expr = []
        for i in range(sNumPlanes):
            if process[i]:
                if i > 0 and (sIsYUV):
                    expr.append(limitExprC)
                else:
                    expr.append(limitExprY)
            else:
                expr.append("")

        if ref is None:
            clip = Expr([flt, src], expr)
        else:
            clip = Expr([flt, src, ref], expr)
        return clip

def _limit_filter_expr(defref, thr, elast, largen_thr, value_range):
    flt = " x "
    src = " y "
    ref = " z " if defref else src

    dif = f" {flt} {src} - "
    dif_ref = f" {flt} {ref} - "
    dif_abs = dif_ref + " abs "

    thr = thr * value_range / 255
    largen_thr = largen_thr * value_range / 255

    if thr <= 0 and largen_thr <= 0:
        limitExpr = f" {src} "
    elif thr >= value_range and largen_thr >= value_range:
        limitExpr = ""
    else:
        if thr <= 0:
            limitExpr = f" {src} "
        elif thr >= value_range:
            limitExpr = f" {flt} "
        elif elast <= 1:
            limitExpr = f" {dif_abs} {thr} <= {flt} {src} ? "
        else:
            thr_1 = thr
            thr_2 = thr * elast
            thr_slope = 1 / (thr_2 - thr_1)
            # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
            limitExpr = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
            limitExpr = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExpr + " ? ? "

        if largen_thr != thr:
            if largen_thr <= 0:
                limitExprLargen = f" {src} "
            elif largen_thr >= value_range:
                limitExprLargen = f" {flt} "
            elif elast <= 1:
                limitExprLargen = f" {dif_abs} {largen_thr} <= {flt} {src} ? "
            else:
                thr_1 = largen_thr
                thr_2 = largen_thr * elast
                thr_slope = 1 / (thr_2 - thr_1)
                # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
                limitExprLargen = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
                limitExprLargen = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExprLargen + " ? ? "
            limitExpr = f" {flt} {ref} > " + limitExprLargen + " " + limitExpr + " ? "

    return limitExpr

def cround(x):
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def m4(x):
    return 16 if x < 16 else cround(x / 4) * 4

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255

def Padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    width = clip.width + left + right
    height = clip.height + top + bottom

    return clip.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)
