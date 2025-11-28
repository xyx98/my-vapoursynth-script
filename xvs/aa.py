from .utils import *
from .other import LDMerge
from .enhance import (
    FastLineDarkenMOD,
    xsUSM,
    mwenhance
)
from .dehalo import abcxyz

#modfied from old havsunc
def daa(
    c: vs.VideoNode,
    nsize: Optional[int] = None,
    nns: Optional[int] = None,
    qual: Optional[int] = None,
    pscrn: Optional[int] = None,
    int16_prescreener: Optional[bool] = None,
    int16_predictor: Optional[bool] = None,
    exp: Optional[int] = None,
    nnedi3_mode: str = "znedi3",
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    Anti-aliasing with contra-sharpening by Didée.

    It averages two independent interpolations, where each interpolation set works between odd-distanced pixels.
    This on its own provides sufficient amount of blurring. Enough blurring that the script uses a contra-sharpening step to counteract the blurring.
    '''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('daa: this is not a clip')

    nn=nnedi3(c,field=3,nsize=nsize, nns=nns, qual=qual, pscrn=pscrn,int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp,device=device,mode=nnedi3_mode)
    dbl = core.std.Merge(nn[::2], nn[1::2])
    dblD = core.std.MakeDiff(c, dbl)
    shrpD = core.std.MakeDiff(dbl, dbl.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1] if c.width > 1100 else [1, 2, 1, 2, 4, 2, 1, 2, 1]))
    DD = core.rgvs.Repair(shrpD, dblD, mode=13)
    return core.std.MergeDiff(dbl, DD)

def XSAA(src,nsize=None,nns=2,qual=None,aamode=-1,maskmode=1,nnedi3_mode="znedi3",device=-1,linedarken=False,preaa=0):
    """
    xyx98's simple aa function
    only process luma
    ####
    nsize,nns,qual: nnedi3 args
    aamode: decide how to aa. 0: merge two deinterlacing fleid ; 1: enlarge video and downscale
    maskmode: 0:no mask ; 1: use the clip before AA to create mask ; 2: use AAed clip to create mask; a video clip: use it as mask
    opencl: if True: use nnedi3cl; else use znedi3
    device:choose device for nnedi3cl
    linedarken: choose whether use FastLineDarkenMOD
    preaa: 0: no pre-AA ;
               1: using LDMerge in muvsfunc for pre-AA,use orginal clip when masked merge with AAed clip
               2: using LDMerge in muvsfunc for pre-AA,use pre-AAed clip when masked merge with AAed clip
    """

    w=src.width
    h=src.height
    if aamode==-1:
        enlarge = False if h>=720 else True
    elif aamode in (0,1):
        enlarge = bool(aamode)
    else:
        raise ValueError("")

    if src.format.color_family==vs.RGB:
        raise TypeError("RGB is unsupported")
    isYUV=src.format.color_family==vs.YUV
    Yclip = getY(src)

    if preaa in (1,2):
        horizontal = core.std.Convolution(Yclip, matrix=[1, 2, 7, 2, 1],mode='h')
        vertical = core.std.Convolution(Yclip, matrix=[1, 2, 7, 2, 1],mode='v')
        clip = LDMerge(horizontal, vertical, Yclip, mrad=1)
    elif preaa ==0:
        clip=Yclip
    else:
        raise ValueError("")
    if enlarge:
        if nsize is None:
            nsize = 1
        if qual is None:
            qual = 2
        aa=nnedi3(clip,dh=True,field=1,nsize=nsize,nns=nns,qual=qual,device=device,mode=nnedi3_mode)
        last=aa.resize.Spline36(w,h)
        t=last
    else:
        if nsize is None:
            nsize = 3
        if qual is None:
            qual = 1
        aa=nnedi3(clip,dh=False,field=3,nsize=nsize,nns=nns,qual=qual,device=device,mode=nnedi3_mode)
        last=core.std.Merge(aa[0::2], aa[1::2])
    if linedarken:
        last = FastLineDarkenMOD(last, strength=48, protection=5, luma_cap=191, threshold=5, thinning=0)


    if maskmode==1:
        mask=clip.tcanny.TCanny(sigma=1.5, t_h=20.0, t_l=8.0)
        mask=mt_expand_multi(mask, 'losange', planes=[0], sw=1, sh=1)
        if preaa==1:
            clip=Yclip
        last=core.std.MaskedMerge(clip, last, mask)
    elif maskmode==2:
        mask=last.tcanny.TCanny(sigma=1.5, t_h=20.0, t_l=8.0)
        mask=mt_expand_multi(mask, 'losange', planes=[0], sw=1, sh=1)
        if preaa==1:
            clip=Yclip
        last=core.std.MaskedMerge(clip, last, mask)
    elif isinstance(maskmode,vs.VideoNode):
        if preaa==1:
            clip=Yclip
        last=core.std.MaskedMerge(clip, last, maskmode)
    if isYUV:
        last = core.std.ShufflePlanes([last,src],[0,1,2], colorfamily=vs.YUV)
    return last

def mwaa(clip, aa_y=True, aa_c=False, cs_h=0, cs_v=0, aa_cmask=True, kernel_y=2, kernel_c=1, show=False,nnedi3_mode="znedi3",device=0):
    """
    Anti-Aliasing function
    Steal from other one's script. Most likely written by mawen1250.
    """
    if clip.format.bits_per_sample != 16:
        raise vs.Error('mwaa: Only 16bit supported')

    ## internal functions
    def aa_kernel_vertical(clip):
        clip_blk = clip.std.BlankClip(height=clip.height * 2)
        clip_y = getplane(clip, 0)
        if kernel_y == 2:
            clip_y = clip_y.eedi2.EEDI2(field=1)
        else:
            clip_y = nnedi3(clip_y,field=1, dh=True,device=device,mode=nnedi3_mode)
        clip_u = getplane(clip, 1)
        clip_v = getplane(clip, 2)
        if kernel_c == 2:
            clip_u = clip_u.eedi2.EEDI2(field=1)
            clip_v = clip_v.eedi2.EEDI2(field=1)
        else:
            clip_u = nnedi3(clip_u,field=1, dh=True,device=device,mode=nnedi3_mode)
            clip_v = nnedi3(clip_v,field=1, dh=True,device=device,mode=nnedi3_mode)
        return core.std.ShufflePlanes([clip_y if aa_y else clip_blk, clip_u if aa_c else clip_blk, clip_v if aa_c else clip_blk], [0,0,0] if aa_c else [0,1,2], vs.YUV)

    def aa_resample_vercial(clip, height, chroma_shift=0):
        return clip.fmtc.resample(h=height, sx=0, sy=[-0.5, -0.5 * (1 << clip.format.subsampling_h) - chroma_shift * 2], kernel=["spline36", "bicubic"], a1=0, a2=0.5, planes=[3 if aa_y else 1,3 if aa_c else 1,3 if aa_c else 1])

    ## parameters
    aa_cmask = aa_c and aa_cmask

    ## kernel
    aa = aa_resample_vercial(aa_kernel_vertical(clip.std.Transpose()), clip.width, cs_h)
    aa = aa_resample_vercial(aa_kernel_vertical(aa.std.Transpose()), clip.height, cs_v)

    ## mask
    aamask = clip.tcanny.TCanny(sigma=1.5, t_h=20.0, t_l=8.0, planes=[0])
    aamask = mt_expand_multi(aamask, 'losange', planes=[0], sw=1, sh=1)

    ## merge
    if aa_y:
        if aa_c:
            if aa_cmask:
                aa_merge = core.std.MaskedMerge(clip, aa, aamask, [0,1,2], True)
            else:
                aa_merge = core.std.MaskedMerge(clip, aa, aamask, [0], False)
                aa_merge = core.std.ShufflePlanes([aa_merge, aa], [0,1,2], vs.YUV)
        else:
            aa_merge = core.std.MaskedMerge(clip, aa, aamask, [0], False)
    else:
        if aa_c:
            if aa_cmask:
                aa_merge = core.std.MaskedMerge(clip, aa, aamask, [1,2], True)
            else:
                aa_merge = core.std.ShufflePlanes([clip, aa], [0,1,2], vs.YUV)
        else:
            aa_merge = clip

    ## output
    return aamask if show else aa_merge

def drAA(src,drf=0.5,lraa=True,nnedi3_mode="znedi3",device=-1,pp=True):
    """
    down resolution Anti-Aliasing for anime with heavy Aliasing
    only process luma
    #######
    drf:set down resolution factor,default is 0.5,range:0.5-1
    lraa:enable XSAA after down resolution,default is True
    opencl:use nnedi3cl and TcannyCL,default is False,means using znedi3 and Tcanny
    device:select device for opencl
    pp:enable post-process,sharpen、linedarken、dering,default is True
    """
    src=src.fmtc.bitdepth(bits=16)
    w=src.width
    h=src.height
    if src.format.color_family==vs.RGB:
        raise TypeError("RGB is unsupported")
    isYUV=src.format.color_family==vs.YUV
    Y = getY(src)
    if not 0.5<=drf<=1:
        raise ValueError("down resolution factor(drf) must between 0.5 and 1")

    ##aa
    aaY=core.resize.Bicubic(Y,int(w*drf),int(h*drf))
    if lraa:
        aaY=XSAA(aaY,aamode=0,preaa=2,maskmode=0,nsize=3,nnedi3_mode=nnedi3_mode,device=device)
    if nnedi3_mode=="nnedi3cl":
        aaY=nnedi3(aaY,field=1,dh=True,dw=True,nsize=3,nns=1,device=device,mode=nnedi3_mode)
    else:
        aaY=nnedi3(aaY,field=1,dh=True,nsize=3,nns=1,device=device,mode=nnedi3_mode).std.Transpose()
        aaY=nnedi3(aaY,field=1,dh=True,nsize=3,nns=1,device=device,mode=nnedi3_mode).std.Transpose()
    if int(w*drf)*2!=w or int(h*drf)*2!=h:
        aaY=core.resize.Spline36(aaY,w,h)

    ##mask
    mask1=core.std.Expr([aaY,Y],["x y - abs 16 * 12288 < 0 65535 ?"]).rgvs.RemoveGrain(3).rgvs.RemoveGrain(11)
    mask2=core.tcanny.TCanny(Y, sigma=0, mode=1,op=1,gmmax=30)
    mask2=core.std.Expr(mask2,"x 14000 < 0 65535 ?").rgvs.RemoveGrain(3).rgvs.RemoveGrain(11)
    mask=core.std.Expr([mask1,mask2],"x y min 256 < 0 65535 ?").rgvs.RemoveGrain(11)

    #pp
    if pp:
        aaY=FastLineDarkenMOD(aaY,strength=96, protection=3, luma_cap=200, threshold=5, thinning=24)
        aaY=xsUSM(aaY)
        aaY=abcxyz(aaY)
        #aaY=SADring(aaY)
        
        low=core.rgvs.RemoveGrain(aaY,11)
        hi=core.std.MakeDiff(aaY, low)
        en=mwenhance(hi,chroma=False,Strength=2.5)
        hi=LimitFilter(en,hi, thr=0.3, elast=8, brighten_thr=0.15)
        aaY=core.std.MergeDiff(aaY, hi)
        #merge
    Ylast=core.std.MaskedMerge(Y, aaY, mask)
    if isYUV:
        last= core.std.ShufflePlanes([Ylast,src], [0,1,2], vs.YUV)
    else:
        last= Ylast
    return last