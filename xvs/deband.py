from .utils import *
from .other import lowbitdepth_sim

@deprecated("No maintenance.")
def SAdeband(src,thr=128,f3kdb_arg={},smoothmask=0,Sdbclip=None,Smask=None,tvrange=True):
    """
    Simple Adaptive Debanding
    -------------------------------------
    thr: only pixel less then will be processed(only affect luma)，default is 128 and vaule is based on 8bit
    f3kdb_arg:use a dict to set parameters of f3kdb
    smoothmask: -1: don't smooth the mask; 0: use removegrain mode11;
                         1: use removegrain mode20; 2: use removegrain mode19
                  a list :use Convolution,and this list will be the matrix
                       default is 0
    Sdbclip: input and use your own debanded clip,must be 16bit
    Smask:input and use your own mask clip,must be 16bit,and 0 means don't process
    tvrange: set false if your clip is pcrange
    """
    clip = src if src.format.bits_per_sample==16 else core.fmtc.bitdepth(src,bits=16)
    db = core.f3kdb.Deband(clip,output_depth=16,**f3kdb_arg) if Sdbclip is None else Sdbclip
    expr = "x {thr} > 0 65535 x {low} - 65535 * {thr} {low} - / - ?".format(thr=scale(thr),low=scale(16) if tvrange else 0)
    mask = core.std.Expr(clip,[expr,'',''])
    if smoothmask==-1:
        mask = mask
    elif smoothmask==0:
        mask = core.rgvs.RemoveGrain(mask,[11,0,0])
    elif smoothmask==1:
        mask = core.rgvs.RemoveGrain(mask,[20,0,0])
    elif smoothmask==2:
        mask = core.rgvs.RemoveGrain(mask,[19,0,0])
    elif isinstance(smoothmask,list):
        mask = core.std.Convolution(mask,matrix=smoothmask,planes=[0])
    else:
        raise TypeError("")
    merge = core.std.MaskedMerge(clip, db, mask, planes=0)
    return core.std.ShufflePlanes([merge,db],[0,1,2], colorfamily=vs.YUV)

@deprecated("Strange idea,not really useful.")
def lbdeband(clip:vs.VideoNode,dbit=6):
    """
    low bitdepth deband
    deband for flat area with heavy details through round to low bitdepth,limitfilter and f3kdb
    only procress luma when YUV,no direct support for RGB.
    You need use trim,mask or other way you can to protect the area without heavy banding.
    """

    if clip.format.color_family==vs.RGB:
        raise TypeError("RGB is unsupported")
    isGary=clip.format.color_family==vs.GRAY
    clip=core.fmtc.bitdepth(clip,bits=16)
    luma=clip if isGary else getY(clip)
    down=lowbitdepth_sim(luma,dbit,dither=1).f3kdb.Deband(31, 64, 0, 0, 0, 0, output_depth=16)
    deband=LimitFilter(down, luma, thr=0.2, elast=8.0).f3kdb.Deband(31, 64, 0, 0, 0, 0, output_depth=16).f3kdb.Deband(15, 64, 0, 0, 0, 0, output_depth=16)
    if isGary:
        return deband
    else:
        return core.std.ShufflePlanes([deband,clip], [0,1,2], vs.YUV)
    
