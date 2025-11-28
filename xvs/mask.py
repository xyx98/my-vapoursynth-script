from .utils import *

#from old havsfunc.
@deprecated("Use VapourSynth-EdgeMasks instead.")
def AvsPrewitt(clip, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AvsPrewitt: this is not a clip')

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    return  Expr([
        clip.std.Convolution(matrix=[1, 1, 0, 1, 0, -1, 0, -1, -1], planes=planes, saturate=False),
        clip.std.Convolution(matrix=[1, 1, 1, 0, 0, 0, -1, -1, -1], planes=planes, saturate=False),
        clip.std.Convolution(matrix=[1, 0, -1, 1, 0, -1, 1, 0, -1], planes=planes, saturate=False),
        clip.std.Convolution(matrix=[0, -1, -1, 1, 0, -1, 1, 1, 0], planes=planes, saturate=False)
        ],
        expr=['x y max z max a max' if i in planes else '' for i in range(clip.format.num_planes)])

def creditmask(clip,nclip,mode=0):
    """
    use non-credit clip to create mask for credit area
    255(8bit) means credit
    output will be 16bit
    ####
    mode: 0: only use Y to create mask ; 1: use all planes to create mask
              only affect the yuv input
    """

    clip = core.fmtc.bitdepth(clip,bits=16)
    nclip = core.fmtc.bitdepth(nclip,bits=16)
    fid=clip.format.id
    def Graymask(src,nc):
        dif = Expr([src,nc],["x y - abs 2560 > 65535 0 ?",'','']).rgvs.RemoveGrain(4)
        mask= core.std.Inflate(dif).rgvs.RemoveGrain(11).rgvs.RemoveGrain(11)
        return mask
    if clip.format.color_family==vs.RGB:
        raise TypeError("RGB is unsupported")
    isYUV=clip.format.color_family==vs.YUV
    if not isYUV:
        mask = Graymask(clip,nclip)
        return mask
    else:
        if mode==0:
            mask=Graymask(getY(clip),getY(nclip))
            mask=core.std.ShufflePlanes(mask,[0,0,0], colorfamily=vs.YUV)
        elif mode==1:
            clip=clip.resize.Bicubic(format=vs.YUV444P16)
            nclip=nclip.resize.Bicubic(format=vs.YUV444P16)
            maskY=Graymask(getY(clip),getY(nclip))
            maskU=Graymask(getU(clip),getU(nclip))
            maskV=Graymask(getV(clip),getV(nclip))
            mask=Expr([maskY,maskU,maskV],"x y max z max")
            mask=core.std.ShufflePlanes(mask,[0,0,0], colorfamily=vs.YUV)
        else:
            raise ValueError("mode must be 0 or 1")
        return mask.resize.Bicubic(format=fid)
    
def mwlmask(clip, l1=80, h1=96, h2=None, l2=None):
    """
    luma mask
    Steal from other one's script. Most likely written by mawen1250.
    """
    sbitPS = clip.format.bits_per_sample
    black = 0
    white = (1 << sbitPS) - 1
    l1 = l1 << (sbitPS - 8)
    h1 = h1 << (sbitPS - 8)
    if h2 is None: h2 = white
    else: h2 = h2 << (sbitPS - 8)
    if l2 is None: l2 = white
    else: l2 = l2 << (sbitPS - 8)
    
    if h2 >= white:
        expr = '{white}'.format(white=white)
    else:
        expr = 'x {h2} <= {white} x {l2} < x {l2} - {slope2} * {black} ? ?'.format(black=black, white=white, h2=h2, l2=l2, slope2=white / (h2 - l2))
    expr = 'x {l1} <= {black} x {h1} < x {l1} - {slope1} * ' + expr + ' ? ?'
    expr = expr.format(black=black, l1=l1, h1=h1, slope1=white / (h1 - l1))
    
    clip = getplane(clip, 0)
    clip = clip.rgvs.RemoveGrain(4)
    clip = Expr(expr)
    return clip

def mwdbmask(clip: vs.VideoNode, chroma=True, sigma=2.5, t_h=1.0, t_l=0.5, yuv444=None, cs_h=0, cs_v=0, lmask=None, sigma2=2.5, t_h2=3.0, t_l2=1.5):
    """
    deband mask
    Steal from other one's script. Most likely written by mawen1250.
    """
    ## clip properties
    bits=clip.format.bits_per_sample
    yuv420 = clip.format.subsampling_w == 1 and clip.format.subsampling_h == 1
    sw = clip.width
    sh = clip.height
    if yuv444 is None:
        yuv444 = not yuv420
    ## Canny edge detector
    emask = clip.tcanny.TCanny(sigma=sigma, t_h=t_h, t_l=t_l, planes=[0,1,2] if chroma else [0])
    if lmask is not None:
        emask2 = clip.tcanny.TCanny(sigma2=sigma2,t_h=t_h2, t_l=t_l2, planes=[0,1,2] if chroma else [0])
        emask = core.std.MaskedMerge(emask, emask2, lmask, [0,1,2] if chroma else [0], True)
    ## apply morphologic filters and merge mask planes
    emaskY = getY(emask)
    if chroma:
        emaskC = Expr([getU(emask),getV(emask)],"x y max")
        if yuv420:
            emaskC = mt_inpand_multi(mt_expand_multi(emaskC, 'losange', sw=3, sh=3), 'rectangle', sw=3, sh=3)
            emaskC = emaskC.fmtc.resample(sw, sh, 0.25 - cs_h / 2, 0 - cs_v / 2, kernel='bilinear', fulls=True).fmtc.bitdepth(bits=bits)
        else:
            emaskY = Expr([emaskY, emaskC],"x y max")
    emaskY = mt_inpand_multi(mt_expand_multi(emaskY, 'losange', sw=5, sh=5), 'rectangle', sw=2, sh=2)
    if chroma and yuv420:
        dbmask = Expr([emaskY, emaskC],"x y max")
    else:
        dbmask = emaskY
    ## convert to final mask, all the planes of which are the same
    if yuv444:
        dbmask = mt_inflate_multi(dbmask, radius=2)
        dbmaskC = dbmask
    else:
        dbmaskC = dbmask.fmtc.resample(sw // 2, sh // 2, -0.5, 0, kernel='bilinear').fmtc.bitdepth(bits=bits)
        dbmask = mt_inflate_multi(dbmask, radius=2)
    dbmask = core.std.ShufflePlanes([dbmask, dbmaskC, dbmaskC], [0,0,0], vs.YUV)
    return dbmask


    
