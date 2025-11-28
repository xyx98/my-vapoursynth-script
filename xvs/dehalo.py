from .utils import *

#from muvsfunc
def abcxyz(clp: vs.VideoNode, rad: float = 3.0, ss: float = 1.5) -> vs.VideoNode:
    """Avisynth's abcxyz()

    Reduces halo artifacts that can occur when sharpening.

    Author: Didée (http://avisynth.nl/images/Abcxyz_MT2.avsi)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rad: (float) Radius for halo removal. Default is 3.0.

        ss: (float) Radius for supersampling / ss=1.0 -> no supersampling. Range: 1.0 - ???. Default is 1.5

    """

    funcName = 'abcxyz'

    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    ox = clp.width
    oy = clp.height

    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample

    if not isGray:
        clp_src = clp
        clp = getY(clp)

    x = core.resize.Bicubic(clp, m4(ox/rad), m4(oy/rad), filter_param_a=1/3, filter_param_b=1/3).resize.Bicubic(ox, oy, filter_param_a=1, filter_param_b=0)
    y = Expr([clp, x], ['x {a} + y < x {a} + x {b} - y > x {b} - y ? ? x y - abs * x {c} x y - abs - * + {c} /'.format(
        a=scale(8, bits), b=scale(24, bits), c=scale(32, bits))])

    z1 = core.rgvs.Repair(clp, y, [1])

    if ss != 1.:
        maxbig = core.std.Maximum(y).resize.Bicubic(m4(ox*ss), m4(oy*ss), filter_param_a=1/3, filter_param_b=1/3)
        minbig = core.std.Minimum(y).resize.Bicubic(m4(ox*ss), m4(oy*ss), filter_param_a=1/3, filter_param_b=1/3)
        z2 = core.resize.Lanczos(clp, m4(ox*ss), m4(oy*ss))
        z2 = Expr([z2, maxbig, minbig], ['x y min z max']).resize.Lanczos(ox, oy)
        z1 = z2  # for simplicity

    if not isGray:
        z1 = core.std.ShufflePlanes([z1, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)

    return z1

#modified from old havsfunc
def EdgeCleaner(c: vs.VideoNode, strength: int = 10, rep: bool = True, rmode: int = 17, smode: int = 0, hot: bool = False) -> vs.VideoNode:
    '''
    EdgeCleaner v1.04
    A simple edge cleaning and weak dehaloing function.

    Parameters:
        c: Clip to process.

        strength: Specifies edge denoising strength.

        rep: Activates Repair for the aWarpSharped clip.

        rmode: Specifies the Repair mode.
            1 is very mild and good for halos,
            16 and 18 are good for edge structure preserval on strong settings but keep more halos and edge noise,
            17 is similar to 16 but keeps much less haloing, other modes are not recommended.

        smode: Specifies what method will be used for finding small particles, ie stars. 0 is disabled, 1 uses RemoveGrain.

        hot: Specifies whether removal of hot pixels should take place.
    '''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('EdgeCleaner: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('EdgeCleaner: RGB format is not supported')

    if hasattr(core,"edgemasks"):
        Prewitt=core.edgemasks.Prewitt
    else:
        from .mask import AvsPrewitt as Prewitt

    bits = c.format.bits_per_sample
    peak = (1 << bits) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = getY(c)
    else:
        c_orig = None

    if smode > 0:
        strength += 4

    main = Padding(c, 6, 6, 6, 6).warp.AWarpSharp2(blur=1, depth=cround(strength / 2)).std.Crop(6, 6, 6, 6)
    if rep:
        main = core.rgvs.Repair(main, c, mode=rmode)
    mask = Prewitt(mask)
    mask = Expr(mask,expr=[f'x {scale(4, peak)} < 0 x {scale(32, peak)} > {peak} x ? ?'])
    mask=  core.std.InvertMask(mask).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])

    final = core.std.MaskedMerge(c, main, mask)
    if hot:
        final = core.rgvs.Repair(final, c, mode=2)
    if smode > 0:
        clean = c.rgvs.RemoveGrain(mode=17)
        diff = core.std.MakeDiff(c, clean)
        mask = Prewitt(diff.std.Levels(min_in=scale(40, peak), max_in=scale(168, peak), gamma=0.35).rgvs.RemoveGrain(mode=7))
        mask=Expr(mask,f'x {scale(4, peak)} < 0 x {scale(16, peak)} > {peak} x ? ?')
        final = core.std.MaskedMerge(final, c, mask)

    if c_orig is not None:
        final = core.std.ShufflePlanes([final, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return final

@deprecated("No maintenance.")
def LazyDering(src,depth=32,diff=8,thr=32):
    """
    LazyDering
    -----------------------------
    port from avs script by Leak&RazorbladeByte
    LazyDering tries to clean up slight ringing around edges by applying aWarpSharp2 only to areas where the difference is small enough so detail isn't destroyed.
    LazyDering it's a modified version of aWarpSharpDering.
    """
    bit = src.format.bits_per_sample
    Y = getplane(src,0)
    sharped = core.warp.AWarpSharp2(Y,depth=depth)
    diff_mask =  Expr([Y,sharped], "x y - x y > *").std.Levels(0,scale(diff,bit),0.5,scale(255,bit),0)
    luma_mask = core.std.Deflate(Y).std.Levels(scale(16,bit),scale(16+thr,bit),0.5,0,scale(255,bit))
    masks = Expr([luma_mask,diff_mask], "x y * {} /".format(scale(255,bit))).std.Deflate()
    merge = core.std.MaskedMerge(Y,sharped, mask=masks)
    return core.std.ShufflePlanes([merge,src],[0,1,2], colorfamily=vs.YUV)

@deprecated("No maintenance.")
def SADering(src,ring_r=2,warp_arg={},warpclip=None,edge_r=2,show_mode=0):
    """
    Simple Awarpsharp2 Dering
    ---------------------------------------
    ring_r: (int)range of ring,higher means more area around edge be set as ring in ringmask,deflaut is 2.Sugest use the smallest value which can dering well
    warp_arg: (dict)set your own args for AWarpSharp2,should be dict. e.g.
    warpclip: (clip)aim to allow you input a own warped clip instead internal warped clip,but I think a rightly blurred clip may also useful
    edge_r: (int)if the non-ring area between nearly edge can't be preserved well,try increase it's value
    show_mode:  0 :output the result 1: edgemask 2: ringmask 3: warped clip
    """
    arg={'depth':16,'type':0}
    w_a = {**arg,**warp_arg}
    isGray = src.format.color_family == vs.GRAY
    clip = src if isGray else getplane(src,0)
    ####warp
    warp = core.warp.AWarpSharp2(clip,**w_a) if warpclip is None else warpclip
    ####mask
    edgemask = core.tcanny.TCanny(clip,mode=0, op=1)
    edgemask = expand(edgemask,cycle=edge_r)
    edgemask = core.std.Deflate(edgemask)
    edgemask = inpand(edgemask,cycle=edge_r)
    #
    mask = expand(edgemask,cycle=ring_r+1)
    mask = inpand(mask,cycle=1)
    #
    ringmask = Expr([edgemask,mask], ["y x -"])
    ####
    merge = core.std.MaskedMerge(clip, warp, ringmask)
    last = merge if isGray else core.std.ShufflePlanes([merge,src],[0,1,2], colorfamily=vs.YUV)
    ####
    if show_mode==0:
        return last
    elif show_mode==1:
        return edgemask
    elif show_mode==2:
        return ringmask
    elif show_mode==3:
        return warp
    else:
        raise vs.Error("show_mode should in [0,1,2,3]")
    
