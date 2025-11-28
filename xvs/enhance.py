from .utils import *
from .metrics import getsharpness
from .dehalo import EdgeCleaner

@deprecated("No maintenance.")
def xsUSM(src=None,blur=11,limit=1,elast=4,maskclip=None,plane=[0]):
    """
    xsUSM: xyx98's simple unsharp mask
    -----------------------------------------------
    blur:   way to get blurclip,default is 11,means RemoveGrain(mode=11)
               you can also use a list like [1,2,1,2,4,2,1,2,1],means Convolution(matrix=[1,2,1,2,4,2,1,2,1])
               or you can input a blur clip made by yourself
    limit:  way to limit the sharp resault,default is 3
               =0 no limit
               >0 use limitfilter thr=limit
               <0 use repair(mode=-limit)
    elast: elast in LimitFilter,only for limit>0
    maskclip: you can input your own mask to merge resault and source if needed,default is None,means skip this step
    plane: setting which plane or planes to be processed,default is [0],means only process Y
    """
    isGray = src.format.color_family == vs.GRAY
    def usm(clip=None,blur=11,limit=1,elast=4,maskclip=None):
        if isinstance(blur,int):
            blurclip=core.rgvs.RemoveGrain(clip,blur)
        elif isinstance(blur,list):
            blurclip=core.std.Convolution(clip,matrix=blur,planes=0)
        else:
            blurclip=blur
        diff=core.std.MakeDiff(clip,blurclip,planes=0)
        sharp = core.std.MergeDiff(clip,diff,planes=0)
        ###
        if limit==0:
            lt=sharp
        elif limit>0:
            lt = LimitFilter(sharp,clip,thr=limit,elast=elast)
        elif limit<0:
            lt = core.rgvs.Repair(sharp,clip,-limit)
        if isinstance(maskclip,vs.VideoNode):
            m =  core.std.MaskedMerge(lt,clip,maskclip,planes=0)
        else:
            m = lt
        return m
    if isGray:
        return usm(src,blur,limit,elast,maskclip)
    else:
        li=[]
        for i in range(3):
            if i in plane:
                a=usm(getplane(src,i),blur,limit,elast,None if maskclip is None else getplane(maskclip,i))
            else:
                a=getplane(src,i)
            li.append(a)
        return core.std.ShufflePlanes(li,[0,0,0], vs.YUV)

@deprecated("No maintenance.")    
def SharpenDetail(src=None,limit=4,thr=32):
    """
    SharpenDetail
    ------------------------------------
    ideas comes from : https://forum.doom9.org/showthread.php?t=163598
    but it's not a port,because their is no asharp in vapoursynth,so I adjust sharpener
    """
    cFormat=src.format.color_family
    depth=src.format.bits_per_sample
    if cFormat==vs.YUV:
        clip=getplane(src,0)
    elif cFormat==vs.GRAY:
        clip=src
    else:
        raise TypeError("")
    thr = thr*depth/8
    bia = 128*depth/8
    blur = core.rgvs.RemoveGrain(clip,19)
    mask = core.std.Expr([clip,blur],"x y - "+str(thr)+" * "+str(bia)+" +")
    mask = core.rgvs.RemoveGrain(mask,2)
    mask = inpand(mask,mode="both")
    mask = core.std.Deflate(mask)
    sharp = xsUSM(clip,blur=[1]*25,limit=limit,maskclip=None)
    last = core.std.MaskedMerge(sharp,clip,mask,planes=0)
    if cFormat==vs.YUV:
        last = core.std.ShufflePlanes([last,src],[0,1,2], vs.YUV)
    return last

#modify from havsfunc
def FastLineDarkenMOD(c, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0):
    """
    ##############################
    # FastLineDarken 1.4x MT MOD #
    ##############################
    Written by Vectrangle    (http://forum.doom9.org/showthread.php?t=82125)
    Didée: - Speed Boost, Updated: 11th May 2007
    Dogway - added protection option. 12-May-2011

    Parameters are:
    strength (integer)   - Line darkening amount, 0-256. Default 48. Represents the _maximum_ amount
                          that the luma will be reduced by, weaker lines will be reduced by
                          proportionately less.
    protection (integer) - Prevents the darkest lines from being darkened. Protection acts as a threshold.
                          Values range from 0 (no prot) to ~50 (protect everything)
    luma_cap (integer)   - value from 0 (black) to 255 (white), used to stop the darkening
                      determination from being 'blinded' by bright pixels, and to stop grey
                        lines on white backgrounds being darkened. Any pixels brighter than
                          luma_cap are treated as only being as bright as luma_cap. Lowering
                          luma_cap tends to reduce line darkening. 255 disables capping. Default 191.
    threshold (integer)  - any pixels that were going to be darkened by an amount less than
                          threshold will not be touched. setting this to 0 will disable it, setting
                          it to 4 (default) is recommended, since often a lot of random pixels are
                          marked for very slight darkening and a threshold of about 4 should fix
                          them. Note if you set threshold too high, some lines will not be darkened
    thinning (integer)   - optional line thinning amount, 0-256. Setting this to 0 will disable it,
                      which is gives a _big_ speed increase. Note that thinning the lines will
                      inherently darken the remaining pixels in each line a little. Default 0.
    """
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FastLineDarkenMOD: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FastLineDarkenMOD: RGB format is not supported')

    peak = (1 << c.format.bits_per_sample) - 1 if c.format.sample_type == vs.INTEGER else 1.0

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = getplane(c, 0)
    else:
        c_orig = None

    ## parameters ##
    Str = strength / 128
    lum = scale(luma_cap, peak)
    thr = scale(threshold, peak)
    thn = thinning / 16

    ## filtering ##
    exin = c.std.Maximum(threshold=peak / (protection + 1)).std.Minimum()
    thick = Expr([c, exin], expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {Str} * x +'])
    if thinning <= 0:
        last = thick
    else:
        diff = Expr([c, exin], expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {scale(127, peak)} +'])
        linemask = Expr(diff.std.Minimum(),expr=[f'x {scale(127, peak)} - {thn} * {peak} +']).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        thin = Expr([c.std.Maximum(), diff], expr=[f'x y {scale(127, peak)} - {Str} 1 + * +'])
        last = core.std.MaskedMerge(thin, thick, linemask)

    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last


def mwenhance(diffClip, chroma=False, Strength=2.0, Szrp8=8, Spwr=4, SdmpLo=4, SdmpHi=48, Soft=0, useExpr=False):
    """
    high frequency enhance
    Steal from other one's script. Most likely written by mawen1250.
    use it on your high frequency layer.
    
    add useExpr for using Expr instead of lut

    """
    # constant values for sharpening LUT
    if Strength<=0:
        return diffClip

    sbitPS = diffClip.format.bits_per_sample
    bpsMul8 = 1 << (sbitPS - 8)
    floor = 0
    ceil = (1 << sbitPS) - 1
    neutral = 1 << (sbitPS - 1)
    miSpwr = 1 / Spwr
    Szrp = Szrp8 * bpsMul8
    Szrp8Sqr = Szrp8 * Szrp8
    SzrpMulStrength = Szrp * Strength
    Szrp8SqrPlusSdmpLo = Szrp8Sqr + SdmpLo
    SdmpHiEqual0 = SdmpHi == 0
    Szrp8DivSdmpHiPower4Plus1 = 1 if SdmpHiEqual0 else (Szrp8 / SdmpHi) ** 4 + 1

    if useExpr:
        #generate expr
        diff=' x {neutral} - '.format(neutral=neutral)
        absDiff=' {diff} abs '.format(diff=diff)
        diff8=' {diff} {bpsMul8} / '.format(diff=diff,bpsMul8=bpsMul8)
        absDiff8=' {diff8} abs '.format(diff8=diff8)
        diff8Sqr = ' {diff8} {diff8} * '.format(diff8=diff8)
        signMul=' {diff} 0 >= 1 -1 ? '.format(diff=diff)

        res1=' {absDiff} {Szrp} / {miSpwr} pow {SzrpMulStrength} * {signMul} * '.format(absDiff=absDiff,Szrp=Szrp,miSpwr=miSpwr,SzrpMulStrength=SzrpMulStrength,signMul=signMul)
        res2=' {diff8Sqr} {Szrp8SqrPlusSdmpLo} * {diff8Sqr} {SdmpLo} + {Szrp8Sqr} * / '.format(diff8Sqr=diff8Sqr,Szrp8SqrPlusSdmpLo=Szrp8SqrPlusSdmpLo,SdmpLo=SdmpLo,Szrp8Sqr=Szrp8Sqr)
        res3=' 0 ' if SdmpHiEqual0 else ' {absDiff8} {SdmpHi} / 4 pow '.format(absDiff8=absDiff8,SdmpHi=SdmpHi)

        enhanced=' {res1} {res2} * {Szrp8DivSdmpHiPower4Plus1} * 1 {res3} + /'.format(res1=res1,res2=res2,res3=res3,Szrp8DivSdmpHiPower4Plus1=Szrp8DivSdmpHiPower4Plus1)
        enhanced=' {ceil} {floor} {neutral} {enhanced} + max min '.format(ceil=ceil,floor=floor,neutral=neutral,enhanced=enhanced)

        expr=' x {neutral} = x {enhanced} ? '.format(neutral=neutral,enhanced=enhanced)

        #apply expr
        if diffClip.format.num_planes==1:
            diffClip=Expr(diffClip,[expr])
        else:
            diffClip=Expr(diffClip,[expr,expr if chroma else ''])

    else:
        # function to generate sharpening LUT
        def diffEhFunc(x):
            if x == neutral:
                return x

            diff = x - neutral
            absDiff = abs(diff)
            diff8 = diff / bpsMul8
            absDiff8 = abs(diff8)
            diff8Sqr = diff8 * diff8
            signMul = 1 if diff >= 0 else -1

            res1 = (absDiff / Szrp) ** miSpwr * SzrpMulStrength * signMul
            res2 = diff8Sqr * Szrp8SqrPlusSdmpLo / ((diff8Sqr + SdmpLo) * Szrp8Sqr)
            res3 = 0 if SdmpHiEqual0 else (absDiff8 / SdmpHi) ** 4
            enhanced = res1 * res2 * Szrp8DivSdmpHiPower4Plus1 / (1 + res3)

            return min(ceil, max(floor, round(neutral + enhanced)))
        # apply sharpening LUT
        diffClip = diffClip.std.Lut([0,1,2] if chroma else [0], function=diffEhFunc)

    #soften the result
    if Soft > 0:
        diffClipEhSoft = diffClip.rgvs.RemoveGrain([19, 19 if chroma else 0])
        diffClipEhSoft = diffClipEhSoft if Soft >= 1 else core.std.Merge(diffClip, diffClipEhSoft, [1 - Soft, Soft])
        limitDiffExpr=' x {neutral} - abs y {neutral} - abs <= x y ? '.format(neutral=neutral)
        diffClip = Expr([diffClip, diffClipEhSoft], [limitDiffExpr, limitDiffExpr if chroma else ''])
    # output
    return diffClip

def mwcfix(clip, kernel=1, restore=5, a=2, grad=2, warp=6, thresh=96, blur=3, repair=1, cs_h=0, cs_v=0,nlm_mode="nlm_ispc",nlm_device_type="auto",nlm_device_id=0,nnedi3_mode="znedi3",nnedi3_device=-1):
    """
    chroma restoration
    Steal from other one's script. Most likely written by mawen1250.
    repalce nnedi3 with znedi3
    """
    if clip.format.bits_per_sample != 16:
        raise vs.Error('mwcfix: Only 16bit supported')

    clip_y,clip_u,clip_v = extractPlanes()

    cssw = clip.format.subsampling_w
    cssh = clip.format.subsampling_h

    if cs_h != 0 or cssw > 0:
        if cssw > 0 and kernel == 1:
            clip_u = core.fmtc.bitdepth(clip_u,bits=8)
            clip_v = core.fmtc.bitdepth(clip_v,bits=8)
        clip_u = clip_u.std.Transpose()
        clip_v = clip_v.std.Transpose()
        field = 1
        for i in range(cssw):
            if kernel >= 2:
                clip_u = clip_u.eedi2.EEDI2(field=field)
                clip_v = clip_v.eedi2.EEDI2(field=field)
            elif kernel == 1:
                clip_u = nnedi3(clip_u,field=field,dh=True,mode=nnedi3_mode,device=nnedi3_device)
                clip_v = nnedi3(clip_v,field=field,dh=True,mode=nnedi3_mode,device=nnedi3_device)
        sy = -cs_h
        clip_u = clip_u.fmtc.resample(h=clip_y.width, sy=sy, center=False, kernel="bicubic", a1=0, a2=0.5)
        clip_v = clip_v.fmtc.resample(h=clip_y.width, sy=sy, center=False, kernel="bicubic", a1=0, a2=0.5)
        clip_u = clip_u.std.Transpose()
        clip_v = clip_v.std.Transpose()

    if cs_v != 0 or cssh > 0:
        if cssh > 0 and kernel == 1:
            clip_u = core.fmtc.bitdepth(clip_u,bits=8)
            clip_v = core.fmtc.bitdepth(clip_v,bits=8)
        field = 1
        for i in range(clip.format.subsampling_w):
            if kernel >= 2:
                clip_u = clip_u.eedi2.EEDI2(field=field)
                clip_v = clip_v.eedi2.EEDI2(field=field)
            elif kernel == 1:
                clip_u = nnedi3(clip_u,field=field,dh=True,mode=nnedi3_mode,device=nnedi3_device)
                clip_v = nnedi3(clip_v,field=field,dh=True,mode=nnedi3_mode,device=nnedi3_device)
            field = 0
        sy = (-0.5 if cssh > 0 and kernel > 0 else 0) - cs_v
        clip_u = clip_u.fmtc.resample(h=clip_y.height, sy=sy, center=True, kernel="bicubic", a1=0, a2=0.5)
        clip_v = clip_v.fmtc.resample(h=clip_y.height, sy=sy, center=True, kernel="bicubic", a1=0, a2=0.5)

    pp_u = clip_u
    pp_v = clip_v

    if restore > 0:
        rst_u = nlm(pp_u,d=0, a=a, s=1, h=restore, rclip=clip_y, device_type=nlm_device_type, device_id=nlm_device_id,mode=nlm_mode).rgvs.Repair(pp_u, 13)
        rst_v = nlm(pp_v,d=0, a=a, s=1, h=restore, rclip=clip_y, device_type=nlm_device_type, device_id=nlm_device_id,mode=nlm_mode).rgvs.Repair(pp_v, 13)
        low_u = rst_u
        low_v = rst_v
        for i in range(grad):
            low_u = low_u.rgvs.RemoveGrain(20)
            low_v = low_v.rgvs.RemoveGrain(20)
        pp_u = Expr([pp_u, rst_u, low_u], 'y z - x +')
        pp_v = Expr([pp_v, rst_v, low_v], 'y z - x +')

    if warp > 0:
        awarp_mask = core.fmtc.bitdepth(clip_y,bits=8).warp.ASobel(thresh).warp.ABlur(blur, 1)
        pp_u8 = core.fmtc.bitdepth(core.fmtc.bitdepth(pp_u,bits=8).warp.AWarp(awarp_mask, warp),bits=16)
        pp_v8 = core.fmtc.bitdepth(core.fmtc.bitdepth(pp_v, bits=8).warp.AWarp(awarp_mask, warp), bits=16)
        pp_u = LimitFilter(pp_u, pp_u8, thr=1.0, elast=2.0)
        pp_v = LimitFilter(pp_v, pp_v8, thr=1.0, elast=2.0)

    if repair > 0:
        pp_u = pp_u.rgvs.Repair(clip_u, repair)
        pp_v = pp_v.rgvs.Repair(clip_v, repair)
    elif repair < 0:
        pp_u = clip_u.rgvs.Repair(pp_u, -repair)
        pp_v = clip_v.rgvs.Repair(pp_v, -repair)

    final = core.std.ShufflePlanes([clip_y, pp_u, pp_v], [0,0,0], vs.YUV)
    final = final.fmtc.resample(csp=clip.format.id, kernel="bicubic", a1=0, a2=0.5)
    return final

@deprecated("Strange idea,not really useful.")
def ssharp(clip,chroma=True,mask=False,compare=False):
    """
    slightly sharp through bicubic
    """
    isGRAY=clip.format.color_family==vs.GRAY
    src=clip.fmtc.bitdepth(bits=16)
    w=src.width
    h=src.height
    if chroma and not isGRAY:
        sha = core.fmtc.resample(src,w*2,h*2,kernel='bicubic',a1=-1,a2=6).resize.Lanczos(w,h)
        last=core.rgvs.Repair(sha, src, 13)
        last=LimitFilter(src, last, thr=1, thrc=0.5, elast=6, brighten_thr=0.5, planes=[0,1,2])
        if mask:
            mask1=src.tcanny.TCanny(sigma=0.5, t_h=20.0, t_l=8.0,mode=1).std.Expr("x 30000 < 0 x ?").rgvs.RemoveGrain(4)
            mask1=inpand(expand(mask1,cycle=1),cycle=1)
            mask2=core.std.Expr([last,src],"x y - abs 96 *").rgvs.RemoveGrain(4)
            mask2=core.std.Expr(mask2,"x 30000 < 0 x ?")
            mask=core.std.Expr([mask1,mask2],"x y min")
            last=core.std.MaskedMerge(last, src, mask,[0,1,2])
    elif not chroma:
        srcy=getY(src)
        sha = core.fmtc.resample(srcy,w*2,h*2,kernel='bicubic',a1=-1,a2=6).resize.Lanczos(w, h)
        last=core.rgvs.Repair(sha, srcy, 13)
        last=LimitFilter(srcy, last, thr=1,elast=6, brighten_thr=0.5, planes=0)
        if mask:
            mask1=srcy.tcanny.TCanny(sigma=0.5, t_h=20.0, t_l=8.0,mode=1).std.Expr("x 30000 < 0 x ?").rgvs.RemoveGrain(4)
            mask1=inpand(expand(mask1,cycle=1),cycle=1)
            mask2=core.std.Expr([last,srcy],"x y - abs 96 *").rgvs.RemoveGrain(4)
            mask2=core.std.Expr(mask2,"x 30000 < 0 x ?")
            mask=core.std.Expr([mask1,mask2],"x y min")
            last=core.std.MaskedMerge(last, srcy, mask,0)
        last=core.std.ShufflePlanes([last,src], [0,1,2],colorfamily=vs.YUV)
    elif isGRAY:
        sha = core.fmtc.resample(src,w*2,h*2,kernel='bicubic',a1=-1,a2=6).resize.Lanczos(w,h)
        last=core.rgvs.Repair(sha, src, 13)
        last=LimitFilter(src, last, thr=1, thrc=0.5, elast=6, brighten_thr=0.5)
        if mask:
            mask1=src.tcanny.TCanny(sigma=0.5, t_h=20.0, t_l=8.0,mode=1).std.Expr("x 30000 < 0 x ?").rgvs.RemoveGrain(4)
            mask1=inpand(expand(mask1,cycle=1),cycle=1)
            mask2=core.std.Expr([last,src],"x y - abs 96 *").rgvs.RemoveGrain(4)
            mask2=core.std.Expr(mask2,"x 30000 < 0 x ?")
            mask=core.std.Expr([mask1,mask2],"x y min")
            last=core.std.MaskedMerge(last, src, mask)
    if not compare:
        return last
    else:
        return core.std.Interleave([src.text.Text("src"),last.text.Text("sharp")])
    
def SCSharpen(clip:vs.VideoNode,ref:vs.VideoNode,max_sharpen_weight=0.3,min_sharpen_weight=0,clean=True):
    """
    Sharpness Considered Sharpen:
    It mainly design for sharpen a bad source after blurry filtered such as strong AA, and source unsuited to be reference when you want sharpen filtered clip to match the sharpness of source.
    It use cas as sharpen core,and calculate sharpness of source(reference clip),filtered clip (the clip you want sharpen),and sharpen clip(by cas).Use these sharpness information adjust merge weight of filtered clip and sharpen clip.
    ############################
    If clean is True,use EdgeCleaner clean edge after sharpen.
    Don't use high max_sharpen_weight or you might need addition filter to resolve artifacts cause by cas(1).
    only luma processed,output is always 16bit.
    """
    if max_sharpen_weight >1 or max_sharpen_weight <=0 :
        raise ValueError("max_sharpen_weight should in (0,1]")
    
    if min_sharpen_weight >1 or min_sharpen_weight <0  or max_sharpen_weight<min_sharpen_weight:
        raise ValueError("min_sharpen_weight should in [0,1] and less than max_sharpen_weight")

    ref,clip=core.fmtc.bitdepth(ref,bits=16),core.fmtc.bitdepth(clip,bits=16)
    if clip.format.color_family == vs.YUV:
        isYUV=True
        last=getY(clip)
    elif clip.format.color_family == vs.GRAY:
        last=clip
        isYUV=False
    else:
        raise vs.ValueError("clip must be YUV or GRAY")

    if ref.format.color_family == vs.YUV:
        ref=getY(ref)
    elif ref.format.color_family == vs.GRAY:
        pass
    else:
        raise vs.ValueError("ref must be YUV or GRAY")

    sharp=core.cas.CAS(last,1,0)
    ref,last,sharp=map(getsharpness,[ref,last,sharp])
    #########################
    base=" z.sharpness y.sharpness - "
    k1=f"x.sharpness y.sharpness - {base} /"
    k2=f"z.sharpness x.sharpness - {base} /"
    L1=max_sharpen_weight
    L2=1-L1
    L3=min_sharpen_weight
    L4=1-L3

    expr=f"{base} 0 = {L1} z * {L2} y * + {k1} {L1} > {L1} z * {L2} y * + {k1} {L3} < {L3} z * {L4} y * + {k1} z * {k2} y * + ? ? ?"
    last=core.akarin.Expr([ref,last,sharp],expr)
    
    if clean:
        last=EdgeCleaner(last,strength=10, rmode=17, smode=0, hot=False)
    if isYUV:
        last=core.std.ShufflePlanes([last,clip],[0,1,2],vs.YUV)

    return last

