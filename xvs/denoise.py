from .utils import *

def bm3d(clip:vs.VideoNode,sigma=[3,3,3],sigma2=None,preset="fast",preset2=None,mode="cpu",radius=0,radius2=None,chroma=False,fast=True,
            block_step1=None,bm_range1=None, ps_num1=None, ps_range1=None,
            block_step2=None,bm_range2=None, ps_num2=None, ps_range2=None,
            extractor_exp=0,device_id=0,bm_error_s="SSD",transform_2d_s="DCT",transform_1d_s="DCT",
            refine=1,dmode=0,
            v2=False):
    """
    warp function for bm3dcpu,bm3dcuda,bm3dcuda_rtc,and bm3dhip,similar to mvs.bm3d but only main function(without colorspace tranform)
    due to difference  between bm3d and bm3d{cpu,cuda,cuda_rtc,hip},result will not match mvf.bm3d
    -------------------------------------------------------------------------
    preset,preset2:set preset for basic estimate and final estimate.Supported value:fast,lc,np,high
    v2:True means use bm3dv2,for vbm3d,it should slightly faster
    For more info about other parameters,read bm3d{cpu,cuda,cuda_rtc,hip}'s doc.
    """
    bits=clip.format.bits_per_sample
    clip=core.fmtc.bitdepth(clip,bits=32)
    if chroma is True and clip.format.id !=vs.YUV444PS:
        raise ValueError("chroma=True only works on yuv444")
    
    if radius2 is None:
        radius2=radius

    isvbm3d=radius>0
    isvbm3d2=radius2>0

    if sigma2 is None:
        sigma2=sigma

    if preset2 is None:
        preset2=preset

    if preset not in ["fast","lc","np","high"] or preset2 not in ["fast","lc","np","high"]:
        raise ValueError("preset and preset2 must be 'fast','lc','np',or'high'")

    parmas1={
        #block_step,bm_range, ps_num, ps_range
        "fast":[8,9,2,4],
        "lc"  :[6,9,2,4],
        "np"  :[4,16,2,5],
        "high":[3,16,2,7],
    }

    vparmas1={
        #block_step,bm_range, ps_num, ps_range
        "fast":[8,7,2,4],
        "lc"  :[6,9,2,4],
        "np"  :[4,12,2,5],
        "high":[3,16,2,7],
    }

    parmas2={
        #block_step,bm_range, ps_num, ps_range
        "fast":[7,9,2,5],
        "lc"  :[5,9,2,5],
        "np"  :[3,16,2,6],
        "high":[2,16,2,8],
    }

    vparmas2={
        #block_step,bm_range, ps_num, ps_range
        "fast":[7,7,2,5],
        "lc"  :[5,9,2,5],
        "np"  :[3,12,2,6],
        "high":[2,16,2,8],
    }

    p1=vparmas1 if isvbm3d  else parmas1
    p2=vparmas2 if isvbm3d2 else parmas2
    
    block_step1=p1[preset][0] if block_step1 is None else block_step1
    bm_range1=p1[preset][1] if bm_range1 is None else bm_range1
    ps_num1=p1[preset][2] if ps_num1 is None else ps_num1
    ps_range1=p1[preset][3] if ps_range1 is None else ps_range1

    block_step2=p2[preset2][0] if block_step2 is None else block_step2
    bm_range2=p2[preset2][1] if bm_range2 is None else bm_range2
    ps_num2=p2[preset2][2] if ps_num2 is None else ps_num2
    ps_range2=p2[preset2][3] if ps_range2 is None else ps_range2

    if v2:
        flt=bm3dv2_core(clip,mode=mode,sigma=sigma,radius=radius,block_step=block_step1,bm_range=bm_range1,ps_num=ps_num1,ps_range=ps_range1,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
        
        for i in range(refine):
            flt=bm3dv2_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
    
    else:
        if isvbm3d:
            flt=bm3d_core(clip,mode=mode,sigma=sigma,radius=radius,block_step=block_step1,bm_range=bm_range1,ps_num=ps_num1,ps_range=ps_range1,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
            flt=core.bm3d.VAggregate(flt,radius=radius,sample=1)
            if isvbm3d2:
                for _ in range(refine):
                    flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
                    flt=core.bm3d.VAggregate(flt,radius=radius,sample=1)
            else:
                for _ in range(refine):
                    flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)

        else:
            flt=bm3d_core(clip,mode=mode,sigma=sigma,radius=radius,block_step=block_step1,bm_range=bm_range1,ps_num=ps_num1,ps_range=ps_range1,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
            if isvbm3d2:
                for _ in range(refine):
                    flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
                    flt=core.bm3d.VAggregate(flt,radius=radius,sample=1)
            else:    
                for _ in range(refine):
                    flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)

    return core.fmtc.bitdepth(flt,bits=bits,dmode=dmode)

def bm3d_core(clip,ref=None,mode="cpu",sigma=3.0,block_step=8,bm_range=9,radius=0,ps_num=2,ps_range=4,chroma=False,fast=True,extractor_exp=0,device_id=0,bm_error_s="SSD",transform_2d_s="DCT",transform_1d_s="DCT"):
    if mode not in ["cpu","cuda","cuda_rtc","hip"]:
        raise ValueError("mode must be cpu,or cuda,or cuda_rtc,or hip!")
    elif mode=="cpu":
        return core.bm3dcpu.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma)
    elif mode=="cuda":
        return core.bm3dcuda.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)
    elif mode=="cuda_rtc":
        return core.bm3dcuda_rtc.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
    else:
        return core.bm3dhip.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)

def bm3dv2_core(clip,ref=None,mode="cpu",sigma=3.0,block_step=8,bm_range=9,radius=0,ps_num=2,ps_range=4,chroma=False,fast=True,extractor_exp=0,device_id=0,bm_error_s="SSD",transform_2d_s="DCT",transform_1d_s="DCT"):
    if mode not in ["cpu","cuda","cuda_rtc","hip"]:
        raise ValueError("mode must be cpu,or cuda,or cuda_rtc,or hip!")
    elif mode=="cpu":
        return core.bm3dcpu.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma)
    elif mode=="cuda":
        return core.bm3dcuda.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)
    elif mode=="cuda_rtc":
        return core.bm3dcuda_rtc.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
    else:
        return core.bm3dhip.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)


@deprecated("No maintenance")
def STPresso(clip=None, limit=3, bias=24, RGmode=4, tthr=12, tlimit=3, tbias=49, back=1):
    """
    STPresso
    ####################
    orginal script by Didée
    ####################
    The goal of STPresso (Spatio-Temporal Pressdown) is
    to "dampen the grain just a little, to keep the original look,
    and make it fast". In other words it makes a video more
    compressible without losing detail and original grain structure.
    ####################
    cllp = Input clip. 
    limit = 3     Spatial limit: the spatial part won't change a pixel more than this. 
    bias = 24     The percentage of the spatial filter that will apply. 
    RGmode = 4 The spatial filter is RemoveGrain, this is its mode. 
    tthr = 12  Temporal threshold for FluxSmooth; Can be set "a good bit bigger" than usually. 
    tlimit = 3  The temporal filter won't change a pixel more than this. 
    tbias = 49  The percentage of the temporal filter that will apply. 
    back = 1  After all changes have been calculated, reduce all pixel changes by this value. (Shift "back" towards original value) 
    ####################
    STPresso is recommended for content up to 720p because
    "the spatial part might be a bit too narrow for 1080p encoding
    (since it's only a 3x3 kernel)". 
    ####################
    Differences:
    high depth support
    automatically adjust parameters to fit into different depth
    you have less choice in RGmode
    """
    depth = clip.format.bits_per_sample
    LIM1= round(limit*100.0/bias-1.0) if limit>0 else round(100.0/bias)
    LIM1 = scale(LIM1,depth)
    #(limit>0) ? round(limit*100.0/bias-1.0) :  round(100.0/bias)
    LIM2  =1 if limit<0 else limit
    LIM2 = scale(LIM2,depth)
    #(limit<0) ? 1 : limit
    BIA   = bias
    BK = scale(back,depth)
    TBIA   = bias
    TLIM1  = round(tlimit*100.0/tbias-1.0) if tlimit>0 else round(100.0/tbias)
    TLIM1 = scale(TLIM1,depth)
    #(tlimit>0) ? string( round(tlimit*100.0/tbias-1.0) ) : string( round(100.0/tbias) )
    TLIM2  = 1 if tlimit<0 else tlimit
    TLIM2 = scale(TLIM2,depth)
    #(tlimit<0) ? "1" : string(tlimit)
    bzz = core.rgvs.RemoveGrain(clip,RGmode)
    ####
    if limit < 0:
        expr  = "x y - abs "+str(LIM1)+" < x x " +str(scale(1,depth))+ " x y - x y - abs / * - ?"
        texpr  = "x y - abs "+str(TLIM1)+" < x x " +str(scale(1,depth))+ " x y - x y - abs / * - ?"
    else:
        expr  =  "x y - abs " +str(scale(1,depth))+ " < x x "+str(LIM1)+" + y < x "+str(LIM2)+" + x "+str(LIM1)+" - y > x "+str(LIM2)+" - "  + "x " +str(scale(100,depth))+" "+str(BIA)+" - * y "+str(BIA)+" * + "+str(scale(100,depth))+" / ? ? ?"
        texpr  =  "x y - abs " +str(scale(1,depth))+ " < x x "+str(TLIM1)+" + y < x "+str(TLIM2)+" + x "+str(TLIM1)+" - y > x "+str(TLIM2)+" - "  + "x " +str(scale(100,depth))+" "+str(TBIA)+" - * y "+str(TBIA)+" * + "+str(scale(100,depth))+" / ? ? ?"
    L=[]
    for i in range(0,3):
        C   = core.std.ShufflePlanes(clip, i, colorfamily=vs.GRAY)
        B = core.std.ShufflePlanes(bzz, i, colorfamily=vs.GRAY)
        O = core.std.Expr([C,B],expr)
        L.append(O)
    if tthr!=0:
        st=core.flux.SmoothT(bzz, temporal_threshold=tthr, planes=[0, 1, 2])
        diff = core.std.MakeDiff(bzz,st,[0,1,2])
        last = core.std.ShufflePlanes(L, [0,0,0], colorfamily=vs.YUV)
        diff2 = core.std.MakeDiff(last,diff,[0,1,2])
        for i in range(0,3):
            c=L[i]
            b=core.std.ShufflePlanes(diff2, i, colorfamily=vs.GRAY)
            L[i] = core.std.Expr([c,b],texpr)
    if back!=0:
        bexpr="x "+str(BK)+" + y < x "+str(BK)+" + x "+str(BK)+" - y > x "+str(BK)+" - y ? ?"
        Y = core.std.ShufflePlanes(clip, 0, colorfamily=vs.GRAY)
        L[0] = core.std.Expr([L[0],Y],bexpr)
    output=core.std.ShufflePlanes(L, [0,0,0], colorfamily=vs.YUV)
    return output

@deprecated("No maintenance")
def SPresso(clip=None, limit=2, bias=25, RGmode=4, limitC=4, biasC=50, RGmodeC=0):
    """
    SPresso
    #########################
    orginal script by Didée
    #########################
       SPresso (Spatial Pressdown) is a purely spatial script designed to achieve better
    compressibility without doing too much harm to the original detail.
       SPresso was not designed for 1080p processing/encoding; due to its 3x3 kernel
    it works better on standard definition (SD) content like DVDs and possibly on 720p.
       On noisy DVD/SD sources, compression gain usually is from 2% to 3% (light settings ->
    changes almost invisible) up to 10 to 12% (stronger settings -> slight, gentle softening, not very obvious). 
    ########################
    clip= Input clip. 
    limit = 2   Limit the maximum change for any given pixel. 
    bias = 25   Something like "aggessivity": '20' is a very light setting, '33' is already quite strong. 
    RGmode = 4  RemoveGrain mode for the luma (Y) channel. The default of "4" is the best in most cases. 
                       Mode 19 and 20 might work better in other cases; if set to 0, luma will be copied from the input clip. 
    limitC = 4    Same as limit but for chroma. 
    biasC = 50   Same as bias but for chroma. 
    RGmodeC = 0   RemoveGrain mode for the chroma channels (UV) channels; by default the chroma is simply copied from the input clip. 
                           To process chroma, set RGmodeC=4 (or 19, 20, or any other compatible mode). 
    ########################
    Differences:
    high depth support
    automatically adjust parameters to fit into different depth
    you have less choice in RGmode
    """
    depth = clip.format.bits_per_sample
    LIM1= round(limit*100.0/bias-1.0) if limit>0 else round(100.0/bias)
    LIM1 = scale(LIM1,depth)
    LIM2  =1 if limit<0 else limit
    LIM2 = scale(LIM2,depth)
    BIA = bias
    LIM1c= round(limitC*100.0/biasC-1.0) if limit>0 else round(100.0/biasC)
    LIM1c = scale(LIM1c,depth)
    LIM2c  =1 if limit<0 else limit
    LIM2c = scale(LIM2c,depth)
    BIAc = biasC
    ###
    if limit < 0:
        expr  = "x y - abs "+str(LIM1)+" < x x " +str(scale(1,depth))+ " x y - x y - abs / * - ?"
    else:
        expr  =  "x y - abs " +str(scale(0,depth))+ " <= x x "+str(LIM1)+" + y < x "+str(LIM2)+" + x "+str(LIM1)+" - y > x "+str(LIM2)+" - "  + "x " +str(scale(100,depth))+" "+str(BIA)+" - * y "+str(BIA)+" * + "+str(scale(100,depth))+" / ? ? ?"
    if limitC < 0:
        exprC  = "x y - abs "+str(LIM1c)+" < x x " +str(scale(1,depth))+ " x y - x y - abs / * - ?"
    else:
        exprC  =  "x y - abs " +str(scale(0,depth))+ " <= x x "+str(LIM1c)+" + y < x "+str(LIM2c)+" + x "+str(LIM1c)+" - y > x "+str(LIM2c)+" - "  + "x " +str(scale(100,depth))+" "+str(BIAc)+" - * y "+str(BIAc)+" * + "+str(scale(100,depth))+" / ? ? ?"
    ###
    rg = core.rgvs.RemoveGrain(clip,[RGmode,RGmodeC])
    Y = core.std.Expr([getplane(clip,0),getplane(rg,0)],expr)
    U = getplane(clip,1) if RGmodeC==0 else core.std.Expr([getplane(clip,1),getplane(rg,1)],exprC)
    V = getplane(clip,2) if RGmodeC==0 else core.std.Expr([getplane(clip,2),getplane(rg,2)],exprC)
    last = core.std.ShufflePlanes([Y,U,V],[0,0,0], colorfamily=vs.YUV)
    return last

@deprecated("No maintenance")
def STPressoMC(clip=None, limit=3, bias=24, RGmode=4, tthr=12, tlimit=3, tbias=49, back=1,s_p={},a_p={},c_p={}):
    """
    STPressoMC
    """
    depth = clip.format.bits_per_sample
    LIM1= round(limit*100.0/bias-1.0) if limit>0 else round(100.0/bias)
    LIM1 = scale(LIM1,depth)
    #(limit>0) ? round(limit*100.0/bias-1.0) :  round(100.0/bias)
    LIM2  =1 if limit<0 else limit
    LIM2 = scale(LIM2,depth)
    #(limit<0) ? 1 : limit
    BIA   = bias
    BK = scale(back,depth)
    TBIA   = bias
    TLIM1  = round(tlimit*100.0/tbias-1.0) if tlimit>0 else round(100.0/tbias)
    TLIM1 = scale(TLIM1,depth)
    #(tlimit>0) ? string( round(tlimit*100.0/tbias-1.0) ) : string( round(100.0/tbias) )
    TLIM2  = 1 if tlimit<0 else tlimit
    TLIM2 = scale(TLIM2,depth)
    #(tlimit<0) ? "1" : string(tlimit)
    bzz = core.rgvs.RemoveGrain(clip,RGmode)
    ####
    if limit < 0:
        expr  = "x y - abs "+str(LIM1)+" < x x " +str(scale(1,depth))+ " x y - x y - abs / * - ?"
        texpr  = "x y - abs "+str(TLIM1)+" < x x " +str(scale(1,depth))+ " x y - x y - abs / * - ?"
    else:
        expr  =  "x y - abs " +str(scale(1,depth))+ " < x x "+str(LIM1)+" + y < x "+str(LIM2)+" + x "+str(LIM1)+" - y > x "+str(LIM2)+" - "  + "x " +str(scale(100,depth))+" "+str(BIA)+" - * y "+str(BIA)+" * + "+str(scale(100,depth))+" / ? ? ?"
        texpr  =  "x y - abs " +str(scale(1,depth))+ " < x x "+str(TLIM1)+" + y < x "+str(TLIM2)+" + x "+str(TLIM1)+" - y > x "+str(TLIM2)+" - "  + "x " +str(scale(100,depth))+" "+str(TBIA)+" - * y "+str(TBIA)+" * + "+str(scale(100,depth))+" / ? ? ?"
    L=[]
    for i in range(0,3):
        C   = core.std.ShufflePlanes(clip, i, colorfamily=vs.GRAY)
        B = core.std.ShufflePlanes(bzz, i, colorfamily=vs.GRAY)
        O = core.std.Expr([C,B],expr)
        L.append(O)
    if tthr!=0:
        st=FluxsmoothTMC(bzz,tthr,s_p,a_p,c_p,[0,1,2])
        diff = core.std.MakeDiff(bzz,st,[0,1,2])
        last = core.std.ShufflePlanes(L, [0,0,0], colorfamily=vs.YUV)
        diff2 = core.std.MakeDiff(last,diff,[0,1,2])
        for i in range(0,3):
            c=L[i]
            b=core.std.ShufflePlanes(diff2, i, colorfamily=vs.GRAY)
            L[i] = core.std.Expr([c,b],texpr)
    if back!=0:
        bexpr="x "+str(BK)+" + y < x "+str(BK)+" + x "+str(BK)+" - y > x "+str(BK)+" - y ? ?"
        Y = core.std.ShufflePlanes(clip, 0, colorfamily=vs.GRAY)
        L[0] = core.std.Expr([L[0],Y],bexpr)
    output=core.std.ShufflePlanes(L, [0,0,0], colorfamily=vs.YUV)
    return output

@deprecated("No maintenance")
def FluxsmoothTMC(src,tthr=12,s_p={},a_p={},c_p={},planes=[0,1,2]):
    """
    port from https://forum.doom9.org/showthread.php?s=d58237a359f5b1f2ea45591cceea5133&p=1572664#post1572664
    allow setting parameters for mvtools
    """
    super_p={"pel":2,"sharp":1,}
    analyse_p={"truemotion":False,"delta":1,"blksize":16,"overlap":8,}
    s = {**super_p, **s_p}
    a = {**analyse_p, **a_p}
    sup = core.mv.Super(src,**s)
    bv = core.mv.Analyse(sup, isb=True, **a)
    fv = core.mv.Analyse(sup, isb=False, **a)
    bc = core.mv.Compensate(src,sup,bv,**c_p)
    fc = core.mv.Compensate(src,sup,fv,**c_p)
    il = core.std.Interleave([fc,src,bc])
    fs = core.flux.SmoothT(il, temporal_threshold=tthr, planes=planes)
    return core.std.SelectEvery(fs,3,1)