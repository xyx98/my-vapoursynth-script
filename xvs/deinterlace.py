from .utils import *

def FIFP(
    src: vs.VideoNode,
    mode: int = 0,
    tff: bool = True,
    mi: int = 40,
    blockx: int = 16,
    blocky: int = 16,
    cthresh: int = 8,
    chroma: bool = False,
    metric: int = 1,
    tc: bool = True,
    _pass: int = 1,
    opencl: int = False,
    device: int = -1,
) -> vs.VideoNode:
    """
    Fix Interlanced Frames in Progressive video
    ---------------------------------------
    analyze setting:
    vfm_mode: set the mode of vfm,default:5
    mi: set the mi of vfm,default:40
    cthresh: set the cthresh of vfm,default:8
    blockx,blocky:set the blockx/blocky of vfm,default:16
    ---------------------------------------
    deinterlace args:
    opencl: if True,use nnedi3cl;else use znedi3
    device: set device for nnedi3cl
    tff: true means Top field first,False means Bottom field first,default:True
    ---------------------------------------
    mode args:
    mode = 0:
           interlaced frames will be deinterlaced in the same fps
    mode = 1:
           interlaced frames will be deinterlaced in double fps,and output timecodes to create a vfr video
           need 2 pass
       _pass:
           1:analyze pass
           2:encode pass,can output timecodes
       tc:if True，will output timecodes,suggest set True only when finally encode,default:True
    ---------------------------------------
    notice:
       analyze.csv will be created when mode=1,_pass=1,you can check and revise it，then use in pass 2
    """
    clip = src if src.format.bits_per_sample==8 else core.fmtc.bitdepth(src,bits=8,dmode=8)
    order = 1 if tff else 0
    
    dect = core.tdm.IsCombed(clip,cthresh=cthresh,blockx=blockx,blocky=blocky,chroma=chroma,mi=mi,metric=metric)

    if mode==0:
        deinterlace = core.nnedi3cl.NNEDI3CL(src, field=order,device=device) if opencl else core.znedi3.nnedi3(src, field=order)
        ###
        def postprocess(n, f, clip, de):
            if f.props['_Combed'] == 1:
                return de
            else:
                return clip
        ###
        last=core.std.FrameEval(src, functools.partial(postprocess, clip=src, de=deinterlace), prop_src=dect)
        last=core.std.Cache(last, make_linear=True)
        return last
    elif mode==1:
        if _pass==1:
            t = open("analyze.csv",'w')
            t.write("frame,combed\n")
            def analyze(n, f, clip):
                t.write(str(n)+","+str(f.props['_Combed'])+"\n")
                return clip
                t.close()
            last=core.std.FrameEval(dect, functools.partial(analyze, clip=dect),prop_src=dect)
            last=core.std.Cache(last, make_linear=True)
            return last
        elif _pass==2:
            lenlst=len(src)
            num=src.fps_num
            den=src.fps_den
            c=open("analyze.csv","r")
            tmp=c.read().split("\n")[1:lenlst]
            lst=[None]*len(tmp)
            for i in tmp:
                i=i.split(",")
                lst[int(i[0])]=int(i[1])
            c.close()
            lenlst=len(lst)
            
            #tc
            if tc:
                c=open("timecodes.txt","w")
                c.write("# timecode format v2\n0\n")
                b=1000/num*den
                for i in range(lenlst):
                    if lst[i]==0:
                        c.write(str(int((i+1)*b))+"\n")
                    elif lst[i]==1:
                        c.write(str(int((i+0.5)*b))+"\n"+str(int((i+1)*b))+"\n")
                    else:
                        raise ValueError("")
                c.close()
            
            #deinterlace
            deinterlace = core.nnedi3cl.NNEDI3CL(src, field=order+2,device=device) if opencl else core.znedi3.nnedi3(src, field=order+2)
            src= core.std.Interleave([src,src])
            def postprocess(n,clip, de):
                if lst[n//2]==0:
                    return clip
                else:
                    return de
            dl=core.std.FrameEval(src, functools.partial(postprocess, clip=src, de=deinterlace))
            dl=core.std.Cache(dl, make_linear=True)
            tlist=[]
            for i in range(lenlst):
                if lst[i]==0:
                    tlist.append(2*i)
                else:
                    tlist.append(2*i)
                    tlist.append(2*i+1)
            last = core.std.SelectEvery(dl,lenlst*2,tlist)
            return last#core.std.AssumeFPS(last,src)
        else:
            ValueError("pass must be 1 or 2")
    else:
        raise ValueError("mode must be 0 or 1")

def ivtc(
    src: vs.VideoNode,
    order:int = 1,
    field:int = 2,
    mode:int = 1,
    mchroma:bool = True,
    cthresh:int = 9,
    mi:int = 80,
    vfm_chroma:bool = True,
    vfm_block: List[int] = [16,16],
    y0:int = 16,
    y1:int = 16,
    micmatch:int = 1,
    cycle:int = 5,
    vd_chroma: bool = True,
    dupthresh: float = 1.1,
    scthresh: float = 15,
    vd_block: List[int] = [32,32],
    pp: bool = True,
    nsize: int = 0,
    nns: int = 1,
    qual: int = 1,
    etype: int = 0,
    pscrn: int = 2,
    opencl: bool = False,
    device: int = -1,
) -> vs.VideoNode:
    """
    warp function for vivtc with a simple post-process use nnedi3 or user-defined filter.
    """
    src8=core.fmtc.bitdepth(src,bits=8)
    src16=core.fmtc.bitdepth(src,bits=16)

    def selector(n,f,match,di):
        if f.props["_Combed"]>0:
            return di
        else:
            return match

    match=core.vivtc.VFM(src8,order=order,field=field,mode=mode,mchroma=mchroma,cthresh=cthresh,mi=mi,chroma=vfm_chroma,blockx=vfm_block[0],blocky=vfm_block[1],y0=y0,y1=y1,scthresh=scthresh,micmatch=micmatch,clip2=src16)

    if callable(pp):
        di=pp(match)
        match=core.std.FrameEval(match,functools.partial(selector,match=match,di=di),prop_src=match)
    elif pp:
        di=nnedi3(match,field=order,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,mode="nnedi3cl" if opencl else "znedi3",device=device)
        match=core.std.FrameEval(match,functools.partial(selector,match=match,di=di),prop_src=match)

    return core.vivtc.VDecimate(match,cycle=cycle,chroma=vd_chroma,dupthresh=dupthresh,scthresh=scthresh,blockx=vd_block[0],blocky=vd_block[1])

def ivtc_t(
    src: vs.VideoNode,
    order: int = 1,
    field: int = -1,
    mode: int = 1,
    slow: int = 1,
    mchroma: bool = True,
    y0: int = 16,
    y1: int = 16,
    scthresh: float = 12.0,
    ubsco: bool = True,
    micmatching: int = 1,
    mmsco: bool = True,
    cthresh: int = 9,
    tfm_chroma: bool = True,
    tfm_block: list[int] = [16,16],
    mi: int = 80,
    metric: int = 0,
    mthresh: int = 5,
    td_mode: int = 0,
    cycleR: int = 1,
    cycle: int = 5,
    rate: float = 24000/1001 ,
    hybrid: int = 0,
    vfrdec: int = 1,
    dupThresh: float | None = None,
    vidThresh: float | None = None,
    sceneThresh: float = 15,
    vidDetect: int = 3,
    conCycle: int = None,
    conCycleTP: int = None,
    nt: int = 0,
    vd_block: list[int] = [32,32],
    tcfv1: bool = True,
    se: bool = False,
    vd_chroma: bool = True,
    noblend: bool = True,
    maxndl: int = None,
    m2PA: bool = False,
    denoise: bool = False,
    ssd: bool = False,
    sdlim: int = 0,
    pp: int = -1,
    nsize: int = 0,
    nns: int = 1,
    qual: int = 1,
    etype: int = 0,
    pscrn: int = 2,
    opencl: bool = False,
    device: int = -1,
) -> vs.VideoNode:
    """
    warp function for tivtc with a simple post-process use nnedi3 or user-defined filter.
    use pp=-1(default) to use nnedi3 or pp>0 to use tfm internal post-process.
    """

    def selector(n,f,match,di):
        if f.props["_Combed"]>0:
            return di
        else:
            return match
       
    match=core.tivtc.TFM(src,order=order,field=field,mode=mode,PP=pp if isinstance(pp,int) and pp>0 else 0,slow=slow,mChroma=mchroma,cthresh=cthresh,MI=mi,chroma=tfm_chroma,blockx=tfm_block[0],blocky=tfm_block[1],y0=y0,y1=y1,mthresh=mthresh,scthresh=scthresh,micmatching=micmatching,metric=metric,mmsco=mmsco,ubsco=ubsco)
    
    if callable(pp):
        di=pp(match)
        match=core.std.FrameEval(match,functools.partial(selector,match=match,di=di),prop_src=match)
    elif pp==-1:
        di=nnedi3(match,field=order,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,mode="nnedi3cl" if opencl else "znedi3",device=device)
        match=core.std.FrameEval(match,functools.partial(selector,match=match,di=di),prop_src=match)

    return core.tivtc.TDecimate(match,mode=td_mode,cycleR=cycleR,cycle=cycle,rate=rate,dupThresh=dupThresh,vidDetect=vidDetect,vidThresh=vidThresh,sceneThresh=sceneThresh,hybrid=hybrid,conCycle=conCycle,conCycleTP=conCycleTP,nt=nt,blockx=vd_block[0],blocky=vd_block[1],tcfv1=tcfv1,se=se,chroma=vd_chroma,noblend=noblend,maxndl=maxndl,m2PA=m2PA,denoise=denoise,ssd=ssd,sdlim=sdlim,vfrDec=vfrdec)