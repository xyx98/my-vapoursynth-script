from .utils import *

def rescale(src:vs.VideoNode,kernel:str,w:int=None,h:int=None,mask:Union[bool,vs.VideoNode]=True,mask_dif_pix:float=2,show: str="result",postfilter_descaled=None,mthr:list[int]=[2,2],maskpp=None,**args):
    """
    descale to target resolution and upscale by nnedi3_resample with optional mask and postfilter.
    -----------------------------------------------------------
    kernel:must in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]
    w,h:target resolution for descale
    mask:True to enable internal mask,or input your clip as mask.
    mask_dif_pix: set protection strength.This value means the threshold of difference to consider as native high resolution,calculate under 8bit.
    mthr:set [expand,inpand] times for mask
    maskpp:use self-defined function to replace internal expand and inpand process for mask.
    postfilter_descaled:self-defined postfilter for descaled clip.
    #################################
    b,c,taps:parameters for kernel
    nsize,nns,qual,etype,pscrn,exp,sigmoid:parameters for nnedi3_resample
    """
    if src.format.color_family not in [vs.YUV,vs.GRAY]:
        raise ValueError("input clip should be YUV or GRAY!")

    src_h,src_w=src.height,src.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")
    
    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")
    
    src=core.fmtc.bitdepth(src,bits=16)
    luma=getY(src)

    taps=args.get("taps")
    b,c=args.get("b"),args.get("c")
    nsize=3 if args.get("nsize") is None else args.get("nsize")#keep behavior before
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")#keep behavior before
    etype=args.get("etype")
    pscrn=args.get("pscrn")
    exp=args.get("exp")
    sigmoid=args.get("sigmoid")

    luma_rescale,mask,luma_de=MRcore(luma,kernel[2:],w,h,mask=mask,mask_dif_pix=mask_dif_pix,postfilter_descaled=postfilter_descaled,mthr=mthr,taps=taps,b=b,c=c,multiple=1,maskpp=maskpp,show="both",nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,sigmoid=sigmoid)
    
    if show=="descale":
        return luma_de
    elif show=="mask":
        return mask
    elif show=="both":
        return luma_de,mask

    if src.format.color_family==vs.GRAY:
        return luma_rescale
    else:
        return core.std.ShufflePlanes([luma_rescale,src],[0,1,2],vs.YUV)

def rescalef(src: vs.VideoNode,kernel: str,w=None,h=None,bh=None,bw=None,mask=True,mask_dif_pix=2,show="result",postfilter_descaled=None,mthr:list[int]=[2,2],maskpp=None,selective=False,upper=0.0001,lower=0.00001,**args):
    #for decimal resolution descale,refer to GetFnative
    if src.format.color_family not in [vs.YUV,vs.GRAY]:
        raise ValueError("input clip should be YUV or GRAY!")

    src_h,src_w=src.height,src.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")

    if bh is None:
        bh=src_h

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")
    
    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")

    src=core.fmtc.bitdepth(src,bits=16)
    luma=getY(src)

    taps=args.get("taps")
    b,c=args.get("b"),args.get("c")
    nsize=3 if args.get("nsize") is None else args.get("nsize")#keep behavior before
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")#keep behavior before
    etype=args.get("etype")
    pscrn=args.get("pscrn")
    exp=args.get("exp")
    sigmoid=args.get("sigmoid")

    luma_rescale,mask,luma_de=MRcoref(luma,kernel[2:],w,h,bh,bw,mask=mask,mask_dif_pix=mask_dif_pix,postfilter_descaled=postfilter_descaled,mthr=mthr,taps=taps,b=b,c=c,multiple=1,maskpp=maskpp,show="both",nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,sigmoid=sigmoid)

    if selective:
        base=upper-lower
        #x:rescale y:src
        expr=f"x.diff {upper} > y x.diff {lower} < x {upper} x.diff -  {base} / y * x.diff {lower} - {base} / x * + ? ?"
        luma_rescale=core.akarin.Expr([luma_rescale,luma], expr)

    if show=="descale":
        return luma_de
    elif show=="mask":
        return mask
    elif show=="both":
        return luma_de,mask

    if src.format.color_family==vs.GRAY:
        return luma_rescale
    else:
        return core.std.ShufflePlanes([luma_rescale,src],[0,1,2],vs.YUV)

def multirescale(clip:vs.VideoNode,kernels:list[dict],w:Optional[int]=None,h:Optional[int]=None,mask:bool=True,mask_dif_pix:float=2.5,postfilter_descaled=None,mthr:list[int]=[2,2],maskpp=None,selective_disable:bool=False,disable_thr:float=0.00001,showinfo=False,save=None,load=None,kindex=False,**args):
    clip=core.fmtc.bitdepth(clip,bits=16)
    luma=getY(clip)
    src_h,src_w=clip.height,clip.width
    def getwh(w,h):
        if w is None and h is None:
            w,h=1280,720
        elif w is None:
            w=int(h*src_w/src_h)
        elif h is None:
            h=int(w*src_h/src_w)

        if w>=src_w or h>=src_h:
            raise ValueError("w,h should less than input resolution")
        return w,h

    w,h=getwh(w,h)

    info_gobal=f"gobal:\nresolution:{w}x{h}\tmask:{mask}\tmask_dif_pix:{mask_dif_pix}\tpostfilter_descaled:{'yes' if callable(postfilter_descaled) else 'no'}\nselective_disable:{selective_disable}\tdisable_thr:{disable_thr:f}\nmthr:{str(mthr)}\tmaskpp:{'yes' if callable(maskpp) else 'no'}\nextra:{str(args)}"
    rescales=[]
    total=len(kernels)
    for i in kernels:
        fmode=False if i.get("fmode") is None else i.get("fmode")
        k=i["k"][2:]
        kb,kc,ktaps=i.get("b"),i.get("c"),i.get("taps")
        kw,kh=i.get("w"),i.get("h")
        if kw is not None or kh is not None:
            kw,kh=getwh(kw,kh)
        else:
            kw,kh=w,h
        kmask=mask if i.get("mask") is None else i.get("mask")
        kmdp=mask_dif_pix if i.get("mask_dif_pix") is None else i.get("mask_dif_pix")
        kmthr=mthr if i.get("mthr") is None else i.get("mthr")
        kpp=postfilter_descaled if i.get("postfilter_descaled") is None else i.get("postfilter_descaled")
        multiple=1 if i.get("multiple") is None else i.get("multiple")
        kmaskpp=maskpp if i.get("maskpp") is None else i.get("maskpp")


        if not fmode:
            rescales.append(MRcore(luma,kernel=k,w=kw,h=kh,mask=kmask,mask_dif_pix=kmdp,postfilter_descaled=kpp,mthr=kmthr,taps=ktaps,b=kb,c=kc,multiple=multiple,maskpp=kmaskpp,**args))
        else:
            kbh=src_h if i.get("bh") is None else i.get("bh")
            kbw=i.get("bw")
            rescales.append(MRcoref(luma,kernel=k,w=kw,h=kh,bh=kbh,bw=kbw,mask=kmask,mask_dif_pix=kmdp,postfilter_descaled=kpp,mthr=kmthr,taps=ktaps,b=kb,c=kc,multiple=multiple,maskpp=kmaskpp,**args))

    if load is not None:
        with open(load,"r",encoding="utf-8") as file:
            plist=file.read().split('\n')

        slist={}
        for line in plist[1:]:
            tmp=line.split("\t")
            if len(tmp)>=2:
                slist[int(tmp[0])]={'select':int(tmp[1]),'diff':[float(i) for i in tmp[2:]]}

    if save is not None and save!=load:
        saves=open(save,"w",encoding="utf-8")
        saves.write("n\tselect\t"+"\t".join([str(i) for i in range(len(kernels))])+"\n")

    def selector(n,f,src,clips):
        kernels_info=[]
        diffs=[]
        if len(f)==1:
            f=[f]
        index,mindiff=0,f[0].props["diff"]
        for i in range(total):
            tmpdiff=f[i].props["diff"]
            diffs.append(tmpdiff)
            kernels_info.append(f"kernel {i}:\t{kernels[i]}\n{tmpdiff:.10f}")
            if tmpdiff<mindiff:
                index,mindiff=i,tmpdiff

        info=info_gobal+"\n--------------------\n"+("\n--------------------\n").join(kernels_info)+"\n--------------------\ncurrent usage:\n"
        info_short="\t".join([str(i) for i in diffs])
        if selective_disable and mindiff>disable_thr:
            usesrc=True
            last=src
            info+="source"
            info_short=f"{n}\t-1\t"+info_short+"\n"
            index=-1
        else:
            last=clips[index]
            info+=kernels_info[index]
            info_short=f"{n}\t{index}\t"+info_short+"\n"
        
        if load is not None:
            info+="--------------------\noverwrite info:\n"
            if slist.get(n) is not None:
                newindex=slist[n]["select"]
                if newindex==-1:
                    if usesrc:
                        info+="same as current"
                    else:
                        last=src
                        info+="source"
                else:
                    if newindex==index:
                        info+="same as current"
                    else:
                        last=clips[newindex]
                        info+=kernels_info[newindex]
            else:
                newindex=-2
                info+="skip"

        if showinfo:
            last=core.text.Text(last,info.replace("\t","    "))
        if save is not None:
            saves.write(info_short)
        if kindex:
            last=core.std.SetFrameProp(last,'kindex',intval=index)
            if load:
                last=core.std.SetFrameProp(last,'nkindex',intval=newindex)
        return last
        saves.close()

    last=core.std.FrameEval(luma,functools.partial(selector,src=luma,clips=rescales),prop_src=rescales)
    if clip.format.color_family==vs.GRAY:
        return last
    else:
        return core.std.ShufflePlanes([last,clip],[0,1,2],vs.YUV)

def resize_core(kernel:str,taps: int=3,b: float=0,c: float=0):
    kernel=kernel.capitalize()
    if kernel in ["Bilinear","Spline16","Spline36","Spline64"]:
        return eval(f"core.resize.{kernel}")
    elif kernel == "Bicubic":
        return functools.partial(core.resize.Bicubic,filter_param_a=b,filter_param_b=c)
    elif kernel == "Lanczos":
        return functools.partial(core.resize.Lanczos,filter_param_a=taps)

def MRcore(clip:vs.VideoNode,kernel:str,w:int,h:int,mask: Union[bool,vs.VideoNode]=True,mask_dif_pix:float=2,postfilter_descaled=None,mthr:list[int]=[2,2],taps:int=3,b:float=0,c:float=0.5,multiple:float=1,maskpp=None,show:str="result",**args):
    src_w,src_h=clip.width,clip.height
    clip32=core.fmtc.bitdepth(clip,bits=32)
    descaled=core.descale.Descale(clip32,width=w,height=h,kernel=kernel.lower(),taps=taps,b=b,c=c)
    upscaled=resize_core(kernel.capitalize(),taps,b,c)(descaled,src_w,src_h)
    diff=Expr([clip32,upscaled],"x y - abs dup 0.015 > swap 0 ?").std.PlaneStats()
    
    def calc(n,f): 
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]*multiple
        return fout

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        descaled=postfilter_descaled(descaled)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")
    etype=args.get("etype")
    pscrn=args.get("pscrn")
    exp=args.get("exp")
    sigmoid=args.get("sigmoid")

    rescale=nnedi3_resample(descaled,src_w,src_h,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,sigmoid=sigmoid).fmtc.bitdepth(bits=16)

    if mask is True:
        mask=Expr([clip,upscaled.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(maskpp):
            mask=maskpp(mask)
        else:
            mask=expand(mask,cycle=mthr[0])
            mask=inpand(mask,cycle=mthr[1])
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    elif isinstance(mask,vs.VideoNode):
        if mask.width!=src_w or mask.height!=src_h or mask.format.color_family!=vs.GRAY:
            raise ValueError("mask should have same resolution as source,and should be GRAY")
        mask=core.fmtc.bitdepth(mask,bits=16,dmode=1)
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    else:
        mask=core.std.BlankClip(rescale)

    if show.lower()=="result":
        return core.std.ModifyFrame(rescale,[diff,rescale],calc)
    elif show.lower()=="mask" and mask:
        return core.std.ModifyFrame(mask,[diff,mask],calc)
    elif show.lower()=="descale":
        return descaled #after postfilter_descaled
    elif show.lower()=="both": #result,mask,descaled
        return core.std.ModifyFrame(rescale,[diff,rescale],calc),core.std.ModifyFrame(mask,[diff,mask],calc),descaled

def MRcoref(clip:vs.VideoNode,kernel:str,w:float,h:float,bh:int,bw:int=None,mask: Union[bool,vs.VideoNode]=True,mask_dif_pix:float=2,postfilter_descaled=None,mthr:list[int]=[2,2],taps:int=3,b:float=0,c:float=0.5,multiple:float=1,maskpp=None,show:str="result",**args):

    src_w,src_h=clip.width,clip.height
    cargs=cropping_args(src_w,src_h,h,bh,bw)
    clip32=core.fmtc.bitdepth(clip,bits=32)
    descaled=core.descale.Descale(clip32,kernel=kernel.lower(),taps=taps,b=b,c=c,**cargs.descale_gen())
    upscaled=resize_core(kernel.capitalize(),taps,b,c)(descaled,**cargs.resize_gen())
    diff=Expr([clip32,upscaled],"x y - abs dup 0.015 > swap 0 ?").std.Crop(10, 10, 10, 10).std.PlaneStats()
    def calc(n,f): 
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]*multiple
        return fout

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        descaled=postfilter_descaled(descaled)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")
    etype=args.get("etype")
    pscrn=args.get("pscrn")
    exp=args.get("exp")
    sigmoid=args.get("sigmoid")

    rescale=nnedi3_resample(descaled,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,sigmoid=sigmoid,**cargs.nnrs_gen()).fmtc.bitdepth(bits=16)

    if mask is True:
        mask=Expr([clip,upscaled.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(maskpp):
            mask=maskpp(mask)
        else:
            mask=expand(mask,cycle=mthr[0])
            mask=inpand(mask,cycle=mthr[1])
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    elif isinstance(mask,vs.VideoNode):
        if mask.width!=src_w or mask.height!=src_h or mask.format.color_family!=vs.GRAY:
            raise ValueError("mask should have same resolution as source,and should be GRAY")
        mask=core.fmtc.bitdepth(mask,bits=16,dmode=1)
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    else:
        mask=core.std.BlankClip(rescale)

    if show.lower()=="result":
        return core.std.ModifyFrame(rescale,[diff,rescale],calc)
    elif show.lower()=="mask" and mask:
        return core.std.ModifyFrame(mask,[diff,mask],calc)
    elif show.lower()=="descale":
        return descaled #after postfilter_descaled
    elif show.lower()=="both": #result,mask,descaled
        return core.std.ModifyFrame(rescale,[diff,rescale],calc),core.std.ModifyFrame(mask,[diff,mask],calc),descaled

def MRkernelgen(k,w=None,h=None,b=None,c=None,taps=None,mask=None,mask_dif_pix=None,mthr=None,pp=None,multiple=None,maskpp=None,fmode=None):
    l=locals()
    tmp={}
    for i in l.keys():
        if l[i] is not None:
            tmp[i]=l[i]
    return tmp

@deprecated("Not real needed function,no maintenance.")
def dpidDown(src,width=None,height=None,Lambda=1.0,matrix_in=None,matrix=None,transfer_in="709",transfer=None,
               primaries_in=None,primaries=None,css=None,depth=16,dither_type="error_diffusion",range_in=None,range_out=None):
    """
    dpidDown
    --------------------------------
    use dpid as kernel in Gamma-aware resize,only downscale.
    need CUDA-Enabled GPU
    """
    def M(a,b):
        if a <= 1024 and b <= 576:
            return "170m"
        elif a <= 2048 and b <= 1536:
            return "709"
        else :
            return "2020ncl"
    ##############
    if width is None:
        width = src.width
    if height is None:
        height = src.height
    if width>src.width or height > src.height:
        raise ValueError("")
    isRGB=src.format.color_family==vs.RGB
    if transfer is None:
        transfer=transfer_in
    if matrix is None:
        matrix=M(width,height)
    if isRGB:
        matrix_in="rgb"
    if css is not None:
        css=str(css).lower()
        if css == "444" or css == "4:4:4":
            css = "11"
        elif css == "440" or css == "4:4:0":
            css = "12"
        elif css == "422" or css == "4:2:2":
            css = "21"
        elif css == "420" or css == "4:2:0":
            css = "22"
        elif css == "411" or css == "4:1:1":
            css = "41"
        elif css == "410" or css == "4:1:0":
            css = "42"
        if css not in ["11","12","21","22","41","42","rgb"]:
            raise ValueError("")
    if range_in is None:
        range_in="full" if isRGB else "limited"
    if range_out is None:
        if css is None:
            range_out=range_in
        elif isRGB and css=="rgb":
            range_out=range_in
        elif not isRGB and css!="rgb":
            range_out=range_in
        elif isRGB and css!="rgb":
            range_out="limited"
        else:
            range_out="full"
    range_in=range_in.lower()
    range_out=range_out.lower()
    if range_in=="tv":
        range_in="limited"
    if range_in=="pc":
        range_in="full"
    if range_in not in ["limited","full"]:
        raise ValueError("")
    if range_out=="tv":
        range_out="limited"
    if range_out=="pc":
        range_out="full"
    if range_out not in ["limited","full"]:
        raise ValueError("")
    rgb=core.resize.Bicubic(src,format=vs.RGBS,matrix_in_s=matrix_in,transfer_in_s=transfer_in,primaries_in_s=primaries_in,primaries_s=primaries,range_in_s=range_in)
    lin=core.resize.Bicubic(rgb,format=vs.RGB48,transfer_in_s=transfer_in,transfer_s="linear")
    res=core.dpid.Dpid(lin,width,height,Lambda)
    res=core.resize.Bicubic(res,format=vs.RGBS,transfer_in_s="linear",transfer_s=transfer)
    if not isRGB and css!="rgb":
        st=vs.FLOAT if depth==32 else vs.INTEGER
        if css is None:
            sh=src.format.subsampling_h
            sw=src.format.subsampling_w
        else:
            sh=int(css[0])//2
            sw=int(css[1])//2
        outformat=core.register_format(vs.YUV,st,depth,sh,sw)
        last=core.resize.Bicubic(res,format=vs.YUV444PS,matrix_s=matrix,range_s=range_out)
        last=core.resize.Bicubic(last,format=outformat.id,dither_type=dither_type)
    else:
        if css is None or css=="rgb":
            last=core.resize.Bicubic(res,range_s=range_out)
        else:
            sh=int(css[0])//2
            sw=int(css[1])//2
            st=vs.FLOAT if depth==32 else vs.INTEGER
            outformat=core.register_format(vs.YUV,st,depth,sh,sw)
            last=core.resize.Bicubic(res,format=vs.YUV444PS,matrix_s=matrix,range_s=range_out)
            last=core.resize.Bicubic(last,format=outformat.id,dither_type=dither_type)
    return last

class cropping_args:
    #rewrite from function descale_cropping_args in getfnative
    def __init__(self,width:int,height:int,src_height: float, base_height: int, base_width= None, mode: str = 'wh'):
        assert base_height >= src_height
        self.mode=mode

        self.width=width
        self.height=height
        self.base_width=self.getw(base_height) if base_width is None else base_width
        self.base_height=base_height
        self.src_width=src_height * width / height
        self.src_height=src_height

        self.cropped_width=self.base_width - 2 * math.floor((self.base_width - self.src_width) / 2)
        self.cropped_height=self.base_height - 2 * math.floor((self.base_height - self.src_height) / 2)

    def descale_gen(self):
        args={"width":self.width,"height":self.height}
        argsw={"width":self.cropped_width,"src_width":self.src_width,"src_left":(self.cropped_width-self.src_width)/2}
        argsh={"height":self.cropped_height,"src_height":self.src_height,"src_top":(self.cropped_height-self.src_height)/2}

        if "w" in self.mode:
            args.update(argsw)
        if "h" in self.mode:
            args.update(argsh)
        return args

    def resize_gen(self):
        args={"width":self.width,"height":self.height}
        argsw={"src_width":self.src_width,"src_left":(self.cropped_width-self.src_width)/2}
        argsh={"src_height":self.src_height,"src_top":(self.cropped_height-self.src_height)/2}

        if "w" in self.mode:
            args.update(argsw)
        if "h" in self.mode:
            args.update(argsh)
        return args

    def nnrs_gen(self):
        args={"target_width":self.width,"target_height":self.height}
        argsw={"src_width":self.src_width,"src_left":(self.cropped_width-self.src_width)/2}
        argsh={"src_height":self.src_height,"src_top":(self.cropped_height-self.src_height)/2}

        if "w" in self.mode:
            args.update(argsw)
        if "h" in self.mode:
            args.update(argsh)
        return args

    def getw(self, height: int):
        width = math.ceil(height * self.width / self.height)
        if height % 2 == 0:
            width = width // 2 * 2
        return width
