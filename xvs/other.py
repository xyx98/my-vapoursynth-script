from .utils import *

import os
import re


def splicev1(clip=[],num=[],den=[],tc_out="tc v1.txt"):
    """
    splice clips with different fps and output timecodes v1
    """
    clip_len = len(clip)
    num_len = len(num)
    den_len = len(den)
    if clip_len > num_len:
        for i in range(num_len,clip_len):
            num.append(None)
    if clip_len > den_len:
        for i in range(den_len,clip_len):
            den.append(None)
    for i in range(0,clip_len):
        if num[i] is None:
            num[i] = clip[i].fps_num
            den[i] = clip[i].fps_den
        elif den[i] is None:
            if num[i] > 10000:
                den[i] = 1001
            else:
                den[i]=1
    fps=[]
    for i in range(0,clip_len):
        fps.append(float(num[i])/den[i])
    fnum=[i.num_frames for i in clip]
    for i in range(1,clip_len):
        fnum[i]+=fnum[i-1]
    tc = open(tc_out,"w")
    tc.write("# timecode format v1\nassume "+str(fps[0])+"\n")
    for i in range(1,clip_len):
        tc.write(str(fnum[i-1])+","+str(fnum[i]-1)+","+str(fps[i])+"\n")
    tc.close()
    last = clip[0]
    for i in range(1,clip_len):
        last+=clip[i]
    last=core.std.AssumeFPS(last,fpsnum=num[0],fpsden=den[0])
    return last

def mvfrc(input,it=140,scp=15,num=60000,den=1001,preset='fast',
        pel=2,block=True,flow_mask=None,block_mode=None,
        blksize = 8,blksizev=8,search=None,truemotion=True,searchparam=2,overlap=0,overlapv=None,
        dct=0,blend=True,badSAD=10000,badrange=24,divide=0,ml=100,Mblur=15):
    """
    change fps by mvtools with motion interpolation
    it = thscd1    ;    scp=thscd2/255*100
    """
    funcName = 'mvfrc'
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': This is not a clip!')
    #############
    if preset == 'fast':
        pnum=0
    elif preset == 'medium':
        pnum=1
    elif preset == 'slow':
        pnum=2
    else:
        raise TypeError(funcName + r":preset should be fast\ medium\slow'")
    overlapv = overlap
    #############
    if search is None : search = [0,3,3][pnum]
    if block_mode is None : block_mode = [0,0,3][pnum]
    if flow_mask is None : flow_mask = [0,0,2][pnum]
    #############
    analParams = {
        'overlap' : overlap,
        'overlapv':overlapv,
        'search' : search,
        'dct':dct,
        'truemotion' : truemotion,
        'blksize' : blksize,
        'blksizev':blksizev,
        'searchparam':searchparam,
        'badsad':badSAD,
        'badrange':badrange,
        'divide':divide
        }
    ############
    #block or flow Params 
    bofp = {
        'thscd1':it,
        'thscd2':int(scp*255/100),
        'blend':blend,
        'num':num,
        'den':den
        }
    ############
    sup = core.mv.Super(input, pel=pel,sharp=2, rfilter=4)
    bvec = core.mv.Analyse(sup, isb=True, **analParams)
    fvec = core.mv.Analyse(sup, isb=False, **analParams)
    if input.fps_num/input.fps_den > num/den:
        input = core.mv.FlowBlur(input, sup, bvec, fvec, blur=Mblur)
    if block == True:
        clip =  core.mv.BlockFPS(input, sup, bvec, fvec,**bofp,mode=block_mode)
    else:
        clip = core.mv.FlowFPS(input, sup, bvec, fvec,**bofp,mask=flow_mask)
    return clip

@deprecated("No maintenance.")
def textsub(input,file,charset=None,fps=None,vfr=None,
              mod=False,Matrix=None):
    """
    ---------------------------
    textsub
    ---------------------------
    Can support high bit and yuv444&yuv422,but not rgb
    Not recommended for yuv420p8
    It's a port from avs script—textsub16 by mawen1250,but have some differences
    ---------------------------
    input,file,charset,fps,vfr: same in vsfilter
    mod: Choose whether to use(vsfiler or vsfiltermod),deflaut is False,means use vsfilter
    """
    def M(a,b):
        if a <= 1024 and b <= 576:
            return "601"
        elif a <= 2048 and b <= 1536:
            return "709"
        else :
            return "2020"
    
    funcName = "textsub16"
    width  = input.width
    height = input.height
    bit    = input.format.bits_per_sample
    U      = getplane(mask,1)
    c_w    = U.width
    c_h    = U.height
    w      = width/c_w
    h      = height/c_h
    if w==1 and h==1:
        f  = vs.YUV444P16
        s  = 0
    elif  w==2 and h==1:
        f  = vs.YUV422P16
        s  = 0.5
    elif  w==2 and h==2:
        f  = vs.YUV420P16
        s  = 0.25
    else:
        TypeError(funcName + ': Only support YUV420、YUV422、YUV444')
    if Matrix is None:
        Matrix = M(width,height)
    ##############
    def vsmode(clip,file,charset,fps,vfr,mod):
        core = vs.core
        if mod == False:
            last = core.vsf.TextSub(clip,file,charset,fps,vfr)
        else:
            last = core.vsfm.TextSubMod(clip,file,charset,fps,vfr)
        return core.std.Cache(last, make_linear=True)

    def mskE(a,b,depth):
        core=vs.core
        expr="x y - abs 1 < 0 255 ?"
        last = Expr([a,b], [expr]*3)
        return core.fmtc.bitdepth(last,bits=depth)
    ##############
    src8  = core.resize.Bilinear(input,format=vs.YUV420P8)
    sub8  = vsmode(src8,file,charset,fps,vfr,mod)
    mask  = mskE(src8,sub8,bit)
    maskY = getplane(mask,0)
    maskU = getplane(mask,1)
    maskU = core.resize.Bilinear(maskU,width,height,src_left=s)
    maskV = getplane(mask,2)
    maskV = core.resize.Bilinear(maskV,width,height,src_left=s)
    mask  = Expr([maskY,maskU,maskV],"x y max z max")
    mask  = core.std.Inflate(mask)
    maskC = core.resize.Bilinear(mask,c_w,c_h,src_left=-s)#,src_width=c_w,src_height=c_h
    if w==1 and h==1:
        mask = core.std.ShufflePlanes(mask,[0,0,0], colorfamily=vs.YUV)
    else:
        mask = core.std.ShufflePlanes([mask,maskC],[0,0,0], colorfamily=vs.YUV)
    ################
    rgb   = core.resize.Bilinear(input,format=vs.RGB24,matrix_in_s=Matrix)
    sub   = vsmode(rgb,file,charset,fps,vfr,mod)
    sub   = core.resize.Bilinear(sub,format=f,matrix_s=Matrix)
    sub   = core.fmtc.bitdepth(sub,bits=bit)
    last  = core.std.MaskedMerge(input, sub, mask=mask, planes=[0,1,2])
    return last

@deprecated("No maintenance.")
def vfrtocfr(clip=None,tc=None,num=None,den=1,blend=False):
    """
    vfrtocfr
    --------------------------------
    clip: input clip
    tc: input timecodes,only support tcv2
    num,den: output fps=num/den
    blend: True means blend the frames instead of delete or copy , default is False
    """
    def tclist(f):
        A=open(f,"r")
        B=re.sub(r'(.\n)*# timecode format v2(.|\n)*\n0',r'0',A.read(),count=1)
        A.close()
        C=B.split()
        T=[]
        for i in C:
            T.append(int(float(i)))
        return T
    #################
    vn = clip.num_frames
    vtc = tclist(tc)
    cn = int(vtc[-1]*num/den/1000)
    ctc = [int(1000*den/num)*i for i in range(0,cn+1)]
    cc = clip[0]
    for i in range(1,cn+1):
        for j in range(1,vn+1):
            if ctc[i]<vtc[1]:
                cc += clip[0]
            elif ctc[i]>=vtc[j] and ctc[i]<vtc[j+1]:
                if blend == False:
                    cl=clip[j-1] if (ctc[i]-vtc[j])>(vtc[j+1]-ctc[i]) else clip[j]
                else:
                    cl=core.std.Merge(clip[j-1],clip[j],weight=(ctc[i]-vtc[j])/(vtc[j+1]-vtc[j]))
                cc += cl
    last = core.std.AssumeFPS(cc,fpsnum=num,fpsden=den)
    return core.std.Cache(last, make_linear=True)

@deprecated("No maintenance.")
def Overlaymod(clipa, clipb, x=0, y=0, alpha=None,aa=False):
    """
    Overlaymod
    -------------------------
    modified overlay by xyx98,
    orginal Overlay by holy in havsfunc
    -------------------------
    difference: mask->alpha,please input the alpha clip read by imwri if needed
                   aa: if is True,use daa in clipb and alpha clip
    """
    if not (isinstance(clipa, vs.VideoNode) and isinstance(clipb, vs.VideoNode)):
        raise TypeError('Overlaymod: This is not a clip')
    if clipa.format.subsampling_w > 0 or clipa.format.subsampling_h > 0:
        clipa_src = clipa
        clipa = core.resize.Point(clipa, format=core.register_format(clipa.format.color_family, clipa.format.sample_type, clipa.format.bits_per_sample, 0, 0).id)
    else:
        clipa_src = None
    if clipb.format.id != clipa.format.id:
        clipb = core.resize.Point(clipb, format=clipa.format.id)
    mask = core.std.BlankClip(clipb, color=[(1 << clipb.format.bits_per_sample) - 1] * clipb.format.num_planes)
    if not isinstance(alpha, vs.VideoNode):
        raise TypeError("Overlaymod: 'alpha' is not a clip")
    if mask.width != clipb.width or mask.height != clipb.height:
        raise TypeError("Overlaymod: 'alpha' must be the same dimension as 'clipb'")

    if aa:
        from .aa import daa
        clipb = daa(clipb)
    # Calculate padding sizes
    l, r = x, clipa.width - clipb.width - x
    t, b = y, clipa.height - clipb.height - y
    # Split into crop and padding values
    cl, pl = min(l, 0) * -1, max(l, 0)
    cr, pr = min(r, 0) * -1, max(r, 0)
    ct, pt = min(t, 0) * -1, max(t, 0)
    cb, pb = min(b, 0) * -1, max(b, 0)
    # Crop and padding
    def cap(clip):
        mask = getplane(clip, 0)
        mask = core.std.CropRel(mask, cl, cr, ct, cb)
        mask = core.std.AddBorders(mask, pl, pr, pt, pb)
        return mask
    clipb = core.std.CropRel(clipb, cl, cr, ct, cb)
    clipb = core.std.AddBorders(clipb, pl, pr, pt, pb)
    # Return padded clip
    mask = cap(mask)
    last = core.std.MaskedMerge(clipa, clipb, mask)
    if alpha is not None:
        alpha=core.fmtc.bitdepth(alpha,bits=clipb.format.bits_per_sample)
        m = 1<<alpha.format.bits_per_sample
        alpha = core.std.Levels(alpha, min_in=16/256*m, max_in=235/256*m, min_out=m-1, max_out=0)
        if aa:
            alpha = daa(alpha)
        mask = cap(alpha)
        last = core.std.MaskedMerge(clipa, last, mask)
    if clipa_src is not None:
        last = core.resize.Point(last, format=clipa_src.format.id)
    return last

@deprecated("No maintenance.")
def InterFrame(Input, Preset='Medium', Tuning='Film', NewNum=None, NewDen=1, GPU=True, gpuid=0,InputType='2D', OverrideAlgo=None, OverrideArea=None,
               FrameDouble=False):
    """
    adjusted InterFrame from havsfunc,support 10bit with new svp
    """
    if not isinstance(Input, vs.VideoNode):
        raise TypeError('InterFrame: This is not a clip')

    sw=Input.format.subsampling_w
    sh=Input.format.subsampling_h
    depth= Input.format.bits_per_sample
    if  not sw ==1 and not sh==1 and depth not in [8,10]:
        raise TypeError('InterFrame: input must be yuv420p8 or yuv420p10')
    oInput=Input
    Input = core.fmtc.bitdepth(Input,bits=8)
    # Validate inputs
    Preset = Preset.lower()
    Tuning = Tuning.lower()
    InputType = InputType.upper()
    if Preset not in ['medium', 'fast', 'faster', 'fastest']:
        raise ValueError("InterFrame: '{Preset}' is not a valid preset".format(Preset=Preset))
    if Tuning not in ['film', 'smooth', 'animation', 'weak']:
        raise ValueError("InterFrame: '{Tuning}' is not a valid tuning".format(Tuning=Tuning))
    if InputType not in ['2D', 'SBS', 'OU', 'HSBS', 'HOU']:
        raise ValueError("InterFrame: '{InputType}' is not a valid InputType".format(InputType=InputType))
    
    def InterFrameProcess(clip,oclip):
        # Create SuperString
        if Preset in ['fast', 'faster', 'fastest']:
            SuperString = '{pel:1,'
        else:
            SuperString = '{'
        
        SuperString += 'gpu:1}' if GPU else 'gpu:0}'
        
        # Create VectorsString
        if Tuning == 'animation' or Preset == 'fastest':
            VectorsString = '{block:{w:32,'
        elif Preset in ['fast', 'faster'] or not GPU:
            VectorsString = '{block:{w:16,'
        else:
            VectorsString = '{block:{w:8,'
        
        if Tuning == 'animation' or Preset == 'fastest':
            VectorsString += 'overlap:0'
        elif Preset == 'faster' and GPU:
            VectorsString += 'overlap:1'
        else:
            VectorsString += 'overlap:2'
        
        if Tuning == 'animation':
            VectorsString += '},main:{search:{coarse:{type:2,'
        elif Preset == 'faster':
            VectorsString += '},main:{search:{coarse:{'
        else:
            VectorsString += '},main:{search:{distance:0,coarse:{'
        
        if Tuning == 'animation':
            VectorsString += 'distance:-6,satd:false},distance:0,'
        elif Tuning == 'weak':
            VectorsString += 'distance:-1,trymany:true,'
        else:
            VectorsString += 'distance:-10,'
        
        if Tuning == 'animation' or Preset in ['faster', 'fastest']:
            VectorsString += 'bad:{sad:2000}}}}}'
        elif Tuning == 'weak':
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250,search:{distance:-1,satd:true}}]}'
        else:
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250}]}'
        
        # Create SmoothString
        if NewNum is not None:
            SmoothString = '{rate:{num:' + repr(NewNum) + ',den:' + repr(NewDen) + ',abs:true},'
        elif clip.fps_num / clip.fps_den in [15, 25, 30] or FrameDouble:
            SmoothString = '{rate:{num:2,den:1,abs:false},'
        else:
            SmoothString = '{rate:{num:60000,den:1001,abs:true},'
        if GPU:
            SmoothString+= 'gpuid:'+repr(gpuid)+','
        if OverrideAlgo is not None:
            SmoothString += 'algo:' + repr(OverrideAlgo) + ',mask:{cover:80,'
        elif Tuning == 'animation':
            SmoothString += 'algo:2,mask:{'
        elif Tuning == 'smooth':
            SmoothString += 'algo:23,mask:{'
        else:
            SmoothString += 'algo:13,mask:{cover:80,'
        
        if OverrideArea is not None:
            SmoothString += 'area:{OverrideArea}'.format(OverrideArea=OverrideArea)
        elif Tuning == 'smooth':
            SmoothString += 'area:150'
        else:
            SmoothString += 'area:0'
        
        if Tuning == 'weak':
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0,limits:{blocks:50}}}'
        else:
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0}}'
        
        # Make interpolation vector clip
        Super = core.svp1.Super(clip, SuperString)
        Vectors = core.svp1.Analyse(Super['clip'], Super['data'], clip, VectorsString)
        
        # Put it together
        return core.svp2.SmoothFps(oclip, Super['clip'], Super['data'], Vectors['clip'], Vectors['data'], SmoothString)
    
    # Get either 1 or 2 clips depending on InputType
    if InputType == 'SBS':
        FirstEye = InterFrameProcess(core.std.CropRel(Input, right=Input.width // 2),
                                     core.std.CropRel(oInput, right=Input.width // 2))
        SecondEye = InterFrameProcess(core.std.CropRel(Input, left=Input.width // 2),
                                      core.std.CropRel(oInput, left=Input.width // 2))
        return core.std.StackHorizontal([FirstEye, SecondEye])
    elif InputType == 'OU':
        FirstEye = InterFrameProcess(core.std.CropRel(Input, bottom=Input.height // 2),
                                     core.std.CropRel(oInput, bottom=Input.height // 2))
        SecondEye = InterFrameProcess(core.std.CropRel(Input, top=Input.height // 2),
                                      core.std.CropRel(oInput, top=Input.height // 2))
        return core.std.StackVertical([FirstEye, SecondEye])
    elif InputType == 'HSBS':
        FirstEye = InterFrameProcess(core.std.CropRel(Input, right=Input.width // 2).resize.Spline36(Input.width, Input.height),
                                     core.std.CropRel(oInput, right=oInput.width // 2).resize.Spline36(oInput.width, oInput.height))
        SecondEye = InterFrameProcess(core.std.CropRel(Input, left=Input.width // 2).resize.Spline36(Input.width, Input.height),
                                      core.std.CropRel(oInput, left=oInput.width // 2).resize.Spline36(oInput.width, oInput.height))
        return core.std.StackHorizontal([core.resize.Spline36(FirstEye, Input.width // 2, Input.height),
                                         core.resize.Spline36(SecondEye, Input.width // 2, Input.height)])
    elif InputType == 'HOU':
        FirstEye = InterFrameProcess(core.std.CropRel(Input, bottom=Input.height // 2).resize.Spline36(Input.width, Input.height),
                                     core.std.CropRel(oInput, bottom=oInput.height // 2).resize.Spline36(oInput.width, oInput.height))
        SecondEye = InterFrameProcess(core.std.CropRel(Input, top=Input.height // 2).resize.Spline36(Input.width, Input.height),
                                      core.std.CropRel(oInput, top=oInput.height // 2).resize.Spline36(oInput.width, oInput.height))
        return core.std.StackVertical([core.resize.Spline36(FirstEye, Input.width, Input.height // 2),
                                       core.resize.Spline36(SecondEye, Input.width, Input.height // 2)])
    else:
        return InterFrameProcess(Input,oInput)

@deprecated("Do not use!")
def xTonemap(clip,nominal_luminance=400,exposure=4.5):
    """
    The way I convert hdr to sdr when I rip 'Kimi No Na Wa'(UHDBD HK ver.).
    I'm not sure It suit for other UHDBD
    ###
    nominal_luminance: nominal_luminance when convert to linear RGBS
    exposure: exposure in Mobius,which do the tonemap
    """

    fid=clip.format.id
    clip=core.resize.Spline36(clip=clip, format=vs.RGBS,range_in_s="limited", matrix_in_s="2020ncl", primaries_in_s="2020", primaries_s="2020", transfer_in_s="st2084", transfer_s="linear",dither_type="none", nominal_luminance=nominal_luminance)
    clip=core.tonemap.Mobius(clip,exposure=exposure)
    clip=core.resize.Spline36(clip, format=fid,matrix_s="709", primaries_in_s="2020", primaries_s="709", transfer_in_s="linear", transfer_s="709",dither_type="none")
    clip=core.std.Expr(clip,["x 4096 - 219 * 235 / 4096 +",""])
    return clip

@deprecated("No maintenance.")
def readmpls(path:str,sfilter='ffms2',cache=None):
    mpls = core.mpls.Read(path)
    if sfilter in ["ffms2","ffms","ff","f","ffvideosource"]:
        if cache is None or cache==0:
            cache=os.path.join(os.getcwd(),'cache')
        elif isinstance(cache,str):
            pass
        elif cache==-1:
            cache=False
        else:
            raise ValueError('unknown cache setting')
        
        if cache:
            clips=[]
            for i in range(mpls['count']):
                clips.append(core.ffms2.Source(source=mpls['clip'][i], cachefile=os.path.join(cache,mpls['filename'][i].decode()+'.ffindex')))
        else:
            clips=[core.ffms2.Source(mpls['clip'][i]) for i in range(mpls['count'])]
    elif sfilter in ['lwi','l','lsmash','l-smash','lsmas','LWLibavSource']:
        clips=[core.lsmas.LWLibavSource(mpls['clip'][i]) for i in range(mpls['count'])]
    else:
        raise ValueError("unknown source filter")
    return core.std.Splice(clips)

#copy from muvsfunc
def LDMerge(flt_h: vs.VideoNode, flt_v: vs.VideoNode, src: vs.VideoNode, mrad: int = 0,
            show: bool = False, planes: PlanesType = None,
            convknl: int = 1, conv_div: Optional[int] = None, calc_mode: int = 0,
            power: float = 1.0
            ) -> vs.VideoNode:
    """Merges two filtered clips based on the gradient direction map from a source clip.

    Args:
        flt_h, flt_v: Two filtered clips.

        src: Source clip. Must be the same format as the filtered clips.

        mrad: (int) Expanding of gradient direction map. Default is 0.

        show: (bool) Whether to output gradient direction map. Default is False.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the first clip, "flt_h".

        convknl: (0 or 1) Convolution kernel used to generate gradient direction map.
            0: Seconde order center difference in one direction and average in perpendicular direction
            1: First order center difference in one direction and weighted average in perpendicular direction.
            Default is 1.

        conv_div: (int) Divisor in convolution filter. Default is the max value in convolution kernel.

        calc_mode: (0 or 1) Method used to calculate the gradient direction map. Default is 0.

        power: (float) Power coefficient in "calc_mode=0".

    Example:
        # Fast anti-aliasing
        horizontal = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='h')
        vertical = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='v')
        blur_src = core.tcanny.TCanny(clip, mode=-1, planes=[0]) # Eliminate noise
        antialiasing = muf.LDMerge(horizontal, vertical, blur_src, mrad=1, planes=[0])

    """

    funcName = 'LDMerge'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')

    if not isinstance(flt_h, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_h\" must be a clip!')
    if src.format.id != flt_h.format.id:
        raise TypeError(funcName + ': \"flt_h\" must be of the same format as \"src\"!')
    if src.width != flt_h.width or src.height != flt_h.height:
        raise TypeError(funcName + ': \"flt_h\" must be of the same size as \"src\"!')

    if not isinstance(flt_v, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_v\" must be a clip!')
    if src.format.id != flt_v.format.id:
        raise TypeError(funcName + ': \"flt_v\" must be of the same format as \"src\"!')
    if src.width != flt_v.width or src.height != flt_v.height:
        raise TypeError(funcName + ': \"flt_v\" must be of the same size as \"src\"!')

    if not isinstance(mrad, int):
        raise TypeError(funcName + '\"mrad\" must be an int!')

    if not isinstance(show, int):
        raise TypeError(funcName + '\"show\" must be an int!')
    if show not in list(range(0, 4)):
        raise ValueError(funcName + '\"show\" must be in [0, 1, 2, 3]!')

    if planes is None:
        planes = list(range(flt_h.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    bits = flt_h.format.bits_per_sample

    if convknl == 0:
        convknl_h = [-1, -1, -1, 2, 2, 2, -1, -1, -1]
        convknl_v = [-1, 2, -1, -1, 2, -1, -1, 2, -1]
    else: # convknl == 1
        convknl_h = [-17, -61, -17, 0, 0, 0, 17, 61, 17]
        convknl_v = [-17, 0, 17, -61, 0, 61, -17, 0, 17]

    if conv_div is None:
        conv_div = max(convknl_h)

    hmap = core.std.Convolution(src, matrix=convknl_h, saturate=False, planes=planes, divisor=conv_div)
    vmap = core.std.Convolution(src, matrix=convknl_v, saturate=False, planes=planes, divisor=conv_div)

    if mrad > 0:
        hmap = mt_expand_multi(hmap, sw=0, sh=mrad, planes=planes)
        vmap = mt_expand_multi(vmap, sw=mrad, sh=0, planes=planes)
    elif mrad < 0:
        hmap = mt_inpand_multi(hmap, sw=0, sh=-mrad, planes=planes)
        vmap = mt_inpand_multi(vmap, sw=-mrad, sh=0, planes=planes)

    if calc_mode == 0:
        ldexpr = '{peak} 1 x 0.0001 + y 0.0001 + / {power} pow + /'.format(peak=(1 << bits) - 1, power=power)
    else:
        ldexpr = 'y 0.0001 + x 0.0001 + dup * y 0.0001 + dup * + sqrt / {peak} *'.format(peak=(1 << bits) - 1)
    ldmap = core.std.Expr([hmap, vmap], [(ldexpr if i in planes else '') for i in range(src.format.num_planes)])

    if show == 0:
        return core.std.MaskedMerge(flt_h, flt_v, ldmap, planes=planes)
    elif show == 1:
        return ldmap
    elif show == 2:
        return hmap
    elif show == 3:
        return vmap
    else:
        raise ValueError

#simplified from mvsfunc.Depth,only lowbitdepth part left.
def lowbitdepth_sim(clip:vs.VideoNode,depth:int,dither:int=1,fulls:bool=None,fulld:bool=None):
    if not 0 < depth < 8 : 
        raise vs.Error("lowbitdepth_sim: depth should be positive integer and less than 8.")

    sFormat = clip.format

    sColorFamily = sFormat.color_family
    sIsYUV = sColorFamily == vs.YUV
    sIsGRAY = sColorFamily == vs.GRAY

    sbitPS = sFormat.bits_per_sample
    sSType = sFormat.sample_type

    dSType = vs.INTEGER
    dbitPS = 8
    
    if fulls is None:
        fulls = False if sIsYUV or sIsGRAY else True
    if fulld is None:
        fulld=fulls

    #from mvsfunc
    def _quantization_parameters(sample=None, depth=None, full=None, chroma=None):
        qp = {}

        if sample is None:
            sample = vs.INTEGER
        if depth is None:
            depth = 8
        elif depth < 1:
            raise vs.Error('"depth" should not be less than 1!')
        if full is None:
            full = True
        if chroma is None:
            chroma = False

        lShift = depth - 8
        rShift = 8 - depth

        if sample == vs.INTEGER:
            if chroma:
                qp['floor'] = 0 if full else 16 << lShift if lShift >= 0 else 16 >> rShift
                qp['neutral'] = 128 << lShift if lShift >= 0 else 128 >> rShift
                qp['ceil'] = (1 << depth) - 1 if full else 240 << lShift if lShift >= 0 else 240 >> rShift
                qp['range'] = qp['ceil'] - qp['floor']
            else:
                qp['floor'] = 0 if full else 16 << lShift if lShift >= 0 else 16 >> rShift
                qp['neutral'] = qp['floor']
                qp['ceil'] = (1 << depth) - 1 if full else 235 << lShift if lShift >= 0 else 235 >> rShift
                qp['range'] = qp['ceil'] - qp['floor']
        elif sample == vs.FLOAT:
            if chroma:
                qp['floor'] = -0.5
                qp['neutral'] = 0.0
                qp['ceil'] = 0.5
                qp['range'] = qp['ceil'] - qp['floor']
            else:
                qp['floor'] = 0.0
                qp['neutral'] = qp['floor']
                qp['ceil'] = 1.0
                qp['range'] = qp['ceil'] - qp['floor']
        else:
            raise vs.Error('Unsupported "sample" specified!')

        return qp
    
    #simplified from mvsfunc
    def _quantization_conversion(clip, depths=None, depthd=None, sample=None, fulls=None, fulld=None,
        chroma=None, clamp=None, dbitPS=None, mode=None):
        # input clip

        # Get properties of input clip
        sFormat = clip.format

        sColorFamily = sFormat.color_family
        sIsYUV = sColorFamily == vs.YUV
        sIsGRAY = sColorFamily == vs.GRAY

        sbitPS = sFormat.bits_per_sample
        sSType = sFormat.sample_type

        if depths is None:
            depths = sbitPS

        if fulls is None:
            # If not set, assume limited range for YUV and Gray input
            fulls = False if sIsYUV or sIsGRAY else True

        if chroma is None:
            chroma = False
        elif not sIsGRAY:
            chroma = False

        # Get properties of output clip
        if depthd is None:
            pass

        if sample is None:
            if depthd is None:
                dSType = sSType
                depthd = depths
            else:
                dSType = vs.FLOAT if dbitPS >= 32 else vs.INTEGER
        else:
            dSType = sample

        if fulld is None:
            fulld = fulls


        if clamp is None:
            clamp = dSType == vs.INTEGER

        if dbitPS is None:
            if depthd < 8:
                dbitPS = 8
            else:
                dbitPS = depthd

        if mode is None:
            mode = 0
        elif depthd >= 8:
            mode = 0

        dFormat = core.query_video_format(sFormat.color_family, dSType, dbitPS, sFormat.subsampling_w, sFormat.subsampling_h)

        # Expression function
        def gen_expr(chroma, mode):
            if dSType == vs.INTEGER:
                exprLower = 0
                exprUpper = 1 << (dFormat.bytes_per_sample * 8) - 1
            else:
                exprLower = float('-inf')
                exprUpper = float('inf')

            sQP = _quantization_parameters(sSType, depths, fulls, chroma)
            dQP = _quantization_parameters(dSType, depthd, fulld, chroma)

            gain = dQP['range'] / sQP['range']
            offset = dQP['neutral' if chroma else 'floor'] - sQP['neutral' if chroma else 'floor'] * gain

            if mode == 1:
                scale = 256
                gain = gain * scale
                offset = offset * scale
            else:
                scale = 1

            if gain != 1 or offset != 0 or clamp:
                expr = " x "
                if gain != 1: expr = expr + f" {gain} * "
                if offset != 0: expr = expr + f" {offset} + "
                if clamp:
                    if dQP['floor'] * scale > exprLower: expr = expr + f" {dQP['floor'] * scale} max "
                    if dQP['ceil'] * scale < exprUpper: expr = expr + f" {dQP['ceil'] * scale} min "
            else:
                expr = ""

            return expr

        # Process
        Yexpr = gen_expr(False, mode)
        Cexpr = gen_expr(True, mode)

        if sIsYUV:
            expr = [Yexpr, Cexpr]
        elif sIsGRAY and chroma:
            expr = Cexpr
        else:
            expr = Yexpr

        clip = Expr(clip, expr, format=dFormat.id)

        # Output
        clip=core.std.SetFrameProp(clip, prop='_ColorRange', intval=0 if fulld else 1)
        return clip


    if dither == 1:
        clip = _quantization_conversion(clip, sbitPS, depth, vs.INTEGER, fulls, fulld, False, False, 8, 0)
        clip = _quantization_conversion(clip, depth, 8, vs.INTEGER, fulld, fulld, False, False, 8, 0)
        return clip
    else:
        full = fulld
        clip = _quantization_conversion(clip, sbitPS, depth, vs.INTEGER, fulls, full, False, False, 16, 1)
        sSType = vs.INTEGER
        sbitPS = 16
        fulls = False
        fulld = False

    clip = core.fmtc.bitdepth(clip, bits=dbitPS, flt=dSType, fulls=fulls, fulld=fulld, dmode=dither)
    clip = core.std.SetFrameProp(clip, prop='_ColorRange', intval=0 if fulld else 1)

    clip = _quantization_conversion(clip, depth, 8, vs.INTEGER, full, full, False, False, 8, 0)
    return clip