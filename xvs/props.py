from .utils import *
from .metrics import SSIM,GMSD

def statsinfo2csv(clip,plane=None,Max=True,Min=True,Avg=False,bits=8,namebase=None):
    """
    write PlaneStats(Max,Min,Avg) to csv
    """

    cbits=clip.format.bits_per_sample
    cfamily=clip.format.color_family
    #########
    def info(clip,t,p):
        statsclip=core.std.PlaneStats(clip, plane=p)
        txt = open(t,'w')
        #############
        head="n"
        head+=",Max" if Max else ""
        head+=",Min" if Min else ""
        head+=",Avg" if Avg else ""
        head+="\n"
        txt.write(head)
        #############
        def write(n, f, clip, core,Max,Min,Avg,bits):
            ma = int(round(f.props.PlaneStatsMax*(1<<bits)/(1<<cbits)))
            mi = int(round(f.props.PlaneStatsMin*(1<<bits)/(1<<cbits)))
            avg=int(round(f.props.PlaneStatsAverage*(1<<bits)))
            ######
            line=str(n)
            line+=(","+str(ma)) if Max else ""
            line+=(","+str(mi)) if Min else ""
            line+=(","+str(avg)) if Avg else ""
            line+="\n"
            txt.write(line)
            ######
            return clip
            txt.close()
        last = core.std.FrameEval(clip,functools.partial(write, clip=clip,core=core,Max=Max,Min=Min,Avg=Avg,bits=bits),prop_src=statsclip)
        return last
    ###############
    if cfamily == vs.YUV:
        pname=('Y','U','V')
    elif cfamily == vs.RGB:
        pname=('R','G','B')
    elif cfamily == vs.GRAY:
        pname=('GRAY')
    else:
        raise ValueError("")
    ###############
    if plane is None:
        if cfamily == vs.GRAY:
            plane = [0]
        else:
            plane=[0,1,2]
    elif isinstance(plane,int):
        plane=[plane]
    elif not isinstance(plane,(list,tuple)):
        raise TypeError()
    ###############
    for i in plane:
        name= pname[i]+".csv" if namebase is None else namebase+'.'+pname[i]+".csv"
        clip = info(clip,name,i)
    return clip

def ssim2csv(clip1,clip2,file="ssim.csv",planes=None, downsample=True, k1=0.01, k2=0.03, fun=None, dynamic_range=1):
    """
    Warp function for ssim in muvsfunc
    Calculate SSIM and write to a csv
    Args:
        clip1: The distorted clip, will be copied to output.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        file:  output file name

        plane: (int/list) Specify which planes to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        k1, k2: (int) Constants in the SSIM index formula.
            According to the paper, the performance of the SSIM index algorithm is fairly insensitive to variations of these values.
            Default are 0.01 and 0.03.

        fun: (function or float) The function of how the clips are filtered.
            If it is None, it will be set to a gaussian filter whose standard deviation is 1.5. Note that the size of gaussian kernel is different from the one in MATLAB.
            If it is a float, it specifies the standard deviation of the gaussian filter. (sigma in core.tcanny.TCanny)
            According to the paper, the quality map calculated from gaussian filter exhibits a locally isotropic property,
            which prevents the present of undesirable “blocking” artifacts in the resulting SSIM index map.
            Default is None.

        dynamic_range: (float) Dynamic range of the internal float point clip. Default is 1.
    """
    isYUV=clip1.format.color_family==vs.YUV
    isRGB=clip1.format.color_family==vs.RGB
    isGRAY=clip1.format.color_family==vs.GRAY
    if isinstance(planes,int):
        planes=[planes]
    if isGRAY:
        clip=SSIM(clip1, clip2, plane=0, downsample=downsample, k1=k1, k2=k2, fun=fun, dynamic_range=dynamic_range, show_map=False)
        txt = open(file,'w')
        txt.write("n,gary\n")
        def tocsv(n, f, clip ,core):
            txt.write(str(n)+","+str(f.props.PlaneSSIM)+"\n")
            return clip
            txt.close()
        last=core.std.FrameEval(clip,functools.partial(tocsv, clip=clip,core=core),prop_src=clip)
    elif isYUV:
        if planes is None:
            planes=[0,1,2]
        Y=SSIM(getY(clip1),getY(clip2),plane=0, downsample=downsample, k1=k1, k2=k2, fun=fun, dynamic_range=dynamic_range, show_map=False) if 0 in planes else getY(clip1)
        U=SSIM(getU(clip1),getU(clip2),plane=0, downsample=downsample, k1=k1, k2=k2, fun=fun, dynamic_range=dynamic_range, show_map=False) if 1 in planes else getU(clip1)
        V=SSIM(getV(clip1),getV(clip2),plane=0, downsample=downsample, k1=k1, k2=k2, fun=fun, dynamic_range=dynamic_range, show_map=False) if 2 in planes else getV(clip1)
        txt = open(file,'w')
        head="n"
        head+=",Y" if 0 in planes else ""
        head+=",U" if 1 in planes else ""
        head+=",V" if 2 in planes else ""
        txt.write(head+"\n")
        def tocsv(n,f,clip,core):
            line=str(n)
            line+=(","+str(f[0].props.PlaneSSIM)) if 0 in planes else ""
            line+=(","+str(f[1].props.PlaneSSIM)) if 1 in planes else ""
            line+=(","+str(f[2].props.PlaneSSIM)) if 2 in planes else ""
            txt.write(line+"\n")
            return clip
            txt.close()
        last=core.std.FrameEval(clip1,functools.partial(tocsv, clip=clip1,core=core),prop_src=[Y,U,V])
    elif isRGB:
        if planes is None:
            planes=[0,1,2]
        R=SSIM(getY(clip1),getY(clip2),plane=0, downsample=downsample, k1=k1, k2=k2, fun=fun, dynamic_range=dynamic_range, show_map=False) if 0 in planes else getY(clip1)
        G=SSIM(getU(clip1),getU(clip2),plane=0, downsample=downsample, k1=k1, k2=k2, fun=fun, dynamic_range=dynamic_range, show_map=False) if 1 in planes else getU(clip1)
        B=SSIM(getV(clip1),getV(clip2),plane=0, downsample=downsample, k1=k1, k2=k2, fun=fun, dynamic_range=dynamic_range, show_map=False) if 2 in planes else getV(clip1)
        txt = open(file,'w')
        head="n"
        head+=",R" if 0 in planes else ""
        head+=",G" if 1 in planes else ""
        head+=",B" if 2 in planes else ""
        txt.write(head+"\n")
        def tocsv(n,f,clip,core):
            line=str(n)
            line+=(","+str(f[0].props.PlaneSSIM)) if 0 in planes else ""
            line+=(","+str(f[1].props.PlaneSSIM)) if 1 in planes else ""
            line+=(","+str(f[2].props.PlaneSSIM)) if 2 in planes else ""
            txt.write(line+"\n")
            return clip
            txt.close()
        last=core.std.FrameEval(clip1,functools.partial(tocsv, clip=clip1,core=core),prop_src=[R,G,B])
    else:
        raise TypeError("unsupport format")
    return last

def GMSD2csv(clip1,clip2,file="GMSD.csv",planes=None, downsample=True,c=0.0026):
    """
    Warp function for GMSD in muvsfunc
    Calculate GMSD and write to a csv

    GMSD is a new effective and efficient image quality assessment (IQA) model, which utilizes the pixel-wise gradient magnitude similarity (GMS)
    between the reference and distorted images combined with standard deviation of the GMS map to predict perceptual image quality.

    The distortion degree of the distorted image will be stored as frame property 'PlaneGMSD' in the output clip.

    The value of GMSD reflects the range of distortion severities in an image.
    The lowerer the GMSD score, the higher the image perceptual quality.
    If "clip1" == "clip2", GMSD = 0.
    
    Args:
        clip1: The distorted clip, will be copied to output.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        file:  output file name

        plane: (int/list) Specify which planes to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        c: (float) A positive constant that supplies numerical stability.
            According to the paper, for all the test databases, GMSD shows similar preference to the value of c.
            Default is 0.0026.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.
    """
    isYUV=clip1.format.color_family==vs.YUV
    isRGB=clip1.format.color_family==vs.RGB
    isGRAY=clip1.format.color_family==vs.GRAY
    if isinstance(planes,int):
        planes=[planes]
    if isGRAY:
        clip=GMSD(clip1, clip2, plane=0, downsample=downsample, c=c,show_map=False)
        txt = open(file,'w')
        txt.write("n,gary\n")
        def tocsv(n, f, clip ,core):
            txt.write(str(n)+","+str(f.props.PlaneGMSD)+"\n")
            return clip
            txt.close()
        last=core.std.FrameEval(clip,functools.partial(tocsv, clip=clip,core=core),prop_src=clip)
    elif isYUV:
        if planes is None:
            planes=[0,1,2]
        Y=GMSD(getY(clip1),getY(clip2),plane=0, downsample=downsample, c=c, show_map=False) if 0 in planes else getY(clip1)
        U=GMSD(getU(clip1),getU(clip2),plane=0, downsample=downsample, c=c, show_map=False) if 1 in planes else getU(clip1)
        V=GMSD(getV(clip1),getV(clip2),plane=0, downsample=downsample, c=c, show_map=False) if 2 in planes else getV(clip1)
        txt = open(file,'w')
        head="n"
        head+=",Y" if 0 in planes else ""
        head+=",U" if 1 in planes else ""
        head+=",V" if 2 in planes else ""
        txt.write(head+"\n")
        def tocsv(n,f,clip,core):
            line=str(n)
            line+=(","+str(f[0].props.PlaneGMSD)) if 0 in planes else ""
            line+=(","+str(f[1].props.PlaneGMSD)) if 1 in planes else ""
            line+=(","+str(f[2].props.PlaneGMSD)) if 2 in planes else ""
            txt.write(line+"\n")
            return clip
            txt.close()
        last=core.std.FrameEval(clip1,functools.partial(tocsv, clip=clip1,core=core),prop_src=[Y,U,V])
    elif isRGB:
        if planes is None:
            planes=[0,1,2]
        R=GMSD(getY(clip1),getY(clip2),plane=0, downsample=downsample, c=c, show_map=False) if 0 in planes else getY(clip1)
        G=GMSD(getU(clip1),getU(clip2),plane=0, downsample=downsample, c=c, show_map=False) if 1 in planes else getU(clip1)
        B=GMSD(getV(clip1),getV(clip2),plane=0, downsample=downsample, c=c, show_map=False) if 2 in planes else getV(clip1)
        txt = open(file,'w')
        head="n"
        head+=",R" if 0 in planes else ""
        head+=",G" if 1 in planes else ""
        head+=",B" if 2 in planes else ""
        txt.write(head+"\n")
        def tocsv(n,f,clip,core):
            line=str(n)
            line+=(","+str(f[0].props.PlaneGMSD)) if 0 in planes else ""
            line+=(","+str(f[1].props.PlaneGMSD)) if 1 in planes else ""
            line+=(","+str(f[2].props.PlaneGMSD)) if 2 in planes else ""
            txt.write(line+"\n")
            return clip
            txt.close()
        last=core.std.FrameEval(clip1,functools.partial(tocsv, clip=clip1,core=core),prop_src=[R,G,B])
    else:
        raise TypeError("unsupport format")
    return last

def csv2props(clip:vs.VideoNode,file:str,sep:str="\t",props:list[list]=None,rawfilter=None,strict=False,charset="utf-8"):
    """
    csv file must contain a column use "n" as title to log frame number,all values should be int >=0.
    props:
        should be a list contain a list with "title of values in csv","prop name you want set","fuction you want use to process raw value" ,"default when value not in csv" in order.
        If leave props unset,use all titles in csv.
        If leave "prop name you want set" None,it will same as "title of values in csv"
        If leave "fuction you want use to process raw value" None,will use raw value as an string.
        If leave "default when value not in csv" None,will use a empty string as default value.

        Notice:Default value only for frame number not in csv,will not affect line with missing value and sep.
    rawfilter:
        overwrite fuction use to process raw value when "fuction you want use to process raw value" unset.
    strict:
        If True, will throw an exception instead of using default value when frame number not in csv. And also force "n column" only contain frame number in clip strictly. 
        If False,it also ignore same frame number in csv and use the last one.(maybe change in future)
    """
    rawfilter=rawfilter if (rawfilter is not None and callable(rawfilter)) else lambda x:x
    with open(file,encoding=charset) as file:
        lines=[line.split(sep) for line in file.read().split("\n") if len(line.split(sep))>1]
    
    if "n" not in lines[0]:
        raise ValueError("""csv file must contain a column use "n" as title to log frame number,all values should be int >=0.""")
    
    if len(set(lines[0])) != len(lines[0]):
        raise ValueError("csv should not have columns with same title")

    titles=lines[0][:]
    titles.remove("n")
    nindex=lines[0].index("n")

    if strict:
        nlist=sorted([int(line[nindex]) for line in lines[1:]])
        if nlist!=list(range(len(clip))):
            raise ValueError(""" "n column" should only contain frame number in clip strictly. """)
    datas={}
    for line in lines[1:]:
        if len(line)<len(lines[0]):
            raise ValueError("missing value in csv")

        tline=line[:]
        del tline[nindex]
        datas[int(line[nindex])]=dict(zip(titles,tline))

    if props is None:
        props=[[titles[i],titles[i],rawfilter,None] for i in range(len(titles))]
    else:
        tmp=[]
        for prop in props:
            tp=[
                None,
                prop+[None,None,None],
                prop+[None,None],
                prop+[None],
                prop
            ][len(prop)]
            if tp is None or tp[0] is None or tp[0] not in titles:
                continue
            if tp[1] is None:
                tp[1]=tp[0]
            if tp[2] is None:
                tp[2]=rawfilter
            if tp[3] is None:
                tp[3]=''
            tmp.append(tp)
        props=tmp

    pdatas={}
    for i in range(len(clip)):
        data=datas.get(i)
        if data is None:
            pdatas[i]=dict(zip([p[1] for p in props],[p[3] for p in props]))
        else:
            pdatas[i]=dict(zip([p[1] for p in props],[p[2](data[p[0]]) for p in props]))

    def attach(n,f): 
        fout=f.copy()
        for k,v in pdatas[n].items():
            fout.props[k]=v
        return fout

    last=core.std.ModifyFrame(clip,clip,attach)
    return last

def props2csv(clip:vs.VideoNode,props:list,titles:list,output="info.csv",sep="\t",charset="utf-8",tostring=None):
    """
    write props which you chosen to csv
    you can rewrite tostring function to process props before write to csv

    props: A list contain the name of props you want to write to csv
    titles:A list contain titles of props
    output:path of output file,default is info.csv
    sep:the separator you want use,default is tab
    charset:the charset of csv file you want use,default is utf-8
    tostring:should be a function to process props before write to csv
    """
    file=open(output,"w",encoding=charset)
    file.write(sep.join(["n"]+titles))
    
    tostring=tostring if callable(tostring) else lambda x: x.decode("utf-8") if isinstance(x,bytes) else str(x)


    def tocsv(n,f,clip):
        file.write("\n"+sep.join([str(n)]+[tostring(eval("f.props."+i,globals(),{'f':f})) for i in props]))
        
        return clip
        file.close()
    return core.std.FrameEval(clip, functools.partial(tocsv, clip=clip),prop_src=clip)

def getPictType(clip,txt=None,show=True):
    """
    getPictType
    """
    sclip=core.std.PlaneStats(clip, plane=0)
    log = txt is not None
    if log:
        t = open(txt,'w')
    def __type(n, f, clip, core):
        ptype = str(f.props._PictType)[2]
        if log:
            t.write(str(n)+","+ptype)
            t.write('\n')
        if show:
            return core.text.Text(clip, "PictType:"+ptype)
        else:
            return clip
        if log:
            t.close()
    last = core.std.FrameEval(clip, functools.partial(__type, clip=clip,core=core),prop_src=sclip)
    return last
