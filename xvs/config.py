import vapoursynth as vs
core=vs.core

import functools

import nnedi3_resample as nnrs
nnedi3_resample=nnrs.nnedi3_resample
if hasattr(core,"znedi3") and "mode" in nnrs.nnedi3_resample.__code__.co_varnames:
        nnedi3_resample=functools.partial(nnrs.nnedi3_resample,mode="znedi3")

Expr=core.akarin.Expr if hasattr(core,"akarin") else core.std.Expr