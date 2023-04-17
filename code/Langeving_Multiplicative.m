(* ::Package:: *)

(* ::Section:: *)
(*Langeving Simulation Multiplicative Noise*)


(* ::Input:: *)
(*Clear[\[Xi],\[Alpha]]*)


(* ::Input:: *)
(*d=0.5;*)
(*h=0.000;*)
(*T=2000;*)
(*dt=0.01;*)
(*nstep=T/dt;*)
(*\[Xi]theor=4/d^2*)


(* ::Input:: *)
(*proc=ItoProcess[\[DifferentialD]x[t]==(d+1)*x[t]\[DifferentialD]t-x[t]^3 \[DifferentialD]t+h \[DifferentialD]t+Sqrt[2]*x[t]*\[DifferentialD]w[t],x[t],{x,1},t,w\[Distributed]WienerProcess[]];*)


(* ::Input:: *)
(*ito=RandomFunction[proc,{0.,T,dt}]*)


(* ::Input:: *)
(*ListPlot[ito["Path"], PlotRange->All]*)


(* ::Input:: *)
(*manyReal=RandomFunction[proc,{0.,T,dt},100];*)


(* ::Input:: *)
(*thermalized = manyReal["Part",All,{2*\[Xi]theor,T}];*)


(* ::Input:: *)
(*low=Ceiling[0.02*\[Xi]theor/dt]*)
(*high=Ceiling[1.4*\[Xi]theor/dt]*)
(*corr=CorrelationFunction[thermalized,{low, high}]*)
(*logcorr=TimeSeries[Log[corr["Values"]],{corr["Times"]}];*)


(* ::Input:: *)
(*ListLogPlot[corr]*)
(*ListLogLogPlot[corr]*)


(* ::Input:: *)
(**)


(* ::Input:: *)
(*lm = LinearModelFit[logcorr,x,x];*)
(*-dt/lm["BestFitParameters"][[2]]*)


(* ::Input:: *)
(*logExpPLM[x_,\[Alpha]_,\[Beta]_,\[Xi]_]:=\[Alpha] -x/\[Xi]-\[Beta]*Log[x]*)


(* ::Input:: *)
(*nlmPL=NonlinearModelFit[logcorr,logExpPLM[x,\[Alpha],\[Beta],\[Xi]],{\[Alpha],\[Beta],\[Xi]},x]*)
(*nlmPL["ParameterConfidenceIntervalTable"]*)


(* ::Input:: *)
(*Show[ListPlot[logcorr],Plot[lm[x],{x,low,high},PlotStyle->Red],Plot[nlmPL[x],{x,low,high},PlotStyle->Green],Frame->True,FrameLabel->{"n","acf[n]"}]*)


(* ::Input:: *)
(*CDFtheor[x_,d_] = (x^d (x^3)^(-d/3) (Gamma[d/3]-Gamma[d/3,x^3/2]))/Gamma[d/3];*)


(* ::Input:: *)
(*p1= DiscretePlot[CDF[thermalized["SliceDistribution",T],x],{x,0,2.3,0.01}];*)
(*p2= Plot[CDFtheor[x,0.5],{x,0,2.3}];*)
(*Show[p1,p2]*)
