"use strict";

// ========== UTILITIES ==========
function makeRNG(seed){var s=seed|0;return function(){s=(s+0x6D2B79F5)|0;var t=Math.imul(s^(s>>>15),1|s);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296}}
function makeGaussian(rng){return function(){var u1=0,u2=0;while(u1===0)u1=rng();while(u2===0)u2=rng();return Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2)}}
function fmt(v,d){if(v==null||isNaN(v))return"--";return v.toFixed(d===undefined?3:d)}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v))}
function qnorm(p){
if(p<=0)return-6;if(p>=1)return 6;
var a=[-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,1.383577518672690e2,-3.066479806614716e1,2.506628277459239];
var b=[-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,6.680131188771972e1,-1.328068155288572e1];
var c=[-7.784894002430293e-3,-3.223964580411365e-1,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783];
var d=[7.784695709041462e-3,3.224671290700398e-1,2.445134137142996,3.754408661907416];
var pl=.02425,ph=1-pl,q,r;
if(p<pl){q=Math.sqrt(-2*Math.log(p));return(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)}
if(p<=ph){q=p-.5;r=q*q;return(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)}
q=Math.sqrt(-2*Math.log(1-p));return-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)}
function theoreticalQuantiles(n){var p=[],i;for(i=1;i<=n;i++)p.push((i-.375)/(n+.25));return p.map(qnorm)}

// ========== STATISTICS ==========
function computeOLS(xs,ys){
var n=xs.length;if(n<2)return null;
var sX=0,sY=0,i;
for(i=0;i<n;i++){sX+=xs[i];sY+=ys[i]}
var xB=sX/n,yB=sY/n,sxx=0,sxy=0,syy=0,dx,dy;
for(i=0;i<n;i++){dx=xs[i]-xB;dy=ys[i]-yB;sxx+=dx*dx;sxy+=dx*dy;syy+=dy*dy}
if(Math.abs(sxx)<1e-15)return null;
var slope=sxy/sxx,intercept=yB-slope*xB,sse=0;
for(i=0;i<n;i++){var e=ys[i]-intercept-slope*xs[i];sse+=e*e}
var sst=syy,r2=sst>1e-15?1-sse/sst:null;
var mse=n>2?sse/(n-2):0,rmse=Math.sqrt(mse);
var adjR2=(r2!==null&&n>2)?1-(1-r2)*(n-1)/(n-2):null;
return{slope:slope,intercept:intercept,r2:r2,adjR2:adjR2,mse:mse,rmse:rmse,sse:sse,sst:sst,n:n,xBar:xB,yBar:yB,sxx:sxx,sxy:sxy}}

function computeDiags(xs,ys,fit){
var n=xs.length,slope=fit.slope,intercept=fit.intercept,mse=fit.mse,xBar=fit.xBar,sxx=fit.sxx;
return xs.map(function(x,i){
var fitted=intercept+slope*x,residual=ys[i]-fitted;
var leverage=1/n+(x-xBar)*(x-xBar)/sxx;
var stdRes=0,cookD=0;
if(mse>1e-15&&leverage<1-1e-10){
stdRes=residual/(Math.sqrt(mse)*Math.sqrt(1-leverage));
cookD=(residual*residual/(2*mse))*(leverage/((1-leverage)*(1-leverage)));
}
return{x:x,y:ys[i],fitted:fitted,residual:residual,leverage:leverage,stdRes:stdRes,cookD:cookD};
})}

function huberRegression(xs,ys,c,maxIter){
c=c||1.345;maxIter=maxIter||50;
var n=xs.length;if(n<2)return null;
var fit=computeOLS(xs,ys);if(!fit)return null;
var slope=fit.slope,intercept=fit.intercept,iter,i;
for(iter=0;iter<maxIter;iter++){
var res=xs.map(function(x,i){return ys[i]-intercept-slope*x});
var absRes=res.map(Math.abs).sort(function(a,b){return a-b});
var mad=Math.max(absRes[Math.floor(n/2)]||1e-10,1e-10)/0.6745;
var weights=res.map(function(r){var u=Math.abs(r)/mad;return u<=c?1:c/u});
var sw=0,swx=0,swy=0,swxx=0,swxy=0,w;
for(i=0;i<n;i++){w=weights[i];sw+=w;swx+=w*xs[i];swy+=w*ys[i];swxx+=w*xs[i]*xs[i];swxy+=w*xs[i]*ys[i]}
var den=sw*swxx-swx*swx;if(Math.abs(den)<1e-15)break;
var newSlope=(sw*swxy-swx*swy)/den,newInt=(swy-newSlope*swx)/sw;
var diff=Math.abs(newSlope-slope)+Math.abs(newInt-intercept);
slope=newSlope;intercept=newInt;
if(diff<1e-10)break;
}
var sse=0;for(i=0;i<n;i++){sse+=(ys[i]-intercept-slope*xs[i])*(ys[i]-intercept-slope*xs[i])}
return{slope:slope,intercept:intercept,sse:sse,mse:sse/Math.max(n-2,1),n:n,method:"Huber"}}

function weightedOLS(xs,ys,weights){
var n=xs.length;if(n<2)return null;
var sw=0,swx=0,swy=0,swxx=0,swxy=0,i,w;
for(i=0;i<n;i++){w=weights[i];sw+=w;swx+=w*xs[i];swy+=w*ys[i];swxx+=w*xs[i]*xs[i];swxy+=w*xs[i]*ys[i]}
var den=sw*swxx-swx*swx;if(Math.abs(den)<1e-15)return null;
var slope=(sw*swxy-swx*swy)/den,intercept=(swy-slope*swx)/sw;
var wsse=0;for(i=0;i<n;i++){var e=ys[i]-intercept-slope*xs[i];wsse+=weights[i]*e*e}
return{slope:slope,intercept:intercept,weightedSse:wsse,n:n,method:"WLS"}}

// ========== DATA GENERATION ==========
function generateData(type,params){
var n=params.n||35,noise=params.noise||1,seed=params.seed||42;
var rng=makeRNG(seed),g=makeGaussian(rng),data=[],i,x,y;
var nonlin=params.nonlin!==undefined?params.nonlin:1;
var hetero=params.hetero!==undefined?params.hetero:1;
var outMag=params.outMag!==undefined?params.outMag:3;
for(i=0;i<n;i++){
x=2+rng()*28;
switch(type){
case"nonlinear":y=30+2.5*x+nonlin*0.03*x*x+g()*8*noise;break;
case"hetero":y=30+2.5*x+g()*(5+hetero*0.3*x)*noise;break;
case"outlier":y=30+2.5*x+g()*8*noise;if(i===Math.floor(n/2))y+=outMag*25;break;
case"leverage":y=30+2.5*x+g()*8*noise;if(i===n-1){x=48;y=30+2.5*48+g()*2*noise}break;
case"influential":y=30+2.5*x+g()*8*noise;if(i===n-1){x=48;y=30+2.5*48-outMag*40}break;
default:y=30+2.5*x+g()*8*noise;
}
data.push({id:i,x:Math.round(x*100)/100,y:Math.round(y*100)/100,excluded:false});
}
return data}

// ========== STATE ==========
var S={data:[],sel:-1,tab:0,showComp:false,cm:false,fit:null,fitNo:null,diags:[],
robustFit:null,logFit:null,wlsFit:null,exp:"A",dragIdx:-1,dragging:false};

function recalc(){
var ai=[],xs=[],ys=[],i;
for(i=0;i<S.data.length;i++){if(!S.data[i].excluded){ai.push(i);xs.push(S.data[i].x);ys.push(S.data[i].y)}}
S.fit=computeOLS(xs,ys);
if(S.fit){
var ad=computeDiags(xs,ys,S.fit);
S.diags=[];
for(i=0;i<S.data.length;i++){
var pos=ai.indexOf(i);
if(pos>=0){
var d=ad[pos];
S.diags.push({id:i,x:d.x,y:d.y,fitted:d.fitted,residual:d.residual,leverage:d.leverage,stdRes:d.stdRes,cookD:d.cookD});
}else{
S.diags.push({id:i,x:S.data[i].x,y:S.data[i].y,fitted:null,residual:null,leverage:null,stdRes:null,cookD:null});
}}
}else{
S.diags=S.data.map(function(d,i){return{id:i,x:d.x,y:d.y,fitted:null,residual:null,leverage:null,stdRes:null,cookD:null}});
}
// Leave-one-out
if(S.sel>=0&&S.sel<S.data.length&&!S.data[S.sel].excluded&&ai.length>2){
var wxs=[],wys=[],j;
for(j=0;j<ai.length;j++){if(ai[j]!==S.sel){wxs.push(xs[j]);wys.push(ys[j])}}
S.fitNo=computeOLS(wxs,wys);
}else{S.fitNo=null}
// Robust
S.robustFit=S.fit?huberRegression(xs,ys):null;
// Log
var allPos=ys.length>0;
for(i=0;i<ys.length;i++){if(ys[i]<=0){allPos=false;break}}
S.logFit=(S.fit&&allPos)?computeOLS(xs,ys.map(Math.log)):null;
// WLS
if(S.fit){var wts=xs.map(function(x){return x>0.5?1/(x*x):1});S.wlsFit=weightedOLS(xs,ys,wts)}else{S.wlsFit=null}
}

// ========== CHART HELPERS ==========
var PC={displayModeBar:true,modeBarButtonsToRemove:["lasso2d","select2d","autoScale2d","toggleSpikelines"],displaylogo:false,responsive:true,scrollZoom:true};

function mkLayout(xt,yt,extra){
var L={font:{family:"Inter,sans-serif",size:11,color:"#475569"},paper_bgcolor:"#fff",plot_bgcolor:"#fff",
margin:{l:55,r:15,t:10,b:50},showlegend:false,
hoverlabel:{bgcolor:"#1a2332",font:{color:"#fff",size:11}}};
L.xaxis={title:xt||"",gridcolor:"#f1f5f9",zerolinecolor:"#e2e8f0",titlefont:{size:11,color:"#1a2332"},tickfont:{size:10}};
L.yaxis={title:yt||"",gridcolor:"#f1f5f9",zerolinecolor:"#e2e8f0",titlefont:{size:11,color:"#1a2332"},tickfont:{size:10}};
if(extra){for(var k in extra){if(k==="xaxis"||k==="yaxis"){for(var ak in extra[k])L[k][ak]=extra[k][ak]}else{L[k]=extra[k]}}}
return L}

function getColors(){var si=S.sel;return S.data.map(function(d,i){if(i===si)return"#ea580c";if(d.excluded)return"#cbd5e1";return"#2563eb"})}
function getSizes(){var si=S.sel;return S.data.map(function(d,i){return i===si?12:7})}

function renderScatter(divId,opts){
opts=opts||{};
var el=document.getElementById(divId);if(!el)return;
var colors=getColors(),sizes=getSizes();
var traces=[];
traces.push({type:"scatter",mode:"markers",
x:S.data.map(function(d){return d.x}),y:S.data.map(function(d){return d.y}),
marker:{color:colors,size:sizes,line:{color:"white",width:1.5}},
customdata:S.diags.map(function(d){return[d.id,d.residual||0,d.leverage||0,d.cookD||0]}),
hovertemplate:"<b>Point %{customdata[0]}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Residual: %{customdata[1]:.3f}<br>Leverage: %{customdata[2]:.4f}<br>Cook D: %{customdata[3]:.4f}<extra></extra>"});
var act=S.data.filter(function(d){return!d.excluded});
if(act.length>=2&&S.fit){
var axs=act.map(function(d){return d.x});
var xR=[Math.min.apply(null,axs),Math.max.apply(null,axs)];
traces.push({type:"scatter",mode:"lines",x:xR,
y:[S.fit.intercept+S.fit.slope*xR[0],S.fit.intercept+S.fit.slope*xR[1]],
line:{color:"#2563eb",width:2.5},name:"OLS",hoverinfo:"skip",showlegend:true});
if(S.fitNo&&(opts.showComp||S.showComp)){
traces.push({type:"scatter",mode:"lines",x:xR,
y:[S.fitNo.intercept+S.fitNo.slope*xR[0],S.fitNo.intercept+S.fitNo.slope*xR[1]],
line:{color:"#ea580c",width:2,dash:"dash"},name:"Without Selected",hoverinfo:"skip",showlegend:true});
}
if(opts.extraLines){opts.extraLines.forEach(function(l){
traces.push({type:"scatter",mode:"lines",x:l.x,y:l.y,
line:{color:l.color,width:l.width||2,dash:l.dash||"solid"},name:l.name,hoverinfo:"skip",showlegend:true})})}
}
var layout=mkLayout(opts.xl||"Advertising Expenditure ($1000s)",opts.yl||"Monthly Sales ($1000s)",
{height:opts.h||440,dragmode:false,
legend:{x:0.01,y:0.99,bgcolor:"rgba(255,255,255,.85)",bordercolor:"#e2e8f0",borderwidth:1,font:{size:10}}});
try{Plotly.react(el,traces,layout,PC)}catch(e){console.error("Plot error:",e)}
}

function renderDiag(){
var dd=S.diags.filter(function(d){return d.residual!==null});
if(!dd.length||!S.fit)return;
var si=S.sel;
var colors=dd.map(function(d){return d.id===si?"#ea580c":"#2563eb"});
var sizes=dd.map(function(d){return d.id===si?11:7});
var ids=dd.map(function(d){return d.id});
var fMin=Math.min.apply(null,dd.map(function(d){return d.fitted}))-5;
var fMax=Math.max.apply(null,dd.map(function(d){return d.fitted}))+5;
try{
// Plot 1: Residuals vs Fitted
Plotly.react("d1",[
{type:"scatter",mode:"markers",x:dd.map(function(d){return d.fitted}),y:dd.map(function(d){return d.residual}),
marker:{color:colors,size:sizes,line:{color:"white",width:1}}},
{type:"scatter",mode:"lines",x:[fMin,fMax],y:[0,0],line:{color:"#94a3b8",width:1,dash:"dash"},hoverinfo:"skip"}
],mkLayout("Fitted Values","Residuals"),PC);

// Plot 2: QQ
var sorted=dd.map(function(d){return d.stdRes}).sort(function(a,b){return a-b});
var tq=theoreticalQuantiles(sorted.length);
var qMin=Math.min(Math.min.apply(null,tq)-.5,Math.min.apply(null,sorted)-.5);
var qMax=Math.max(Math.max.apply(null,tq)+.5,Math.max.apply(null,sorted)+.5);
Plotly.react("d2",[
{type:"scatter",mode:"markers",x:tq,y:sorted,
marker:{color:"#2563eb",size:7,line:{color:"white",width:1}}},
{type:"scatter",mode:"lines",x:[qMin,qMax],y:[qMin,qMax],line:{color:"#94a3b8",width:1.5,dash:"dash"},hoverinfo:"skip"}
],mkLayout("Theoretical Quantiles","Standardized Residuals"),PC);

// Plot 3: Residuals vs Leverage
var maxLev=Math.max.apply(null,dd.map(function(d){return d.leverage}));
var levR=[0,Math.max(.2,maxLev*1.3)];
var srMin=Math.min(-4,Math.min.apply(null,dd.map(function(d){return d.stdRes}))-1);
var srMax=Math.max(4,Math.max.apply(null,dd.map(function(d){return d.stdRes}))+1);
var contours=[.5,1].map(function(cl){
var cx=[],cy=[],h,r;
for(h=.001;h<levR[1];h+=.005){r=Math.sqrt(cl*2*S.fit.mse*(1-h)*(1-h)/h);
if(r<srMax*3){cx.push(h);cy.push(r);cx.push(h);cy.push(-r)}}
return{type:"scatter",mode:"lines",x:cx,y:cy,line:{color:"#e2e8f0",width:1,dash:"dot"},hoverinfo:"skip"}
});
Plotly.react("d3",
[{type:"scatter",mode:"markers",x:dd.map(function(d){return d.leverage}),y:dd.map(function(d){return d.stdRes}),
marker:{color:colors,size:sizes,line:{color:"white",width:1}}}].concat(contours),
mkLayout("Leverage","Standardized Residuals",{xaxis:{title:"Leverage",range:levR},yaxis:{title:"Standardized Residuals",range:[srMin,srMax]}}),PC);

// Plot 4: Cook's Distance
var cdMax=Math.max.apply(null,dd.map(function(d){return d.cookD}));
var cdRef=4/dd.length;
Plotly.react("d4",[
{type:"bar",x:ids,y:dd.map(function(d){return d.cookD}),
marker:{color:dd.map(function(d){return d.cookD>cdRef?"#ea580c":"#2563eb"})},
hovertemplate:"Point %{x}<br>Cook D: %{y:.4f}<extra></extra>"},
{type:"scatter",mode:"lines",x:[ids[0]-1,ids[ids.length-1]+1],y:[cdRef,cdRef],
line:{color:"#dc2626",width:1.5,dash:"dash"},name:"4/n",showlegend:true,hoverinfo:"skip"}
],mkLayout("Observation ID","Cook's Distance",
{yaxis:{title:"Cook's Distance",range:[0,Math.max(cdMax*1.1,cdRef*2)]},
legend:{x:.99,y:.99,xanchor:"right",font:{size:10}}}),PC);

// Plot 5: Standardized Residuals
Plotly.react("d5",[
{type:"bar",x:ids,y:dd.map(function(d){return d.stdRes}),
marker:{color:dd.map(function(d){return Math.abs(d.stdRes)>2?"#ea580c":"#2563eb"})},
hovertemplate:"Point %{x}<br>Std Residual: %{y:.3f}<extra></extra>"},
{type:"scatter",mode:"lines",x:[ids[0]-1,ids[ids.length-1]+1],y:[2,2],line:{color:"#dc2626",width:1,dash:"dash"},hoverinfo:"skip"},
{type:"scatter",mode:"lines",x:[ids[0]-1,ids[ids.length-1]+1],y:[-2,-2],line:{color:"#dc2626",width:1,dash:"dash"},hoverinfo:"skip"},
{type:"scatter",mode:"lines",x:[ids[0]-1,ids[ids.length-1]+1],y:[3,3],line:{color:"#dc2626",width:1.5},hoverinfo:"skip"},
{type:"scatter",mode:"lines",x:[ids[0]-1,ids[ids.length-1]+1],y:[-3,-3],line:{color:"#dc2626",width:1.5},hoverinfo:"skip"}
],mkLayout("Observation ID","Standardized Residual",
{yaxis:{title:"Standardized Residual",range:[Math.min(-4,srMin),Math.max(4,srMax)]}}),PC);
}catch(e){console.error("Diag plot error:",e)}
}

// ========== UI UPDATES ==========
function updateStats(spId,eqId){
var fit=S.fit,si=S.sel,sd=si>=0?S.diags[si]:null,h="";
h+='<div class="sg"><h3>Regression Model</h3>';
if(fit){
h+='<div class="sr"><span class="sl">Slope</span><span class="sv">'+fmt(fit.slope)+'</span></div>';
h+='<div class="sr"><span class="sl">Intercept</span><span class="sv">'+fmt(fit.intercept)+'</span></div>';
h+='<div class="sr"><span class="sl">R-squared</span><span class="sv">'+fmt(fit.r2)+'</span></div>';
h+='<div class="sr"><span class="sl">Adj. R-squared</span><span class="sv">'+fmt(fit.adjR2)+'</span></div>';
h+='<div class="sr"><span class="sl">MSE</span><span class="sv">'+fmt(fit.mse)+'</span></div>';
h+='<div class="sr"><span class="sl">RMSE</span><span class="sv">'+fmt(fit.rmse)+'</span></div>';
h+='<div class="sr"><span class="sl">SSE</span><span class="sv">'+fmt(fit.sse)+'</span></div>';
h+='<div class="sr"><span class="sl">n</span><span class="sv">'+fit.n+'</span></div>';
}else{h+='<div class="sr"><span class="sl">Unable to fit model</span></div>'}
h+='</div>';
if(sd&&sd.residual!==null){
h+='<div class="sg"><h3>Selected Observation</h3>';
h+='<div class="sr"><span class="sl">Point ID</span><span class="sv">'+sd.id+'</span></div>';
h+='<div class="sr"><span class="sl">X value</span><span class="sv">'+fmt(sd.x)+'</span></div>';
h+='<div class="sr"><span class="sl">Y value</span><span class="sv">'+fmt(sd.y)+'</span></div>';
h+='<div class="sr"><span class="sl">Fitted</span><span class="sv">'+fmt(sd.fitted)+'</span></div>';
var rc=Math.abs(sd.residual)>2*S.fit.rmse?"wa":"";
h+='<div class="sr"><span class="sl">Residual</span><span class="sv '+rc+'">'+fmt(sd.residual)+'</span></div>';
var lc=sd.leverage>4/fit.n?"hi":"";
h+='<div class="sr"><span class="sl">Leverage</span><span class="sv '+lc+'">'+fmt(sd.leverage,4)+'</span></div>';
h+='<div class="sr"><span class="sl">Std. Residual</span><span class="sv '+(Math.abs(sd.stdRes)>2?"hi":"")+'">'+fmt(sd.stdRes)+'</span></div>';
var dc=sd.cookD>4/fit.n?"wa":"";
h+='<div class="sr"><span class="sl">Cook\'s D</span><span class="sv '+dc+'">'+fmt(sd.cookD,4)+'</span></div>';
if(S.fitNo){
h+='<div class="sr"><span class="sl">LOO slope</span><span class="sv">'+fmt(S.fitNo.slope)+'</span></div>';
h+='<div class="sr"><span class="sl">Slope change</span><span class="sv">'+fmt(fit.slope-S.fitNo.slope)+'</span></div>';
}
h+='</div>'}
if(fit){
h+='<div class="sg"><h3>Screening Thresholds</h3>';
h+='<div class="sr"><span class="sl">2k/n (leverage)</span><span class="sv">'+fmt(4/fit.n,4)+'</span></div>';
h+='<div class="sr"><span class="sl">4/n (Cook D)</span><span class="sv">'+fmt(4/fit.n,4)+'</span></div>';
h+='<div class="sr"><span class="sl">Note</span><span class="sv" style="font-family:var(--fo);font-size:10px;color:var(--mu)">Heuristics only</span></div>';
h+='</div>'}
var el=document.getElementById(spId);if(el)el.innerHTML=h;
if(fit&&eqId){
var eq="y = "+fmt(fit.intercept,2);
eq+=fit.slope>=0?" + "+fmt(fit.slope,2)+" x":" - "+fmt(Math.abs(fit.slope),2)+" x";
var eEl=document.getElementById(eqId);
if(eEl){
eEl.innerHTML='<div class="eq-t">'+eq+'</div>'+
'<div class="eq-s"><span class="l">R-squared</span><span class="v">'+fmt(fit.r2)+'</span></div>'+
'<div class="eq-s"><span class="l">MSE</span><span class="v">'+fmt(fit.mse)+'</span></div>'+
'<div class="eq-s"><span class="l">n</span><span class="v">'+fit.n+'</span></div>'+
'<div class="eq-s"><span class="sl">RMSE</span><span class="v">'+fmt(fit.rmse)+'</span></div>';
}}}

function getExplanation(){
var parts=[],si=S.sel,sd=si>=0?S.diags[si]:null,fit=S.fit;
if(!fit){parts.push({t:"Unable to fit a regression model. Need at least 2 observations with different X values.",c:"warn"});return parts}
var n=fit.n;
if(sd&&sd.residual!==null){
if(sd.leverage>4/n){parts.push({t:"Observation "+si+" has leverage h="+fmt(sd.leverage,4)+", exceeding 2k/n="+fmt(4/n,4)+". High leverage means its X value is far from the sample mean, giving it potential to affect the fitted line. Influence also depends on the residual.",c:"acc"})}
if(Math.abs(sd.stdRes)>3){parts.push({t:"Observation "+si+" has standardized residual "+fmt(sd.stdRes,3)+", beyond the common screening range of +3/-3.",c:"warn"})}
else if(Math.abs(sd.stdRes)>2){parts.push({t:"Observation "+si+" has standardized residual "+fmt(sd.stdRes,3)+", which is notable. Values beyond +2/-2 deserve attention.",c:"info"})}
if(sd.cookD>4/n*2){parts.push({t:"Observation "+si+" has Cook's Distance D="+fmt(sd.cookD,4)+", substantially above 4/n="+fmt(4/n,4)+". This observation has significant influence and deserves further investigation. This is a heuristic, not an automatic reason to delete data.",c:"warn"})}
else if(sd.cookD>4/n){parts.push({t:"Observation "+si+" has Cook's Distance D="+fmt(sd.cookD,4)+", above the 4/n="+fmt(4/n,4)+" screening threshold.",c:"info"})}
if(S.fitNo){
var sd_=Math.abs(fit.slope-S.fitNo.slope);
var sp=S.fitNo.slope!==0?Math.abs(sd_/Math.abs(S.fitNo.slope)):null;
if(sp&&sp>0.02){parts.push({t:"Removing observation "+si+" changes slope from "+fmt(fit.slope,3)+" to "+fmt(S.fitNo.slope,3)+" ("+fmt(sp*100,1)+"% change). Intercept: "+fmt(fit.intercept,3)+" vs "+fmt(S.fitNo.intercept,3)+".",c:"info"})}}
if(sd.leverage>4/n&&Math.abs(sd.stdRes)>2){parts.push({t:"Observation "+si+" has both high leverage and a large residual. Check whether it is a data entry error, a rare but valid case, or an indication the model is misspecified.",c:"warn"})}
if(sd.leverage>4/n&&Math.abs(sd.stdRes)<=1){parts.push({t:"Observation "+si+" has high leverage but a small residual. It lies close to the regression trend but far from the mean of X. If it moved slightly, it could strongly affect the slope.",c:"info"})}
if(sd.leverage<=4/n&&Math.abs(sd.stdRes)>3){parts.push({t:"Observation "+si+" is a response outlier with ordinary leverage. Its X is typical but its Y is unusual. It has a large residual but less potential to change the slope than a high-leverage point.",c:"acc"})}
}
if(parts.length===0){parts.push({t:"The model appears reasonable. Drag any observation to see how it affects regression. Move a point vertically to create an outlier, or horizontally to change leverage.",c:"info"})}
return parts}

function updateExplanation(divId,extra){
var el=document.getElementById(divId);if(!el)return;
var parts=getExplanation();
var h=extra?extra:"";
h+=parts.map(function(p){return'<div class="ei '+p.c+'">'+p.t+'</div>'}).join("");
el.innerHTML=h}

function updateSidebar(){
var si=S.sel;
if(si>=0&&si<S.data.length){
document.getElementById("nosel").classList.add("hid");
document.getElementById("selctrl").classList.remove("hid");
document.getElementById("selid").textContent=si;
document.getElementById("csx").value=S.data[si].x;
document.getElementById("csy").value=S.data[si].y;
document.getElementById("lsx").textContent=S.data[si].x.toFixed(1);
document.getElementById("lsy").textContent=S.data[si].y.toFixed(1);
document.getElementById("cex").checked=S.data[si].excluded;
}else{
document.getElementById("nosel").classList.remove("hid");
document.getElementById("selctrl").classList.add("hid");
}}

function updatePlayground(){
renderScatter("s1",{showComp:S.showComp,h:440});
updateStats("sp1","eq1");
updateExplanation("ex1");
updateSidebar();
updateDiagIfVisible();
}

function updateOutlier(){
renderScatter("s2",{showComp:S.showComp,h:440});
updateStats("sp2","eq2");
var extra="";
if(S.exp==="A")extra='<div class="ei info"><strong>Experiment A -- Ordinary Observation:</strong> The selected point is near the regression line within the normal range of X. It has ordinary leverage and a small residual. Cook\'s Distance should be low.</div>';
if(S.exp==="B")extra='<div class="ei warn"><strong>Experiment B -- Response Outlier:</strong> The observation has an ordinary X value but a large Y value. It has a large residual and moderate leverage. It pulls the regression line toward it.</div>';
if(S.exp==="C")extra='<div class="ei acc"><strong>Experiment C -- High Leverage, Low Residual:</strong> The observation is far from the mean of X but close to the regression trend. It has high leverage but a small residual. High leverage alone does not make a point influential.</div>';
if(S.exp==="D")extra='<div class="ei warn"><strong>Experiment D -- High Leverage and High Influence:</strong> The observation is far from the mean of X and far from the trend. It combines high leverage with a large residual, giving it maximum influence.</div>';
if(S.exp==="E")extra='<div class="ei info"><strong>Experiment E -- Delete and Compare:</strong> The selected observation is excluded. Compare both regression lines. A large slope change confirms the observation was influential. Deletion is a sensitivity analysis, not an automatic recommendation.</div>';
updateExplanation("ex2",extra);
updateSidebar();
updateDiagIfVisible();
}

function updateDiagIfVisible(){
if(S.tab===2)renderDiag();
}

function updateAssumptions(){
var type=document.getElementById("dstype").value;
var params={n:+document.getElementById("cn4").value,noise:+document.getElementById("cno4").value,
seed:+document.getElementById("cs4").value,
nonlin:+document.getElementById("cnl4").value,
hetero:+document.getElementById("cht4").value,
outMag:+document.getElementById("com4").value};
var data=generateData(type,params);
var xs=data.map(function(d){return d.x}),ys=data.map(function(d){return d.y});
var fit=computeOLS(xs,ys);if(!fit)return;
var diags=computeDiags(xs,ys,fit);
try{
var acts=data.map(function(){return"#2563eb"});
Plotly.react("s4",[
{type:"scatter",mode:"markers",x:xs,y:ys,marker:{color:acts,size:7,line:{color:"white",width:1}},
hovertemplate:"X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"},
{type:"scatter",mode:"lines",x:[Math.min.apply(null,xs)-2,Math.max.apply(null,xs)+2],
y:[fit.intercept+fit.slope*(Math.min.apply(null,xs)-2),fit.intercept+fit.slope*(Math.max.apply(null,xs)+2)],
line:{color:"#2563eb",width:2.5},name:"OLS",showlegend:true,hoverinfo:"skip"}
],mkLayout("X","Y",{height:380,dragmode:false,legend:{x:.01,y:.99,font:{size:10}}}),PC);

var fMin=Math.min.apply(null,diags.map(function(d){return d.fitted}))-5;
var fMax=Math.max.apply(null,diags.map(function(d){return d.fitted}))+5;
Plotly.react("d14",[
{type:"scatter",mode:"markers",x:diags.map(function(d){return d.fitted}),y:diags.map(function(d){return d.residual}),
marker:{color:"#2563eb",size:6,line:{color:"white",width:1}}},
{type:"scatter",mode:"lines",x:[fMin,fMax],y:[0,0],line:{color:"#94a3b8",width:1,dash:"dash"},hoverinfo:"skip"}
],mkLayout("Fitted","Residuals"),PC);

var sorted=diags.map(function(d){return d.stdRes}).sort(function(a,b){return a-b});
var tq=theoreticalQuantiles(sorted.length);
var qMin=Math.min(Math.min.apply(null,tq)-.5,Math.min.apply(null,sorted)-.5);
var qMax=Math.max(Math.max.apply(null,tq)+.5,Math.max.apply(null,sorted)+.5);
Plotly.react("d24",[
{type:"scatter",mode:"markers",x:tq,y:sorted,marker:{color:"#2563eb",size:6,line:{color:"white",width:1}}},
{type:"scatter",mode:"lines",x:[qMin,qMax],y:[qMin,qMax],line:{color:"#94a3b8",width:1.5,dash:"dash"},hoverinfo:"skip"}
],mkLayout("Theoretical","Std. Residuals"),PC);
}catch(e){console.error("Assump plot error:",e)}
var msgs=[],questions=[];
switch(type){
case"clean":msgs.push({t:"Clean linear data: OLS should fit well. Residuals should be randomly scattered. Q-Q points should follow the reference line.",c:"info"});
questions=["Are residuals randomly scattered around zero?","Does the Q-Q plot show approximate normality?","Would you expect unusual observations here?"];break;
case"nonlinear":msgs.push({t:"Nonlinear data with curvature strength "+params.nonlin+". A straight line cannot capture the quadratic pattern. Look for curved patterns in the residual plot.",c:"warn"});
questions=["What pattern do you see in residuals vs fitted?","Would a polynomial or transformation help?","How does R-squared compare to clean?"];break;
case"hetero":msgs.push({t:"Heteroscedastic errors with strength "+params.hetero+". Error variance increases with X. Look for a funnel shape. Constant variance assumption is violated.",c:"warn"});
questions=["Do residuals show a funnel or fan shape?","Why might WLS be appropriate?","Does OLS slope remain unbiased despite heteroscedasticity?"];break;
case"outlier":msgs.push({t:"Large response outlier with magnitude "+params.outMag+". Check its standardized residual and Cook's Distance.",c:"warn"});
questions=["Which observation is unusual and why?","What happens to R-squared and MSE?","Should this observation be deleted?"];break;
case"leverage":msgs.push({t:"High-leverage observation at an extreme X value. Far from the mean of X but may or may not be off the trend line.",c:"acc"});
questions=["What is the leverage of the extreme observation?","Does it have a large residual?","What would happen if it moved off the trend line?"];break;
case"influential":msgs.push({t:"Highly influential observation at extreme X, far from the trend. Combines high leverage with large residual.",c:"warn"});
questions=["How does the fitted line change without this point?","What is its Cook's Distance?","Is this a data error, rare valid case, or sign the model is wrong?"];break;
}
var h=msgs.map(function(m){return'<div class="ei '+m.c+'">'+m.t+'</div>'}).join("");
h+='<div class="ei acc"><strong>Questions for students:</strong><br>'+questions.map(function(q){return"-- "+q}).join("<br>")+'</div>';
document.getElementById("ex4").innerHTML=h;
}

function updateRemedies(){
var fit=S.fit;if(!fit)return;
var extraLines=[];
var act=S.data.filter(function(d){return!d.excluded});
var axs=act.map(function(d){return d.x});
var xR=[Math.min.apply(null,axs),Math.max.apply(null,axs)];
if(document.getElementById("mhub").checked&&S.robustFit){
extraLines.push({x:xR,y:[S.robustFit.intercept+S.robustFit.slope*xR[0],S.robustFit.intercept+S.robustFit.slope*xR[1]],
color:"#16a34a",name:"Huber Robust",dash:"solid"})}
if(document.getElementById("mexc").checked&&S.fitNo){
extraLines.push({x:xR,y:[S.fitNo.intercept+S.fitNo.slope*xR[0],S.fitNo.intercept+S.fitNo.slope*xR[1]],
color:"#ea580c",name:"Excl. Selected",dash:"dash"})}
if(document.getElementById("mwls").checked&&S.wlsFit){
extraLines.push({x:xR,y:[S.wlsFit.intercept+S.wlsFit.slope*xR[0],S.wlsFit.intercept+S.wlsFit.slope*xR[1]],
color:"#dc2626",name:"WLS",dash:"dot"})}
renderScatter("s5",{extraLines:extraLines,h:420,showComp:false});
// Stats
var h='<div class="sg"><h3>OLS Baseline</h3>';
h+='<div class="sr"><span class="sl">Slope</span><span class="sv">'+fmt(fit.slope)+'</span></div>';
h+='<div class="sr"><span class="sl">Intercept</span><span class="sv">'+fmt(fit.intercept)+'</span></div>';
h+='<div class="sr"><span class="sl">R-squared</span><span class="sv">'+fmt(fit.r2)+'</span></div>';
h+='<div class="sr"><span class="sl">MSE</span><span class="sv">'+fmt(fit.mse)+'</span></div></div>';
if(S.robustFit){
h+='<div class="sg"><h3>Huber Robust</h3>';
h+='<div class="sr"><span class="sl">Slope</span><span class="sv gd">'+fmt(S.robustFit.slope)+'</span></div>';
h+='<div class="sr"><span class="sl">Intercept</span><span class="sv gd">'+fmt(S.robustFit.intercept)+'</span></div>';
h+='<div class="sr"><span class="sl">SSE</span><span class="sv gd">'+fmt(S.robustFit.sse)+'</span></div></div>'}
if(S.fitNo){
h+='<div class="sg"><h3>Without Selected</h3>';
h+='<div class="sr"><span class="sl">Slope</span><span class="sv hi">'+fmt(S.fitNo.slope)+'</span></div>';
h+='<div class="sr"><span class="sl">Intercept</span><span class="sv hi">'+fmt(S.fitNo.intercept)+'</span></div></div>'}
if(S.logFit){
h+='<div class="sg"><h3>Log(Y) Transform</h3>';
h+='<div class="sr"><span class="sl">Slope (log)</span><span class="sv am">'+fmt(S.logFit.slope,5)+'</span></div>';
h+='<div class="sr"><span class="sl">Intercept</span><span class="sv am">'+fmt(S.logFit.intercept,3)+'</span></div>';
h+='<div class="sr"><span class="sl">R-squared</span><span class="sv am">'+fmt(S.logFit.r2)+'</span></div>';
h+='<div class="sr"><span class="sl">Note</span><span class="sv" style="font-family:var(--fo);font-size:10px;color:var(--mu)">On log scale</span></div></div>'}
if(S.wlsFit){
h+='<div class="sg"><h3>Weighted LS</h3>';
h+='<div class="sr"><span class="sl">Slope</span><span class="sv" style="color:var(--rd)">'+fmt(S.wlsFit.slope)+'</span></div>';
h+='<div class="sr"><span class="sl">Intercept</span><span class="sv" style="color:var(--rd)">'+fmt(S.wlsFit.intercept)+'</span></div></div>'}
document.getElementById("sp5").innerHTML=h;
// Table
var rows='<tr><td>OLS</td><td>'+fmt(fit.slope)+'</td><td>'+fmt(fit.intercept)+'</td><td>R-sq='+fmt(fit.r2)+'</td><td>Unbiased, standard inference</td><td>Sensitive to outliers</td></tr>';
if(S.robustFit)rows+='<tr><td>Huber Robust</td><td>'+fmt(S.robustFit.slope)+'</td><td>'+fmt(S.robustFit.intercept)+'</td><td>SSE='+fmt(S.robustFit.sse)+'</td><td>Resists outliers</td><td>Less efficient on clean data</td></tr>';
if(S.fitNo)rows+='<tr><td>Excl. Selected</td><td>'+fmt(S.fitNo.slope)+'</td><td>'+fmt(S.fitNo.intercept)+'</td><td>R-sq='+fmt(S.fitNo.r2)+'</td><td>Sensitivity analysis</td><td>Must justify removal</td></tr>';
if(S.logFit)rows+='<tr><td>Log(Y)</td><td>'+fmt(S.logFit.slope,5)+'</td><td>'+fmt(S.logFit.intercept)+'</td><td>R-sq='+fmt(S.logFit.r2)+' (log)</td><td>Multiplicative errors</td><td>Different scale</td></tr>';
if(S.wlsFit)rows+='<tr><td>WLS</td><td>'+fmt(S.wlsFit.slope)+'</td><td>'+fmt(S.wlsFit.intercept)+'</td><td>wSSE='+fmt(S.wlsFit.weightedSse)+'</td><td>Heteroscedasticity</td><td>Needs variance structure</td></tr>';
document.getElementById("ct5").innerHTML='<table><thead><tr><th>Method</th><th>Slope</th><th>Intercept</th><th>Fit Measure</th><th>Advantage</th><th>Limitation</th></tr></thead><tbody>'+rows+'</tbody></table>';
document.getElementById("ex5").innerHTML='<div class="ei info"><strong>Remedies must match the problem:</strong> Outliers with ordinary leverage may need robust regression or investigation. High-leverage points require checking for errors or model misspecification. Heteroscedasticity calls for WLS or transformations. Nonlinear patterns need a different functional form.</div><div class="ei acc"><strong>R-squared note:</strong> R-squared values on transformed scales (e.g., log(Y)) are not directly comparable to those on the original scale.</div><div class="ei"><strong>Exclusion note:</strong> Removing an observation is a sensitivity analysis that should be documented and justified.</div>';
}

function updateAll(){
recalc();
switch(S.tab){
case 0: updatePlayground();break;
case 1: updateOutlier();break;
case 2: updatePlayground();renderDiag();break;
case 3: break;
case 4: updateRemedies();break;
}}

function switchTab(n){
S.tab=n;
var btns=document.querySelectorAll(".tab-b"),panels=document.querySelectorAll(".tab-p"),i;
for(i=0;i<btns.length;i++)btns[i].classList.remove("active");
for(i=0;i<panels.length;i++)panels[i].classList.remove("active");
document.querySelector('.tab-b[data-t="'+n+'"]').classList.add("active");
document.getElementById("p"+n).classList.add("active");
recalc();
switch(n){
case 0: updatePlayground();break;
case 1: updateOutlier();break;
case 2: updatePlayground();renderDiag();break;
case 3: updateAssumptions();break;
case 4: updateRemedies();break;
}}

// ========== DRAG ==========
function findNearest(gd,cx,cy,thresh){
thresh=thresh||18;
if(!gd||!gd._fullLayout)return-1;
var rect=gd.getBoundingClientRect();
var mx=cx-rect.left,my=cy-rect.top;
var xa=gd._fullLayout.xaxis,ya=gd._fullLayout.yaxis;
var minD=thresh,best=-1,i,px,py,d;
for(i=0;i<S.data.length;i++){
try{px=xa.d2p(S.data[i].x);py=ya.d2p(S.data[i].y)}catch(e){continue}
d=Math.sqrt((px-mx)*(px-mx)+(py-my)*(py-my));
if(d<minD){minD=d;best=i}}
return best}

function pixToData(gd,px,py){
var xa=gd._fullLayout.xaxis,ya=gd._fullLayout.yaxis;
var xr=xa.range,yr=ya.range;
return{
x:xr[0]+(px-xa._offset)/xa._length*(xr[1]-xr[0]),
y:yr[1]-(py-ya._offset)/ya._length*(yr[1]-yr[0])
}}

function setupDrag(divId){
var gd=document.getElementById(divId);if(!gd)return;
function onDown(e){
if(!gd._fullLayout)return;
var cx,cy;
if(e.touches){cx=e.touches[0].clientX;cy=e.touches[0].clientY}else{cx=e.clientX;cy=e.clientY}
var idx=findNearest(gd,cx,cy,18);
if(idx>=0){
e.preventDefault();e.stopPropagation();
S.sel=idx;S.dragIdx=idx;S.dragging=true;
updateAll();
}}
function onMove(e){
if(!S.dragging)return;
e.preventDefault();
var cx,cy;
if(e.touches){cx=e.touches[0].clientX;cy=e.touches[0].clientY}else{cx=e.clientX;cy=e.clientY}
if(!gd._fullLayout)return;
try{
var rect=gd.getBoundingClientRect();
var coords=pixToData(gd,cx-rect.left,cy-rect.top);
if(S.dragIdx>=0&&S.dragIdx<S.data.length){
S.data[S.dragIdx].x=Math.round(clamp(coords.x,-5,65)*100)/100;
S.data[S.dragIdx].y=Math.round(clamp(coords.y,-20,400)*100)/100;
updateAll();
}}catch(ex){}}
function onUp(){
if(S.dragging){S.dragging=false;S.dragIdx=-1;updateAll()}}
gd.addEventListener("mousedown",onDown,true);
gd.addEventListener("touchstart",onDown,{passive:false,capture:true});
document.addEventListener("mousemove",onMove);
document.addEventListener("touchmove",onMove,{passive:false});
document.addEventListener("mouseup",onUp);
document.addEventListener("touchend",onUp);
try{gd.on("plotly_click",function(ev){
if(ev.points&&ev.points.length>0){S.sel=ev.points[0].pointIndex;updateAll()}})}catch(e){}
}

// ========== EXPERIMENTS ==========
function runExperiment(exp){
if(S.sel<0)S.sel=Math.floor(S.data.length/2);
var si=S.sel;
var xs=[],ys=[],i;
for(i=0;i<S.data.length;i++){if(i!==si&&!S.data[i].excluded){xs.push(S.data[i].x);ys.push(S.data[i].y)}}
var baseFit=computeOLS(xs,ys);if(!baseFit)return;
var d=S.data[si];d.excluded=false;
var maxX=Math.max.apply(null,xs);
var midX=(Math.min.apply(null,xs)+maxX)/2;
switch(exp){
case"A":d.x=Math.round(midX*100)/100;d.y=Math.round((baseFit.intercept+baseFit.slope*midX)*100)/100;break;
case"B":d.x=Math.round(midX*100)/100;d.y=Math.round((baseFit.intercept+baseFit.slope*midX+50)*100)/100;break;
case"C":d.x=Math.round((maxX+15)*100)/100;d.y=Math.round((baseFit.intercept+baseFit.slope*d.x)*100)/100;break;
case"D":d.x=Math.round((maxX+15)*100)/100;d.y=Math.round((baseFit.intercept+baseFit.slope*d.x-40)*100)/100;break;
case"E":d.excluded=true;break;
}
S.exp=exp;
updateAll();
}

// ========== EVENTS ==========
function reveal(n){var el=document.getElementById("ca"+n);if(el)el.classList.toggle("show")}

function setupEvents(){
// Tab nav
document.querySelectorAll(".tab-b").forEach(function(b){
b.addEventListener("click",function(){switchTab(+this.getAttribute("data-t"))})});

// Sidebar toggle
document.getElementById("btn-sb").addEventListener("click",function(){
var sb=document.getElementById("sidebar");
sb.classList.toggle("hidden");
this.innerHTML=sb.classList.contains("hidden")?"&#x25B6;":"&#x25C0;";});

// Dataset controls
document.getElementById("cn").addEventListener("input",function(){
document.getElementById("ln").textContent=this.value;
S.data=generateData("clean",{n:+this.value,noise:+document.getElementById("cno").value,seed:+document.getElementById("cs").value});
S.sel=-1;updateAll()});
document.getElementById("cno").addEventListener("input",function(){
document.getElementById("lno").textContent=this.value;
S.data=generateData("clean",{n:+document.getElementById("cn").value,noise:+this.value,seed:+document.getElementById("cs").value});
S.sel=-1;updateAll()});
document.getElementById("cs").addEventListener("change",function(){
S.data=generateData("clean",{n:+document.getElementById("cn").value,noise:+document.getElementById("cno").value,seed:+this.value});
S.sel=-1;updateAll()});

document.getElementById("breset").addEventListener("click",function(){
S.data=generateData("clean",{n:+document.getElementById("cn").value,noise:+document.getElementById("cno").value,seed:+document.getElementById("cs").value});
S.sel=-1;updateAll()});
document.getElementById("bnew").addEventListener("click",function(){
var seed=Math.floor(Math.random()*100000);
document.getElementById("cs").value=seed;
S.data=generateData("clean",{n:+document.getElementById("cn").value,noise:+document.getElementById("cno").value,seed:seed});
S.sel=-1;updateAll()});
document.getElementById("bout").addEventListener("click",function(){
var maxY=0;S.data.forEach(function(d){if(d.y>maxY)maxY=d.y});
var maxX=0;S.data.forEach(function(d){if(d.x>maxX)maxX=d.x});
var nid=S.data.length;
S.data.push({id:nid,x:Math.round((maxX+5)*100)/100,y:Math.round((maxY+30)*100)/100,excluded:false});
S.sel=nid;updateAll()});

document.getElementById("brm").addEventListener("click",function(){
if(S.sel>=0){S.data[S.sel].excluded=true;updateAll()}});
document.getElementById("bexp").addEventListener("click",function(){
var divId=S.tab===2?"d1":"s"+(S.tab===3?4:S.tab===4?5:S.tab+1);
try{Plotly.downloadImage(document.getElementById(divId),{format:"png",width:1200,height:800,filename:"regression-diagnostic"})}catch(e){}});
document.getElementById("bcsv").addEventListener("click",function(){
var csv="ID,X,Y,Excluded,Fitted,Residual,Leverage,StdResidual,CooksD\n";
S.data.forEach(function(d,i){
var sd=S.diags[i]||{};
csv+=d.id+","+d.x+","+d.y+","+d.excluded+","+fmt(sd.fitted,4)+","+fmt(sd.residual,4)+","+fmt(sd.leverage,6)+","+fmt(sd.stdRes,4)+","+fmt(sd.cookD,6)+"\n"});
var blob=new Blob([csv],{type:"text/csv"});
var url=URL.createObjectURL(blob);
var a=document.createElement("a");a.href=url;a.download="regression-data.csv";a.click();
URL.revokeObjectURL(url)});
document.getElementById("bcmp").addEventListener("click",function(){switchTab(4)});

// Selected observation sliders
document.getElementById("csx").addEventListener("input",function(){
if(S.sel>=0){S.data[S.sel].x=+this.value;document.getElementById("lsx").textContent=(+this.value).toFixed(1);updateAll()}});
document.getElementById("csy").addEventListener("input",function(){
if(S.sel>=0){S.data[S.sel].y=+this.value;document.getElementById("lsy").textContent=(+this.value).toFixed(1);updateAll()}});
document.getElementById("cex").addEventListener("change",function(){
if(S.sel>=0){S.data[S.sel].excluded=this.checked;updateAll()}});

// Options
document.getElementById("ccomp").addEventListener("change",function(){
S.showComp=this.checked;updateAll()});
document.getElementById("cclass").addEventListener("change",function(){
S.cm=this.checked;document.body.classList.toggle("cm",this.checked);
document.getElementById("sidebar").classList.toggle("hidden",this.checked);updateAll()});

// Experiment buttons
document.querySelectorAll(".ebt").forEach(function(b){
b.addEventListener("click",function(){
document.querySelectorAll(".ebt").forEach(function(x){x.classList.remove("active")});
this.classList.add("active");
runExperiment(this.getAttribute("data-e"))})});

// Tab 4 controls
document.getElementById("bgen4").addEventListener("click",updateAssumptions);
[["cn4","ln4"],["cno4","lno4"],["cnl4","lnl4"],["cht4","lht4"],["com4","lom4"]].forEach(function(p){
document.getElementById(p[0]).addEventListener("input",function(){document.getElementById(p[1]).textContent=this.value})});

// Tab 5 toggles
["mhub","mexc","mlog","mwls"].forEach(function(id){
document.getElementById(id).addEventListener("change",function(){updateRemedies()})});

// Diagnostic plot click sync
["d1","d2","d3","d4","d5"].forEach(function(pid){
var el=document.getElementById(pid);if(!el)return;
try{el.on("plotly_click",function(ev){
if(ev.points&&ev.points.length>0){
var idx=ev.points[0].pointIndex;
var dd=S.diags.filter(function(d){return d.residual!==null});
if(idx>=0&&idx<dd.length){S.sel=dd[idx].id;updateAll();renderDiag()}}})}catch(e){}});
}

// ========== INIT ==========
function init(){
S.data=generateData("clean",{n:35,noise:1,seed:42});
S.sel=-1;
recalc();
setupEvents();
setupDrag("s1");
setupDrag("s2");
updatePlayground();
updateAssumptions();
updateRemedies();
}

if(document.readyState==="loading"){document.addEventListener("DOMContentLoaded",init)}else{init()}
