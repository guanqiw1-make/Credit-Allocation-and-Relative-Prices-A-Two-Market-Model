"""Run: python code/reproduce.py (from any working directory).

Recreates every figure, CSV, manuscript result macro, and numerical audit.
Only NumPy, SciPy and Matplotlib are required. No downloads or TeX rendering.
"""
from pathlib import Path
import csv
import json
import platform
import sys
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from model import (ROOT, parameters, baseline, bank_lending, simulate,
                   stationary_prices, stationary_reallocation_derivative)

P = parameters()
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
BLUE, RED, GREEN, GRAY = "#236092", "#b44a36", "#31826a", "#5f6470"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "axes.titlesize":11, "axes.labelsize":10,
                     "legend.fontsize":8.5, "xtick.labelsize":9,
                     "ytick.labelsize":9, "axes.spines.top":False,
                     "axes.spines.right":False, "axes.grid":True,
                     "grid.alpha":0.18, "grid.linewidth":0.6,
                     "pdf.fonttype":42, "ps.fonttype":42,
                     "savefig.facecolor":"white"})


def save(fig, name):
    fig.savefig(FIG / (name+".pdf"), bbox_inches="tight", pad_inches=.10, dpi=300,
                metadata={"Creator":"Reproducible model illustration", "Author":""})
    fig.savefig(FIG / (name+".png"), dpi=180, bbox_inches="tight", pad_inches=.10)
    plt.close(fig)


def export_csv(name, data):
    with (RES / (name+".csv")).open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        keys=list(data)
        w.writerow(keys)
        for k in range(len(data[keys[0]])):
            w.writerow(["" if np.isnan(data[key][k]) else f"{data[key][k]:.12g}" for key in keys])


def line(ax, t, y, label, color, style="-", width=1.8):
    ax.plot(t,y,label=label,color=color,ls=style,lw=width)


def make_mechanism():
    fig,ax=plt.subplots(figsize=(9.0,4.3))
    ax.set(xlim=(0,10), ylim=(0,4.35)); ax.axis("off")
    boxes=[(.65,3.45,8.7,.65,"Start of quarter: inherited loans, seller deposits and production capability",GRAY),
           (.65,1.65,2.3,1.1,"Bank decision\nGross loans $L_F,L_R$",BLUE),
           (3.55,1.65,2.6,1.1,"Purchase budgets\n$E_F=e_F+L_F$\n$E_R=e_R+L_R+\\mu H_{t-1}$",GREEN),
           (6.75,1.65,2.6,1.1,"Market clearing\n$P_j=(E_j/A_j)^{1/d_j}$\n$Q_j=E_j/P_j$",RED),
           (.65,.15,8.7,.8,"End of quarter: update $C_{F,t}, C_{R,t}$, seller deposits $H_t$ and supply state $z_{t+1}$",GRAY)]
    for x,y,w,h,txt,col in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08",facecolor=col+"0D",edgecolor=col,lw=1.2))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=10)
    def arrow(a,b,txt=None,offset=(0,0)):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color=GRAY,lw=1.2))
        if txt:
            mid=((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1])
            ax.text(*mid,txt,ha="center",va="center",fontsize=8.5,color=GRAY)
    for x,label in [(1.8,"$C_{F,t-1}, C_{R,t-1}, K_t$"),(4.85,"$H_{t-1}$"),(8.05,"$z_t$")]:
        arrow((x,3.35),(x,2.85))
        ax.text(x+.18,3.10,label,ha="left",va="center",fontsize=9,color=GRAY)
        arrow((x,1.55),(x,1.05))
    arrow((3.05,2.2),(3.45,2.2))
    arrow((6.25,2.2),(6.65,2.2))
    ax.plot([.55,.15,.15],[.55,.55,3.78],color=GRAY,lw=1.1)
    arrow((.15,3.78),(.55,3.78))
    ax.text(.30,2.17,"Next quarter",rotation=90,ha="center",va="center",fontsize=8.5,color=GRAY,
            bbox=dict(facecolor="white",edgecolor="none",pad=1))
    save(fig,"fig01_mechanism")


def make_reallocation(low,high):
    t=low["t"]
    fig,axs=plt.subplots(2,2,figsize=(9.0,6.6),layout="constrained")
    ax=axs[0,0]
    line(ax,t,100*low["rel_log"],rf"Weak supply response ($\chi={P['chi']:g}$)",BLUE)
    line(ax,t,100*high["rel_log"],rf"Strong supply response ($\chi={P['chi_high']:g}$)",RED)
    ax.axhline(0,color=GRAY,lw=.8); ax.set(title="(a) Relative price level",ylabel="100 × log difference from baseline")
    ax.legend(loc="best")
    ax=axs[0,1]
    line(ax,t,100*np.log(low["PF"]),"Existing-asset price",GRAY,"--")
    line(ax,t,100*np.log(low["PR"]),"Output price: weak supply response",BLUE)
    line(ax,t,100*np.log(high["PR"]),"Output price: strong supply response",RED)
    ax.axhline(0,color=GRAY,lw=.8); ax.set(title="(b) Price levels by market",ylabel="100 × log difference from baseline")
    ax.legend(loc="center right",bbox_to_anchor=(1,.69))
    ax=axs[1,0]
    line(ax,t,low["s"],"Gross-origination share $s_t$",GREEN)
    line(ax,t,low["c"],"Outstanding-stock share $c_t$",BLUE)
    b=baseline(P); ax.scatter([-1],[P["LF0"]/(P["LF0"]+P["LR0"])],s=23,c=GREEN,zorder=4)
    ax.scatter([-1],[b["CF"]/(b["CF"]+b["CR"])],s=23,c=BLUE,zorder=4)
    ax.set(title="(c) Two different allocation shares",ylabel="Share",xlabel="Quarters after reallocation",ylim=(.35,.86))
    ax.legend(loc="lower right")
    ax=axs[1,1]
    line(ax,t,100*high["G"],"Relative credit growth $G_t$",GRAY,"--")
    line(ax,t,100*high["D"],"Endogenous adjustment $D_t$",GREEN)
    line(ax,t,100*high["B"],"Relative price growth $B_t$",RED)
    ax.axhline(0,color=GRAY,lw=.8)
    ax.set(title="(d) Accounting decomposition: strong response",ylabel="100 × quarterly log change",xlabel="Quarters after reallocation")
    ax.legend(loc="best")
    for ax in axs.flat: ax.set_xlim(-2,P["periods"]-1)
    save(fig,"fig02_reallocation")


def make_threshold():
    rho=np.linspace(0,1,121)
    chi=np.linspace(0,1.9,121)
    R,C=np.meshgrid(rho,chi)
    ef=P["eF"]+P["LF0"]
    er=P["eR"]+P["LR0"]+R*ef
    eps=C*P["LR0"]/(P["nu"]+P["LR0"])
    dF,dR=1+P["etaF"],1+P["etaR"]
    derivative=1/(dF*ef)+(1-R)/((dR-eps)*er)-eps/((dR-eps)*P["LR0"])
    # The derivative is per nominal unit reallocated. A 10 pp share shift
    # moves one tenth of the reference total at fixed gross lending.
    loan_shift = 0.10 * (P["LF0"] + P["LR0"])
    effect = 100 * loan_shift * derivative
    fig,ax=plt.subplots(figsize=(8.0,4.7),layout="constrained")
    norm=matplotlib.colors.TwoSlopeNorm(vmin=min(-1,float(effect.min())),vcenter=0,vmax=max(1,float(effect.max())))
    # Rasterize only the heatmap fill to avoid PDF-viewer seams between cells.
    # Text, contours, symbols and all line figures remain vector elements.
    mesh=ax.pcolormesh(R,C,effect,cmap="RdBu_r",norm=norm,shading="auto",rasterized=True)
    cs=ax.contour(R,C,effect,levels=[0],colors="black",linewidths=1.5)
    ax.clabel(cs,fmt={0:"Zero marginal effect"},inline=True,fontsize=9)
    ax.scatter([P["rho"],P["rho"]],[P["chi"],P["chi_high"]],c="white",edgecolors="black",s=50,zorder=5)
    ax.annotate("Weak response",(P["rho"],P["chi"]),xytext=(9,7),textcoords="offset points",fontsize=9)
    ax.annotate("Strong response",(P["rho"],P["chi_high"]),xytext=(9,7),textcoords="offset points",fontsize=9)
    ax.set(xlabel=r"Share of sale receipts entering the spending buffer, $\rho$",
           ylabel=r"Supply-target elasticity, $\chi$",title="Local stationary effect of reallocating credit toward existing assets")
    fig.colorbar(mesh,ax=ax,label="100 × linearised log effect per 10 pp shift")
    save(fig,"fig03_threshold")
    with (RES/"threshold_grid.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["rho","chi","derivative_per_unit_reallocation","linearized_effect_log_percent_per_10pp"])
        w.writerows(zip(R.ravel(),C.ravel(),derivative.ravel(),effect.ravel()))


def make_bank(tight,ample):
    fig,axs=plt.subplots(2,2,figsize=(9.0,6.5),layout="constrained")
    t=tight["t"]
    ax=axs[0,0]
    for name,col in [("LF",BLUE),("LR",RED)]:
        line(ax,t,tight[name],name.replace("L","$L_")+"$: finite capacity",col)
        line(ax,t,ample[name],name.replace("L","$L_")+"$: ample capacity",col,"--")
    ax.set(title="(a) Gross loan originations",ylabel="Nominal units per quarter"); ax.legend()
    ax=axs[0,1]
    line(ax,t,tight["risk_stock"],"Finite-capacity loan book",BLUE)
    line(ax,t,ample["risk_stock"],"Ample-capacity loan book",GRAY,"--")
    ax.axhline(P["K"],color=RED,lw=1.2,ls=":",label=f"$K={P['K']:g}$")
    ax.set(title="(b) Risk-weighted outstanding loans",ylabel="Nominal units"); ax.legend()
    ax=axs[1,0]
    line(ax,t,100*tight["rel_log"],"Finite capacity",BLUE)
    line(ax,t,100*ample["rel_log"],"Ample capacity",GRAY,"--")
    ax.axhline(0,color=GRAY,lw=.8)
    ax.set(title="(c) Relative-price response to the rate cut",ylabel="100 × log difference from baseline",xlabel="Quarters after rate cut")
    ax.legend()
    ax=axs[1,1]
    line(ax,t,tight["s"],"Gross share: finite capacity",GREEN)
    line(ax,t,ample["s"],"Gross share: ample capacity",GREEN,"--")
    line(ax,t,tight["c"],"Stock share: finite capacity",BLUE)
    line(ax,t,ample["c"],"Stock share: ample capacity",BLUE,"--")
    ax.set(title="(d) Allocation and inherited composition",ylabel="Share",xlabel="Quarters after rate cut"); ax.legend()
    save(fig,"fig04_bank")


def make_legacy(series):
    fig,axs=plt.subplots(1,2,figsize=(9.0,3.6),layout="constrained")
    for ax,key,title in zip(axs,["ample","finite"],["(a) Capacity remains slack","(b) Capacity can bind"]):
        for portfolio,col,ls in [("A",BLUE,"-"),("B",RED,"--")]:
            cut,base=series[key+portfolio]
            line(ax,cut["t"],100*(cut["rel_log"]-base["rel_log"]),f"Initial portfolio {portfolio}",col,ls,2.)
        ax.axhline(0,color=GRAY,lw=.8)
        ax.set(title=title,xlabel="Quarters after rate cut",ylabel="100 × log difference from baseline")
        ax.legend()
    save(fig,"fig05_inheritance")


def quarterly_decomposition_checks(series):
    """Recover the quarter-end share and the price gap from separate model inputs."""
    reference = baseline(P)
    errors = {"quarterly_share_history_max_error": 0.,
              "quarterly_share_log_odds_gap_max_error": 0.,
              "quarterly_net_ratio_log_gap_max_error": 0.,
              "quarterly_sector_net_flow_log_gap_max_error": 0.,
              "quantity_turnover_closure_max_error": 0.}
    defined_quarters = 0
    for path in series.values():
        nf, nr = path["NF"], path["NR"]
        # Recover the two initial stocks, then accumulate flows over the whole horizon.
        cf0, cr0 = path["CF"][0] - nf[0], path["CR"][0] - nr[0]
        cf = cf0 + np.cumsum(nf)
        cr = cr0 + np.cumsum(nr)
        cf_prev = np.r_[cf0, cf[:-1]]
        cr_prev = np.r_[cr0, cr[:-1]]

        sector_gap = np.log1p(nf / cf_prev) - np.log1p(nr / cr_prev)
        opening, closing = path["cprev"], path["c"]
        log_odds_open = np.log(opening) - np.log1p(-opening)
        log_odds_close = np.log(closing) - np.log1p(-closing)
        residuals = {"quarterly_share_history_max_error": cf / (cf + cr) - path["c"],
                     "quarterly_share_log_odds_gap_max_error": log_odds_close - log_odds_open - path["D"] - path["B"],
                     "quarterly_sector_net_flow_log_gap_max_error": sector_gap - path["D"] - path["B"]}
        defined = np.isfinite(path["psi"])
        defined_quarters += int(defined.sum())
        if defined.any():
            c0, g, psi = (path[key][defined] for key in ("cprev", "gC", "psi"))
            credit_gap = np.log1p(g * psi / c0) - np.log1p(g * (1 - psi) / (1 - c0))
            residuals["quarterly_net_ratio_log_gap_max_error"] = credit_gap - path["D"][defined] - path["B"][defined]

        # Q = A^(1/d) E^(eta/d) and V = E/C, without using reported prices or quantities.
        changes = []
        for j in ("F", "R"):
            eta = P["eta" + j]
            expenditure = np.diff(np.log(path["E" + j]), prepend=np.log(reference["E" + j]))
            supply = np.diff(path["z"], prepend=0.) if j == "R" else 0.
            credit = np.log1p(nf / cf_prev) if j == "F" else np.log1p(nr / cr_prev)
            quantity = (supply + eta * expenditure) / (1 + eta)
            turnover = expenditure - credit
            changes.append(quantity - turnover)
        residuals["quantity_turnover_closure_max_error"] = changes[0] - changes[1] - path["D"]
        for key, values in residuals.items():
            errors[key] = max(errors[key], float(np.max(np.abs(values))))
    assert defined_quarters > 0
    assert all(value < 1e-11 for value in errors.values())
    return {**errors, "quarterly_share_log_odds_observations": sum(len(path["c"]) for path in series.values()),
            "quarterly_net_ratio_defined_observations": defined_quarters}


def continuous_share_integral_error():
    """Verify the integral on separate smooth stock paths, including zero total net flow."""
    # Total credit is 5 + 0.4 t - 0.1 t^2; both sector stocks stay positive on [0,3].
    def total(t):
        return 5 + .4*t - .1*t*t

    def asset_stock(t):
        return 2 + .05*t + .2*t*t

    def growth(t):
        return (.4 - .2*t) / total(t)

    def asset_net_flow_per_credit(t):
        return (.05 + .4*t) / total(t)

    def discount(start, end):
        return np.exp(-quad(growth, start, end)[0])

    errors = []
    for t in (.5, 1., 2., 2.5, 3.):
        # n_F / C equals g*psi where the ratio exists and remains finite at t=2.
        accumulated = quad(lambda u: discount(u, t) * asset_net_flow_per_credit(u), 0., t)[0]
        predicted = asset_stock(0.) / total(0.) * discount(0., t) + accumulated
        errors.append(abs(predicted - asset_stock(t) / total(t)))
    error = float(max(errors))
    assert error < 1e-12
    return error


def verify(series,low,high):
    checks={}
    allseries=list(series.values())
    checks.update(quarterly_decomposition_checks(series))
    checks["continuous_share_integral_max_error"] = continuous_share_integral_error()
    checks["max_price_identity_residual"]=max(float(np.max(np.abs(d["B"]-(d["G"]-d["D"])))) for d in allseries)
    checks["max_share_identity_residual"]=max(float(np.max(np.abs(d["share_residual"]))) for d in allseries)
    checks["max_cumulative_price_identity_residual"]=max(float(np.max(np.abs(np.cumsum(d["B"])-d["rel_log"]))) for d in allseries)
    assert checks["max_price_identity_residual"]<1e-11
    assert checks["max_share_identity_residual"]<1e-11
    assert checks["max_cumulative_price_identity_residual"]<1e-11
    checks["minimum_free_deposits"]=min(float(d["free_deposits"].min()) for d in allseries)
    checks["minimum_prefunding_slack"]=min(float(d["prefunding_slack"].min()) for d in allseries)
    assert checks["minimum_free_deposits"]>0
    assert checks["minimum_prefunding_slack"]>0
    for d in allseries:
        assert all(np.all(d[k]>0) for k in ["CF","CR","EF","ER","PF","PR","QF","QR","VF","VR"])
    # Independent closed forms for inherited credit and spending-buffer paths.
    b=baseline(P); t=low["t"]; shift=P["reallocation"]
    lf,lr=P["LF0"]+shift,P["LR0"]-shift
    exact_cf=lf/P["deltaF"]+(b["CF"]-lf/P["deltaF"])*(1-P["deltaF"])**(t+1)
    exact_cr=lr/P["deltaR"]+(b["CR"]-lr/P["deltaR"])*(1-P["deltaR"])**(t+1)
    exact_ER=b["ER"]-shift+P["rho"]*shift*(1-(1-P["mu"])**t)
    checks["credit_path_error"]=float(max(np.max(abs(low["CF"]-exact_cf)),np.max(abs(low["CR"]-exact_cr))))
    checks["recycling_path_error"]=float(np.max(abs(low["ER"]-exact_ER)))
    assert checks["credit_path_error"]<1e-10 and checks["recycling_path_error"]<1e-10
    # The reallocation has offsetting sector flows at impact. The sector formula
    # remains exact there even though the net-flow share cannot be formed.
    assert low["NF"][0] > 0 and low["NR"][0] < 0
    assert abs(low["NF"][0] + low["NR"][0]) < 1e-12
    assert np.isnan(low["psi"][0])
    impact_gap = np.log1p(low["NF"][0] / b["CF"]) - np.log1p(low["NR"][0] / b["CR"])
    checks["offsetting_net_flows_impact_log_gap_error"] = float(abs(impact_gap - low["D"][0] - low["B"][0]))
    assert checks["offsetting_net_flows_impact_log_gap_error"] < 1e-11
    # Independent optimizer at random feasible states, including inactive sectors.
    rng=np.random.default_rng(2731)
    error=0.
    for _ in range(120):
        cf,cr=rng.uniform(1,25,2)
        K=P["wF"]*(1-P["deltaF"])*cf+P["wR"]*(1-P["deltaR"])*cr+rng.uniform(.01,1.8)
        rate=float(rng.uniform(0,.035))
        f,r,lam,room=bank_lending(P,cf,cr,rate,K)
        margins=np.array([P["alphaF"]-rate-P["spread"]-P["costF"],P["alphaR"]-rate-P["spread"]-P["costR"]])
        weights=np.array([P["wF"],P["wR"]]); slopes=np.array([P["bF"],P["bR"]])
        objective=lambda x: float(np.sum(slopes*x*x-margins*x))
        result=minimize(objective,np.zeros(2),jac=lambda x:2*slopes*x-margins,
                        bounds=[(0,None),(0,None)],constraints=[{"type":"ineq","fun":lambda x:room-weights@x,"jac":lambda x:-weights}],
                        method="SLSQP",options={"ftol":1e-13,"maxiter":500})
        assert result.success
        error=max(error,float(np.max(abs(result.x-[f,r]))))
        assert weights@np.array([f,r])<=room+1e-10
        assert abs(lam*(room-weights@np.array([f,r])))<1e-10
    checks["bank_optimizer_max_difference"]=error
    assert error<2e-5
    # Analytic stationary derivative against independently solved finite changes.
    errs=[]
    for rho in [0.,.25,.6,1.]:
        for chi in [0.,.3,.9,1.5,1.9]:
            p=dict(P,rho=rho)
            analytic,_=stationary_reallocation_derivative(p,P["LF0"],P["LR0"],chi)
            h=1e-5
            plus=stationary_prices(p,P["LF0"]+h,P["LR0"]-h,chi)["rel_log"]
            minus=stationary_prices(p,P["LF0"]-h,P["LR0"]+h,chi)["rel_log"]
            errs.append(abs(analytic-(plus-minus)/(2*h)))
    checks["stationary_derivative_max_error"]=max(errs)
    assert max(errs)<1e-7
    # Fixed-state binding-policy derivative, checked without updating the stocks.
    cf,cr=20.,5.
    h=1e-7
    plus=np.array(bank_lending(P,cf,cr,P["i1"]+h,P["K"])[:2])
    minus=np.array(bank_lending(P,cf,cr,P["i1"]-h,P["K"])[:2])
    w=np.array([P["wF"],P["wR"]]); bsl=np.array([P["bF"],P["bR"]])
    A=np.sum(w/(2*bsl)); H=np.sum(w*w/(2*bsl))
    analytic=(-1+w*A/H)/(2*bsl)
    checks["binding_policy_derivative_error"]=float(np.max(abs((plus-minus)/(2*h)-analytic)))
    assert checks["binding_policy_derivative_error"]<1e-7
    # The stationary bank solution uses w/delta for stocks but w in the
    # one-period first-order conditions; compare it with the long recursion.
    delta=np.array([P["deltaF"],P["deltaR"]])
    unc=np.array(bank_lending(P,16.,6.,P["i1"],P["K_ample"])[:2])
    lam=(np.sum(w*unc/delta)-P["K"])/np.sum(w*w/(2*bsl*delta))
    stationary=unc-lam*w/(2*bsl)
    long_bank=simulate(P,periods=1600,rate=P["i1"],K=P["K"])
    checks["stationary_bank_lending"]=stationary.tolist()
    checks["stationary_bank_error"]=float(np.max(abs(stationary-[long_bank["LF"][-1],long_bank["LR"][-1]])))
    assert checks["stationary_bank_error"]<1e-10
    for chi in [P["chi"],P["chi_high"]]:
        long=simulate(P,periods=1600,chi=chi,loans=lambda t:(lf,lr))
        steady=stationary_prices(P,lf,lr,chi)
        assert abs(long["PR"][-1]-steady["PR"])<1e-10
    checks["long_horizon_matches_independent_stationary_root"]=True
    checks["tests_passed"]=True
    return checks


def main():
    x=P["reallocation"]
    loan=lambda t:(P["LF0"]+x,P["LR0"]-x)
    low=simulate(P,loans=loan,chi=P["chi"])
    high=simulate(P,loans=loan,chi=P["chi_high"])
    series={"reallocation_weak":low,"reallocation_strong":high}
    tight=simulate(P,periods=P["rate_periods"],rate=P["i1"],K=P["K"])
    ample=simulate(P,periods=P["rate_periods"],rate=P["i1"],K=P["K_ample"])
    base=simulate(P,periods=P["rate_periods"],rate=P["i0"],K=P["K"])
    assert np.max(abs(base["rel_log"]))<1e-12
    assert tight["risk_stock"].max()<=P["K"]+1e-10
    series.update(rate_cut_finite=tight,rate_cut_ample=ample,rate_baseline=base)
    legacy={}
    for key,K in [("ample",P["K_ample"]),("finite",P["K"])]:
        for name,initial in [("A",(16.,6.)),("B",(24.,2.))]:
            cut=simulate(P,periods=P["rate_periods"],rate=P["i1"],K=K,initial_credit=initial)
            reference=simulate(P,periods=P["rate_periods"],rate=P["i0"],K=K,initial_credit=initial)
            legacy[key+name]=(cut,reference)
            series[f"inheritance_{key}_{name}_cut"]=cut
            series[f"inheritance_{key}_{name}_baseline"]=reference
    assert np.max(abs(legacy["ampleA"][0]["rel_log"]-legacy["ampleB"][0]["rel_log"]))<1e-12
    audit=verify(series,low,high)
    audit["slack_capacity_inheritance_price_difference"]=float(np.max(abs(legacy["ampleA"][0]["rel_log"]-legacy["ampleB"][0]["rel_log"])))
    audit["versions"]={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}
    audit["method"]="Deterministic quarterly recursion; bracketed scalar roots; independent SLSQP and analytic checks."
    make_mechanism(); make_reallocation(low,high); make_threshold(); make_bank(tight,ample); make_legacy(legacy)
    for name,d in series.items(): export_csv(name,d)
    weak=stationary_prices(P,P["LF0"]+x,P["LR0"]-x,P["chi"])
    strong=stationary_prices(P,P["LF0"]+x,P["LR0"]-x,P["chi_high"])
    _,eth=stationary_reallocation_derivative(P,P["LF0"],P["LR0"],P["chi"])
    chith=eth*(P["nu"]+P["LR0"])/P["LR0"]
    crossings=np.where(high["rel_log"]<0)[0]
    binding=np.where(tight["shadow"]>1e-10)[0]
    metrics={"impact_relative_log_percent":100*high["rel_log"][0],
             "weak_stationary_relative_log_percent":100*weak["rel_log"],
             "strong_stationary_relative_log_percent":100*strong["rel_log"],
             "strong_first_negative_relative_level_quarter":int(crossings[0]) if len(crossings) else None,
             "finite_capacity_first_binding_quarter":int(binding[0]) if len(binding) else None,
             "baseline_effective_elasticity_threshold":eth,"baseline_chi_threshold":chith,
             "rate_finite_final_relative_log_percent":100*tight["rel_log"][-1],
             "rate_ample_final_relative_log_percent":100*ample["rel_log"][-1],
             "weak_stationary_PF":weak["PF"],"weak_stationary_PR":weak["PR"],
             "strong_stationary_PR":strong["PR"],
             "initial_stock_share":baseline(P)["CF"]/(baseline(P)["CF"]+baseline(P)["CR"]),
             "final_stock_share_limit":((P["LF0"]+x)/P["deltaF"])/((P["LF0"]+x)/P["deltaF"]+(P["LR0"]-x)/P["deltaR"])}
    (RES/"summary.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    (RES/"validation.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    names={"ImpactRelative":"impact_relative_log_percent","WeakLongRelative":"weak_stationary_relative_log_percent",
           "StrongLongRelative":"strong_stationary_relative_log_percent","ChiThreshold":"baseline_chi_threshold",
           "RateFiniteLast":"rate_finite_final_relative_log_percent","RateAmpleLast":"rate_ample_final_relative_log_percent"}
    macros=["% Generated by code/reproduce.py; do not hand-edit numerical results."]
    macros += [rf"\newcommand{{\{k}}}{{{metrics[v]:.2f}}}" for k,v in names.items()]
    macros += [rf"\newcommand{{\ReversalQuarter}}{{{metrics['strong_first_negative_relative_level_quarter']}}}",
               rf"\newcommand{{\BindingQuarter}}{{{metrics['finite_capacity_first_binding_quarter']}}}"]
    (RES/"numbers.tex").write_text("\n".join(macros)+"\n",encoding="utf-8")
    print(json.dumps({"results":metrics,"validation":audit},indent=2))


if __name__=="__main__":
    main()
