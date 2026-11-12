"""Recursive two-market model. All flows and rates refer to one quarter.

No empirical data or random draws are used. Prices are dimensionless indices.
Credit and deposits are end-of-period nominal stocks. z and H are inherited
at the start of a period. See the paper for the partial-equilibrium scope.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]


def parameters():
    return json.loads((Path(__file__).with_name("parameters.json")).read_text())


def baseline(p):
    ef = p["eF"] + p["LF0"]
    er = p["eR"] + p["LR0"] + p["rho"] * ef
    return dict(CF=p["LF0"] / p["deltaF"],
                CR=p["LR0"] / p["deltaR"],
                H=p["rho"] * ef / p["mu"], z=0.,
                EF=ef, ER=er, AF=ef, AR=er,
                PF=1., PR=1., QF=ef, QR=er, M=p["M0"])


def bank_lending(p, cf, cr, rate, K):
    """Exact concave-program solution, including zero-loan corners."""
    weights = np.array([p["wF"], p["wR"]])
    inherited = np.array([(1-p["deltaF"])*cf, (1-p["deltaR"])*cr])
    room = K - weights @ inherited
    if room < -1e-10:
        raise ValueError("Inherited surviving loans violate capacity; forced deleveraging is not modelled.")
    room = max(0., room)
    margin = np.array([p["alphaF"]-rate-p["spread"]-p["costF"],
                       p["alphaR"]-rate-p["spread"]-p["costR"]])
    slopes = np.array([p["bF"], p["bR"]])
    def loans(lam):
        return np.maximum(0., (margin-lam*weights)/(2*slopes))
    if weights @ loans(0.) <= room + 1e-13:
        lam = 0.
    else:
        hi = max(1., float(np.max(margin/weights))+1.)
        lam = brentq(lambda x: weights @ loans(x)-room, 0., hi,
                     xtol=1e-14, rtol=1e-14)
    lf, lr = loans(lam)
    return float(lf), float(lr), float(lam), float(room)


def prices(p, ef, er, z):
    b = baseline(p)
    ar = b["AR"] * np.exp(z)
    pf = (ef / b["AF"])**(1/(1+p["etaF"]))
    pr = (er / ar)**(1/(1+p["etaR"]))
    return pf, pr, ef/pf, er/pr, ar


def simulate(p, *, periods=None, chi=None, loans=None, rate=None, K=None,
             initial_credit=None):
    """loans: callable t -> (LF, LR), or None to solve the bank problem.

    Predetermined-lending experiments deliberately bypass the bank constraint.
    In a bank experiment the starting weighted loan stock must be <= K.
    """
    b = baseline(p)
    T = periods or p["periods"]
    chi = p["chi"] if chi is None else chi
    if not (0 <= chi < 1+p["etaR"]):
        raise ValueError("Supply stability restriction requires 0 <= chi < 1+etaR.")
    K = p["K"] if K is None else K
    cf, cr = initial_credit or (b["CF"], b["CR"])
    if loans is None and p["wF"]*cf+p["wR"]*cr > K+1e-10:
        raise ValueError("Infeasible initial bank portfolio.")
    h, z, m = b["H"], 0., p["M0"]
    prev_pf, prev_pr = 1., 1.
    prev_qf, prev_qr = b["QF"], b["QR"]
    prev_vf, prev_vr = b["EF"]/cf, b["ER"]/cr
    records = []
    for t in range(T):
        it = p["i0"] if rate is None else (rate(t) if callable(rate) else rate)
        if loans is None:
            lf, lr, lam, room = bank_lending(p, cf, cr, it, K)
        else:
            lf, lr = loans(t)
            lam, room = np.nan, np.nan
        if lf < 0 or lr < 0:
            raise ValueError("Negative gross lending.")
        rf, rr = p["deltaF"]*cf, p["deltaR"]*cr
        nf, nr = lf-rf, lr-rr
        cf1, cr1 = cf+nf, cr+nr
        ef = p["eF"] + lf
        recycled = p["mu"] * h
        er = p["eR"] + lr + recycled
        pf, pr, qf, qr, ar = prices(p, ef, er, z)
        h1 = (1-p["mu"])*h + p["rho"]*ef
        z1 = (1-p["omega"])*z + p["omega"]*chi*np.log(
            (p["nu"]+lr/pr)/(p["nu"]+p["LR0"]))
        m1 = m+nf+nr
        vf, vr = ef/cf1, er/cr1
        gf, gr = np.log(cf1/cf), np.log(cr1/cr)
        pi_f, pi_r = np.log(pf/prev_pf), np.log(pr/prev_pr)
        D = np.log(qf/prev_qf)-np.log(qr/prev_qr)-np.log(vf/prev_vf)+np.log(vr/prev_vr)
        total_net = nf+nr
        psi = nf/total_net if abs(total_net) > 1e-10*max(1.,cf+cr) else np.nan
        cprev = cf/(cf+cr)
        c = cf1/(cf1+cr1)
        records.append(dict(t=t, LF=lf, LR=lr, CF=cf1, CR=cr1,
                            NF=nf, NR=nr, repayment=rf+rr, psi=psi,
                            s=lf/(lf+lr) if lf+lr>0 else np.nan,
                            c=c, cprev=cprev, gC=total_net/(cf+cr),
                            EF=ef, ER=er, H=h1, recycled=recycled,
                            z=z, AR=ar, PF=pf, PR=pr, QF=qf, QR=qr,
                            VF=vf, VR=vr, G=gf-gr, D=D, B=pi_f-pi_r,
                            piF=pi_f, piR=pi_r, rel_log=np.log(pf/pr),
                            M=m1, free_deposits=m1-h1,
                            prefunding_slack=m-h-rf-rr-p["eF"]-p["eR"],
                            shadow=lam, room=room, rate=it,
                            risk_stock=p["wF"]*cf1+p["wR"]*cr1,
                            share_residual=c-cprev-((1-cprev)*nf-cprev*nr)/(cf1+cr1)))
        cf, cr, h, z, m = cf1, cr1, h1, z1, m1
        prev_pf, prev_pr, prev_qf, prev_qr = pf, pr, qf, qr
        prev_vf, prev_vr = vf, vr
    return {key:np.array([r[key] for r in records]) for key in records[0]}


def stationary_prices(p, lf, lr, chi):
    """Independent scalar-root solution, not a simulated terminal point."""
    b = baseline(p)
    ef = p["eF"]+lf
    er = p["eR"]+lr+p["rho"]*ef
    dR = 1+p["etaR"]
    def eq(logpr):
        return dR*logpr - np.log(er/b["AR"]) + chi*np.log(
            (p["nu"]+lr*np.exp(-logpr))/(p["nu"]+p["LR0"]))
    logpr = brentq(eq, -50., 50., xtol=1e-13)
    pf = (ef/b["AF"])**(1/(1+p["etaF"]))
    pr = np.exp(logpr)
    epsilon = chi*lr/(p["nu"]*pr+lr)
    return dict(PF=pf, PR=pr, EF=ef, ER=er, epsilon=epsilon,
                rel_log=np.log(pf/pr))


def stationary_reallocation_derivative(p, lf, lr, chi):
    r = stationary_prices(p, lf, lr, chi)
    eps = r["epsilon"]
    dF, dR = 1+p["etaF"], 1+p["etaR"]
    derivative = 1/(dF*r["EF"]) + (1-p["rho"])/((dR-eps)*r["ER"]) - eps/((dR-eps)*lr)
    threshold = lr*(dR+(1-p["rho"])*dF*r["EF"]/r["ER"])/(lr+dF*r["EF"])
    return derivative, threshold
