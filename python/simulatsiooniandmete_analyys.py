import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter

# Konstandid ja parameetrid

R1 = 22e3
R2 = 220
R3 = 22e3
R5 = 3.3e3
R6 = 2.2e3
R7 = 220
US = 9.0

iv_failid = {
    "lm324": "lm324_iv.txt",
    "ad704": "ad704_iv.txt",
    "ad817": "ad817_iv.txt",
    "opa445": "opa445_iv.txt",
    "op07": "op07_iv.txt",
}

opampi_sildid = {
    "lm324": "LM324",
    "ad704": "AD704",
    "ad817": "AD817A",
    "ad817a": "AD817A",
    "opa445": "OPA445",
    "op07": "OP07",
}

varvid = {
    "lm324": "#800000",
    "ad704": "#B8860B",
    "ad817": "#1B5E20",
    "ad817a": "#1B5E20",
    "opa445": "#0B1F5C",
    "op07": "#6A1B9A",
}

dunaamika_failid = {
    "ad704": {
        "signaal": "ad704_signaal_1.55k.txt",
        "faas": "ad704_r4_1.55k.txt",
        "aeg_xlim": (0.0, 0.015),
    },
    "ad817": {
        "signaal": "ad817_signaal_2.05k.txt",
        "faas": "ad817a_r4_2.05k.txt",
        "aeg_xlim": (0.0, 0.015),
    },
    "lm324": {
        "signaal": "lm324_signaal_1.85k.txt",
        "faas": "lm324_r4_1.85k.txt",
        "aeg_xlim": (0.035, 0.050),
    },
    "opa445": {
        "signaal": "opa445_signaal_2.00k.txt",
        "faas": "opa445_r4_2.00k.txt",
        "aeg_xlim": (0.0, 0.015),
    },
    "op07": {
        "signaal": "op07_signaal_1.95k.txt",
        "faas": "op07_r4_1.95k.txt",
        "aeg_xlim": (0.0, 0.015),
    },
}

arengu_failid = [
    "lm324_r4_1.87k.txt",
    "lm324_r4_1.88k.txt",
    "lm324_r4_1.93k.txt",
    "lm324_r4_1.97k.txt",
    "lm324_r4_2.01k.txt",
    "lm324_r4_2.02k.txt",
    "lm324_r4_2.04k.txt",
    "lm324_r4_2.10k.txt",
    "lm324_r4_2.20k.txt",
]

# Dünaamiliste piirangute arvutuse parameetrid

opampide_andmed = {
    "lm324": {
        "nimi": "LM324",
        "r4_min": 1340,
        "r4_max": 1870,
        "gbw_hz": 0.829e6,
        "epsilon_A_percent": 0.011,
        "sr_sim_999": 2.019,
        "eta_sr": 0.26,
    },
    "ad704": {
        "nimi": "AD704",
        "r4_min": 1290,
        "r4_max": 1610,
        "gbw_hz": 0.392e6,
        "epsilon_A_percent": 0.018,
        "sr_sim_999": 0.148,
        "eta_sr": 0.95,
    },
    "ad817": {
        "nimi": "AD817A",
        "r4_min": 1670,
        "r4_max": 2070,
        "gbw_hz": 45.7e6,
        "epsilon_A_percent": 0.195,
        "sr_sim_999": 0.434,
        "eta_sr": 462.6,
    },
    "opa445": {
        "nimi": "OPA445",
        "r4_min": 1580,
        "r4_max": 2030,
        "gbw_hz": 2.14e6,
        "epsilon_A_percent": 0.006,
        "sr_sim_999": 2.205,
        "eta_sr": 2.72,
    },
    "op07": {
        "nimi": "OP07",
        "r4_min": 1560,
        "r4_max": 1960,
        "gbw_hz": 1.53e6,
        "epsilon_A_percent": 0.005,
        "sr_sim_999": 0.334,
        "eta_sr": 1.78,
    },
}

amplituudi_vahemikud = {
    "ad704": (1730, 1900),
    "ad817a": (2080, 2350),
    "lm324": (1880, 2200),
    "opa445": (2050, 2350),
    "op07": (2040, 2300),
}

# Funktsioonid

def turvaline_jagamine(a, b):
    return np.nan if abs(b) < 1e-30 else a / b


def suhteline_abs_viga(moot, ideaal):
    return np.nan if abs(ideaal) < 1e-30 else abs(moot - ideaal) / abs(ideaal)


def protsent(x):
    return 100.0 * x


def normaliseeri_opamp_key(opamp):
    """Ühtlustab failinimedest tulevad op-võimendi tähised."""
    if opamp is None:
        return None

    opamp = opamp.lower()

    if opamp == "ad817a":
        return "ad817"

    return opamp


def loe_kaheveeruline_fail(fail, sorteeri=True):
    andmed = np.loadtxt(fail)
    x = andmed[:, 0]
    y = andmed[:, 1]

    if sorteeri:
        indeksid = np.argsort(x)
        x = x[indeksid]
        y = y[indeksid]

    return x, y


def suurim_true_vahemik(mask):
    m = mask.astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    algused = np.where(d == 1)[0]
    lopud = np.where(d == -1)[0]

    if len(algused) == 0:
        return None

    pikkused = lopud - algused
    k = np.argmax(pikkused)

    return int(algused[k]), int(lopud[k])


def sirge_sobitus(x, y):
    if len(x) < 2:
        raise RuntimeError("Lineaarseks sobituseks on vaja vähemalt 2 punkti.")

    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def ideaalparameetrid(R1, R2, R3, R5, R6, R7, US):
    B1 = (R5 / (R3 + R5)) * US
    B2 = (R6 / (R7 + R6)) * US

    m0 = -1.0 / R5 - 1.0 / R6
    m1 = 1.0 / R1 - 1.0 / R6
    m2 = 1.0 / R1 + 1.0 / R2

    return {"B1": B1, "B2": B2, "m0": m0, "m1": m1, "m2": m2}


def joonista_yks_iv(fail, nimi):
    u, i = loe_kaheveeruline_fail(fail)

    plt.figure(figsize=(8, 5))
    plt.plot(u, i, color="maroon", linewidth=0.7)
    plt.xlabel(r"$u_1\;(\mathrm{V})$", fontsize=13)
    plt.ylabel(r"$f(u_1)\;(\mathrm{A})$", fontsize=13)
    plt.grid(True, linestyle="dotted", alpha=0.5)
    plt.xlim(-9, 9)
    plt.title(nimi)
    plt.tight_layout()
    plt.show()


def joonista_koik_iv(failid):
    plt.figure(figsize=(8, 5))

    for opamp, fail in failid.items():
        u, i = loe_kaheveeruline_fail(fail)
        plt.plot(
            u,
            i,
            label=opampi_sildid[opamp],
            color=varvid[opamp],
            linewidth=0.7,
        )

    plt.xlabel(r"$u_1$ (V)", fontsize=13)
    plt.ylabel(r"$f(u_1)$ (A)", fontsize=13)
    plt.grid(True, linestyle="dotted", alpha=0.5)
    plt.xlim(-8.5, 8.5)
    plt.ylim(-0.012, 0.012)
    plt.legend()
    plt.tight_layout()
    plt.show()


def leia_lm324_parema_haru_lopp(u_parem, i_parem):
    if len(u_parem) < 6:
        return None

    tuletis = np.gradient(i_parem, u_parem)

    akna_pikkus = 5 if len(tuletis) >= 5 else 3
    tuum = np.ones(akna_pikkus) / akna_pikkus
    tuletis_silutud = np.convolve(tuletis, tuum, mode="same")

    n_ref = max(3, min(8, len(tuletis_silutud) // 3))
    algtous = np.median(tuletis_silutud[:n_ref])

    for k in range(n_ref, len(tuletis_silutud) - 2):
        if abs(algtous) < 1e-30:
            break

        muutus = abs(tuletis_silutud[k] - algtous) / abs(algtous)

        if muutus > 0.6:
            return u_parem[k]

    return None


def leia_viisloiguline_lahendus(fail, opamp, lm324=False, plot=True):
    u, i = loe_kaheveeruline_fail(fail)
    tuletis = np.gradient(i, u)

    negatiivne = tuletis < 0
    indeksid = np.where(negatiivne)[0]

    if len(indeksid) < 10:
        raise RuntimeError(f"{opamp}: negatiivset piirkonda ei leitud.")

    indeks_1 = indeksid[0]
    indeks_4 = indeksid[-1]

    bp1 = u[indeks_1]
    bp4 = u[indeks_4]

    u_sisemine = u[indeks_1:indeks_4 + 1]
    tuletis_sisemine = tuletis[indeks_1:indeks_4 + 1]

    lavi = 0.5 * (np.median(tuletis_sisemine) + np.min(tuletis_sisemine))
    j2rsk_mask = tuletis_sisemine < lavi

    vahemik = suurim_true_vahemik(j2rsk_mask)

    if vahemik is None:
        raise RuntimeError(f"{opamp}: keskset järsku piirkonda ei leitud.")

    indeks_2, indeks_3 = vahemik

    bp2 = u_sisemine[indeks_2]
    bp3 = u_sisemine[indeks_3 - 1]

    mask_vasak_valimine = u < bp1
    mask_vasak_sisemine = (u >= bp1) & (u < bp2)
    mask_keskne = (u >= bp2) & (u <= bp3)
    mask_parem_sisemine = (u > bp3) & (u <= bp4)

    if lm324:
        u_parem = u[u > bp4]
        i_parem = i[u > bp4]
        bp5 = leia_lm324_parema_haru_lopp(u_parem, i_parem)

        if bp5 is None:
            mask_parem_valimine = u > bp4
        else:
            mask_parem_valimine = (u > bp4) & (u < bp5)
    else:
        bp5 = None
        mask_parem_valimine = u > bp4

    m_ol, b_ol = sirge_sobitus(u[mask_vasak_valimine], i[mask_vasak_valimine])
    m_il, b_il = sirge_sobitus(u[mask_vasak_sisemine], i[mask_vasak_sisemine])
    m_st, b_st = sirge_sobitus(u[mask_keskne], i[mask_keskne])
    m_ir, b_ir = sirge_sobitus(u[mask_parem_sisemine], i[mask_parem_sisemine])
    m_or, b_or = sirge_sobitus(u[mask_parem_valimine], i[mask_parem_valimine])

    tulemus = {
        "breakpoints": (bp1, bp2, bp3, bp4),
        "outer_right_end": bp5,
        "fits": {
            "outer_left": (m_ol, b_ol),
            "inner_left": (m_il, b_il),
            "inner_steep": (m_st, b_st),
            "inner_right": (m_ir, b_ir),
            "outer_right": (m_or, b_or),
        },
    }

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(u, i, linewidth=0.8, label="andmed")

        def joonista_sirge(xmin, xmax, m, b, silt):
            x = np.linspace(xmin, xmax, 100)
            plt.plot(x, m * x + b, "--", linewidth=1.0, label=silt)

        joonista_sirge(u.min(), bp1, m_ol, b_ol, "välimine vasak")
        joonista_sirge(bp1, bp2, m_il, b_il, "sisemine vasak")
        joonista_sirge(bp2, bp3, m_st, b_st, "keskne")
        joonista_sirge(bp3, bp4, m_ir, b_ir, "sisemine parem")

        u_fit = u[mask_parem_valimine]
        joonista_sirge(u_fit.min(), u_fit.max(), m_or, b_or, "välimine parem")

        for bp in [bp1, bp2, bp3, bp4]:
            plt.axvline(bp, linestyle=":", linewidth=1)

        if bp5 is not None:
            plt.axvline(bp5, linestyle=":", linewidth=1, alpha=0.7)

        plt.title(opampi_sildid[opamp])
        plt.xlabel(r"$u_1\;(\mathrm{V})$")
        plt.ylabel(r"$i\;(\mathrm{A})$")
        plt.grid(True, linestyle="dotted", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return tulemus


def arvuta_koondparameetrid(tulemus, R1, R2, R3, R5, R6, R7, US):
    bp1, bp2, bp3, bp4 = tulemus["breakpoints"]

    m_ol, b_ol = tulemus["fits"]["outer_left"]
    m_il, b_il = tulemus["fits"]["inner_left"]
    m_st, b_st = tulemus["fits"]["inner_steep"]
    m_ir, b_ir = tulemus["fits"]["inner_right"]
    m_or, b_or = tulemus["fits"]["outer_right"]

    ideaal = ideaalparameetrid(R1, R2, R3, R5, R6, R7, US)

    B1_vasak = abs(bp2)
    B1_parem = abs(bp3)
    B2_vasak = abs(bp1)
    B2_parem = abs(bp4)

    B1_kesk = 0.5 * (B1_vasak + B1_parem)
    B2_kesk = 0.5 * (B2_vasak + B2_parem)

    abs_m0 = abs(m_st)
    abs_m1_vasak = abs(m_il)
    abs_m1_parem = abs(m_ir)
    abs_m2_vasak = abs(m_ol)
    abs_m2_parem = abs(m_or)

    abs_m1 = 0.5 * (abs_m1_vasak + abs_m1_parem)
    abs_m2 = 0.5 * (abs_m2_vasak + abs_m2_parem)

    return {
        r"$\Delta B_1$ (V)": bp3 - bp2,
        r"$\Delta B_2$ (V)": bp4 - bp1,
        r"$\delta_{B,1}$ (%)": protsent(turvaline_jagamine(abs(B1_vasak - B1_parem), B1_kesk)),
        r"$\delta_{B,2}$ (%)": protsent(turvaline_jagamine(abs(B2_vasak - B2_parem), B2_kesk)),
        r"$\delta_{m,1}$ (%)": protsent(turvaline_jagamine(abs(abs_m1_vasak - abs_m1_parem), abs_m1)),
        r"$\delta_{m,2}$ (%)": protsent(turvaline_jagamine(abs(abs_m2_vasak - abs_m2_parem), abs_m2)),
        r"$\delta_{m,0}$ (%)": protsent(turvaline_jagamine(abs(b_st), abs_m0 * B1_kesk)),
        r"$|m_1|/|m_0|$": turvaline_jagamine(abs_m1, abs_m0),
        r"$|m_2|/|m_0|$": turvaline_jagamine(abs_m2, abs_m0),
        r"$|m_2|/|m_1|$": turvaline_jagamine(abs_m2, abs_m1),
        r"$U_{\mathrm{sat}}^{(B_1)}$ (V)": B1_kesk * (R3 + R5) / R5,
        r"$U_{\mathrm{sat}}^{(B_2)}$ (V)": B2_kesk * (R7 + R6) / R6,
        r"$\varepsilon(B_1)$ (%)": protsent(suhteline_abs_viga(B1_kesk, ideaal["B1"])),
        r"$\varepsilon(B_2)$ (%)": protsent(suhteline_abs_viga(B2_kesk, ideaal["B2"])),
        r"$\varepsilon(m_0)$ (%)": protsent(suhteline_abs_viga(abs_m0, abs(ideaal["m0"]))),
        r"$\varepsilon(m_1)$ (%)": protsent(suhteline_abs_viga(abs_m1, abs(ideaal["m1"]))),
        r"$\varepsilon(m_2)$ (%)": protsent(suhteline_abs_viga(abs_m2, abs(ideaal["m2"]))),
    }


def analyysi_iv_koondtabel(failid):
    tulemused = {}
    koond_read = []
    sobituse_read = []

    for opamp, fail in failid.items():
        tulemus = leia_viisloiguline_lahendus(
            fail,
            opamp=opamp,
            lm324=(opamp == "lm324"),
            plot=True,
        )

        tulemused[opamp] = tulemus
        koond = arvuta_koondparameetrid(tulemus, R1, R2, R3, R5, R6, R7, US)

        koond_read.append({"opamp": opampi_sildid[opamp], **koond})

        bp1, bp2, bp3, bp4 = tulemus["breakpoints"]
        m_ol, b_ol = tulemus["fits"]["outer_left"]
        m_il, b_il = tulemus["fits"]["inner_left"]
        m_st, b_st = tulemus["fits"]["inner_steep"]
        m_ir, b_ir = tulemus["fits"]["inner_right"]
        m_or, b_or = tulemus["fits"]["outer_right"]

        sobituse_read.append({
            "opamp": opampi_sildid[opamp],
            "bp1": bp1,
            "bp2": bp2,
            "bp3": bp3,
            "bp4": bp4,
            "m_ol": m_ol,
            "m_il": m_il,
            "m_st": m_st,
            "m_ir": m_ir,
            "m_or": m_or,
            "b_ol": b_ol,
            "b_il": b_il,
            "b_st": b_st,
            "b_ir": b_ir,
            "b_or": b_or,
        })

    return tulemused, pd.DataFrame(koond_read), pd.DataFrame(sobituse_read)


def ehita_loigud(tulemus):
    bp1, bp2, bp3, bp4 = tulemus["breakpoints"]

    m_ol, b_ol = tulemus["fits"]["outer_left"]
    m_il, b_il = tulemus["fits"]["inner_left"]
    m_st, b_st = tulemus["fits"]["inner_steep"]
    m_ir, b_ir = tulemus["fits"]["inner_right"]
    m_or, b_or = tulemus["fits"]["outer_right"]

    return [
        {"nimi": "outer_left", "xmin": -np.inf, "xmax": bp1, "m": m_ol, "q": b_ol},
        {"nimi": "inner_left", "xmin": bp1, "xmax": bp2, "m": m_il, "q": b_il},
        {"nimi": "center", "xmin": bp2, "xmax": bp3, "m": m_st, "q": b_st},
        {"nimi": "inner_right", "xmin": bp3, "xmax": bp4, "m": m_ir, "q": b_ir},
        {"nimi": "outer_right", "xmin": bp4, "xmax": np.inf, "m": m_or, "q": b_or},
    ]


def kuulub_vahemikku(x, xmin, xmax, tol=1e-10):
    vasak = (x >= xmin - tol) if np.isfinite(xmin) else True
    parem = (x <= xmax + tol) if np.isfinite(xmax) else True
    return vasak and parem


def leia_koormussirge_loiked(loigud, R4):
    juured = []

    for loik in loigud:
        nimetaja = loik["m"] + 1.0 / R4

        if abs(nimetaja) < 1e-14:
            continue

        u_taht = -loik["q"] / nimetaja

        if kuulub_vahemikku(u_taht, loik["xmin"], loik["xmax"]):
            juured.append({
                "segment": loik["nimi"],
                "u": u_taht,
                "i": -u_taht / R4,
                "slope": loik["m"],
            })

    juured = sorted(juured, key=lambda r: r["u"])
    unikaalsed = []

    for r in juured:
        if not unikaalsed or abs(r["u"] - unikaalsed[-1]["u"]) > 1e-7:
            unikaalsed.append(r)

    return unikaalsed


def klassifitseeri_pusipunktid(juured):
    keskne = [r for r in juured if r["segment"] == "center"]

    if len(keskne) == 0:
        return "muu"

    c = min(keskne, key=lambda r: abs(r["u"]))

    vasak = [r for r in juured if r["u"] < c["u"] - 1e-8]
    parem = [r for r in juured if r["u"] > c["u"] + 1e-8]

    if len(vasak) == 0 and len(parem) == 0:
        return "ainult keskne"

    if len(vasak) == 0 or len(parem) == 0:
        return "muu"

    l = max(vasak, key=lambda r: r["u"])
    r = min(parem, key=lambda r: r["u"])

    lt = "neg" if l["slope"] < 0 else "pos"
    rt = "neg" if r["slope"] < 0 else "pos"

    return f"keskne + vasak {lt}, parem {rt}"


def koonda_vahemikud(R4_vaartused, sildid):
    vahemikud = []
    algus = R4_vaartused[0]
    praegune = sildid[0]

    for i in range(1, len(R4_vaartused)):
        if sildid[i] != praegune:
            vahemikud.append((algus, R4_vaartused[i - 1], praegune))
            algus = R4_vaartused[i]
            praegune = sildid[i]

    vahemikud.append((algus, R4_vaartused[-1], praegune))
    return vahemikud


def skaneeri_pusipunktid(tulemused, R4_min=500, R4_max=3500, samm=1):
    koik_vahemikud = {}

    for opamp, tulemus in tulemused.items():
        loigud = ehita_loigud(tulemus)
        R4_vaartused = np.arange(R4_min, R4_max + samm, samm)
        sildid = []

        for R4 in R4_vaartused:
            juured = leia_koormussirge_loiked(loigud, R4)
            sildid.append(klassifitseeri_pusipunktid(juured))

        koik_vahemikud[opamp] = koonda_vahemikud(R4_vaartused, sildid)

    return koik_vahemikud


def loe_R4_failinimest(fail):
    match = re.search(r"r4_([0-9]*\.?[0-9]+)k", fail.lower())

    if match is None:
        return np.nan

    return float(match.group(1)) * 1000.0


def joonista_bifurkatsioon(opamp, muster):
    R4_punktid = []
    y_punktid = []

    for fail in sorted(glob.glob(muster)):
        R4 = loe_R4_failinimest(fail)

        if not np.isfinite(R4):
            continue

        andmed = np.loadtxt(fail)
        v = andmed[:, 0]
        i = andmed[:, 1]

        algus = int(0.5 * len(v))
        v = v[algus:]
        i = i[algus:]

        indeksid = np.where(i[:-1] * i[1:] < 0)[0]

        if indeksid.size == 0:
            continue

        i0 = i[indeksid]
        i1 = i[indeksid + 1]
        v0 = v[indeksid]
        v1 = v[indeksid + 1]

        kaal = -i0 / (i1 - i0)
        v_nullis = v0 + kaal * (v1 - v0)

        R4_punktid.extend([R4] * len(v_nullis))
        y_punktid.extend(v_nullis.tolist())

    plt.figure(figsize=(9, 5))
    plt.plot(R4_punktid, y_punktid, ".", markersize=0.5, color="maroon")
    plt.xlabel(r"$R_4\,(\Omega)$", fontsize=13)
    plt.ylabel(r"$U_{C_1}|_{I_{R_4}=0}\;(\mathrm{V})$", fontsize=13)
    plt.title(opampi_sildid.get(opamp, opamp))
    plt.tight_layout()
    plt.show()


def loe_signaal(fail):
    andmed = np.loadtxt(fail)
    return andmed[:, 0], andmed[:, 1]


def arvuta_fft(aeg, signaal):
    aeg_uhtlane = np.linspace(aeg[0], aeg[-1], len(aeg))
    signaal_uhtlane = np.interp(aeg_uhtlane, aeg, signaal)
    signaal_uhtlane = signaal_uhtlane - np.mean(signaal_uhtlane)

    dt = aeg_uhtlane[1] - aeg_uhtlane[0]
    sagedused = np.fft.rfftfreq(len(signaal_uhtlane), d=dt)
    amplituudid = 2 * np.abs(np.fft.rfft(signaal_uhtlane)) / len(signaal_uhtlane)

    return sagedused, amplituudid


def leia_pohisagedus(sagedused, amplituudid, fmin=500, fmax=10000):
    mask = (sagedused > fmin) & (sagedused < fmax)

    if not np.any(mask):
        return np.nan

    indeks = np.argmax(amplituudid[mask])
    return sagedused[mask][indeks]


def loe_fdom_failinimi(fail):
    nimi = os.path.splitext(os.path.basename(fail))[0].lower()
    match = re.match(r"(.+)_fdom_r4_([0-9.]+)k$", nimi)

    if match is None:
        return None, None

    opamp = match.group(1)
    R4 = float(match.group(2)) * 1000

    return opamp, R4


def loe_fdom_fail(fail):
    andmed = np.loadtxt(fail)

    aeg = andmed[:, 0]
    signaal = andmed[:, -1]

    return aeg, signaal


def leia_domineeriv_sagedus(aeg, signaal, fmin=500, fmax=10000):
    sagedused, amplituudid = arvuta_fft(aeg, signaal)
    mask = (sagedused >= fmin) & (sagedused <= fmax)

    if not np.any(mask):
        return np.nan

    indeks = np.argmax(amplituudid[mask])
    return sagedused[mask][indeks]


def arvuta_fdom_kaosevahemikes(muster="*_fdom_r4_*.txt", fmin=500, fmax=10000):
    """Leiame iga sobiva R4 faili domineeriva sageduse kaosevahemikus.

    Failinimi peab olema kujul näiteks
    lm324_fdom_r4_1.50k.txt või ad817a_fdom_r4_2.00k.txt.
    """
    read = []

    for fail in sorted(glob.glob(muster)):
        opamp_raw, R4 = loe_fdom_failinimi(fail)
        opamp = normaliseeri_opamp_key(opamp_raw)

        if opamp is None or opamp not in opampide_andmed:
            continue

        info = opampide_andmed[opamp]

        if not (info["r4_min"] <= R4 <= info["r4_max"]):
            continue

        aeg, signaal = loe_fdom_fail(fail)
        f_dom = leia_domineeriv_sagedus(aeg, signaal, fmin=fmin, fmax=fmax)

        read.append({
            "opamp": info["nimi"],
            "opamp_key": opamp,
            "R4_ohm": R4,
            "fail": os.path.basename(fail),
            "f_dom_Hz": f_dom,
        })

    fdom_andmed = pd.DataFrame(read)

    if fdom_andmed.empty:
        fdom_kokkuvote = pd.DataFrame(
            columns=[
                "opamp",
                "opamp_key",
                "f_dom_95_Hz",
                "f_dom_95_kHz",
                "f_dom_min_Hz",
                "f_dom_max_Hz",
            ]
        )
        return fdom_andmed, fdom_kokkuvote

    fdom_andmed = fdom_andmed.sort_values(["opamp", "R4_ohm"])

    fdom_kokkuvote = (
        fdom_andmed
        .groupby(["opamp", "opamp_key"])
        .agg(
            f_dom_95_Hz=("f_dom_Hz", lambda x: np.nanpercentile(x, 95)),
            f_dom_95_kHz=("f_dom_Hz", lambda x: np.nanpercentile(x, 95) / 1000),
            f_dom_min_Hz=("f_dom_Hz", "min"),
            f_dom_max_Hz=("f_dom_Hz", "max"),
        )
        .reset_index()
    )

    return fdom_andmed, fdom_kokkuvote


def joonista_dunaamika(failid):
    pohisagedused = {}

    for opamp, info in failid.items():
        aeg, signaal = loe_signaal(info["signaal"])
        i_faas, u_faas = loe_signaal(info["faas"])

        sagedused, amplituudid = arvuta_fft(aeg, signaal)
        pohisagedused[opamp] = leia_pohisagedus(sagedused, amplituudid)

        fig, teljed = plt.subplots(1, 3, figsize=(17, 5))

        teljed[0].plot(i_faas, u_faas, color="maroon", linewidth=0.7)
        teljed[0].set_xlabel(r"$I_{R4}$ (A)", fontsize=16)
        teljed[0].set_ylabel(r"$U_{C2}$ (V)", fontsize=16)
        teljed[0].grid(True, linestyle="dotted", alpha=0.5)
        teljed[0].xaxis.set_major_locator(MaxNLocator(4))
        teljed[0].yaxis.set_major_locator(MaxNLocator(5))
        teljed[0].xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        teljed[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        teljed[1].plot(aeg, signaal, color="maroon", linewidth=0.7)
        teljed[1].set_xlim(info["aeg_xlim"])
        teljed[1].set_xlabel(r"$t$ (s)", fontsize=16)
        teljed[1].set_ylabel(r"$I_{R4}$ (A)", fontsize=16)
        teljed[1].grid(True, linestyle="dotted", alpha=0.5)

        teljed[2].plot(sagedused, amplituudid, color="maroon", linewidth=0.7)
        teljed[2].set_xlim(0, 5000)
        teljed[2].set_xlabel(r"$f$ (Hz)", fontsize=16)
        teljed[2].set_ylabel(r"$|Y(f)|$ (A)", fontsize=16)
        teljed[2].grid(True, linestyle="dotted", alpha=0.5)

        plt.tight_layout()
        plt.show()

    return pohisagedused


def R4_silt_failist(fail):
    match = re.search(r"(\d+(?:\.\d+)?)k", Path(fail).stem)

    if match:
        R4 = int(round(float(match.group(1)) * 1000))
        return rf"$R_4 = {R4}\,\Omega$"

    return Path(fail).stem


def joonista_arenguportreed(failid):
    fig, teljed = plt.subplots(3, 3, figsize=(14, 12))
    teljed = teljed.ravel()

    for ax, fail in zip(teljed, failid):
        andmed = np.loadtxt(fail)
        x = andmed[:, 0]
        y = andmed[:, 1]

        ax.plot(x, y, linewidth=0.5, color="maroon")
        ax.set_title(R4_silt_failist(fail))
        ax.set_xlabel(r"$I_{R4}$ (A)", fontsize=13)
        ax.set_ylabel(r"$U_{C2}$ (V)", fontsize=13)
        ax.grid(True, linestyle="dotted", alpha=0.5)

        ax.xaxis.set_major_locator(MaxNLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(2))

        vormindaja = ScalarFormatter(useMathText=True)
        vormindaja.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(vormindaja)
        ax.yaxis.set_major_formatter(vormindaja)

    for ax in teljed[len(failid):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def loe_opamp_ja_R4(fail):
    nimi = os.path.splitext(os.path.basename(fail))[0].lower()
    match = re.match(r"(.+)_r4_([0-9.]+)(k?)$", nimi)

    if match is None:
        return None, None

    opamp = match.group(1)
    R4 = float(match.group(2))

    if match.group(3) == "k":
        R4 *= 1000

    return opamp, R4


def arvuta_amplituudid(vahemikud):
    read = []

    for fail in glob.glob("*.txt"):
        opamp, R4 = loe_opamp_ja_R4(fail)

        if opamp is None or opamp not in vahemikud:
            continue

        Rmin, Rmax = vahemikud[opamp]

        if not (Rmin <= R4 <= Rmax):
            continue

        andmed = np.loadtxt(fail)
        i = andmed[:, 0]
        u = andmed[:, 1]

        read.append({
            "opamp": opamp.upper(),
            "R4_ohm": R4,
            "fail": fail,
            "u_min": np.min(u),
            "u_max": np.max(u),
            "u_pp": np.max(u) - np.min(u),
            "u_amp": 0.5 * (np.max(u) - np.min(u)),
            "i_min": np.min(i),
            "i_max": np.max(i),
            "i_pp": np.max(i) - np.min(i),
            "i_amp": 0.5 * (np.max(i) - np.min(i)),
        })

    tulemused = pd.DataFrame(read).sort_values(["opamp", "R4_ohm"])

    kokkuvote = (
        tulemused
        .groupby("opamp")
        .agg(
            R4_min=("R4_ohm", "min"),
            R4_max=("R4_ohm", "max"),
            u_pp_max=("u_pp", "max"),
            i_pp_max=("i_pp", "max"),
        )
        .reset_index()
    )

    return tulemused, kokkuvote


def arvuta_dunaamilised_omadused(fdom_kokkuvote=None):
    """Koostab dünaamiliste piirangute koondtabeli.

    Siia on toodud juurde arvutused eraldi failist:
    - efektiivne ribalaius f_BW_eff = GBW / A_cl;
    - f_dom 95-protsentiil kaosevahemiku simulatsioonidest;
    - ribalaiuse varutegur f_BW_eff / f_dom,95;
    - SR-i simulatsioonipõhine hinnang ja eta_SR.
    """
    A_cl_1 = 1 + R3 / R5
    A_cl_2 = 1 + R7 / R6
    A_cl = max(A_cl_1, A_cl_2)

    read = []

    for opamp, info in opampide_andmed.items():
        f_dom_95 = np.nan
        f_dom_95_kHz = np.nan

        if fdom_kokkuvote is not None and not fdom_kokkuvote.empty:
            alam = fdom_kokkuvote[fdom_kokkuvote["opamp_key"] == opamp]

            if len(alam) > 0:
                f_dom_95 = float(alam["f_dom_95_Hz"].iloc[0])
                f_dom_95_kHz = float(alam["f_dom_95_kHz"].iloc[0])

        f_bw_eff = info["gbw_hz"] / A_cl
        bw_suhe = f_bw_eff / f_dom_95 if np.isfinite(f_dom_95) and f_dom_95 != 0 else np.nan

        read.append({
            "Op-võimendi": info["nimi"],
            "opamp_key": opamp,
            "A_cl": A_cl,
            "epsilon_A_percent": info["epsilon_A_percent"],
            "GBW_MHz": info["gbw_hz"] / 1e6,
            "f_BW_eff_Hz": f_bw_eff,
            "f_BW_eff_kHz": f_bw_eff / 1000,
            "f_dom_95_Hz": f_dom_95,
            "f_dom_95_kHz": f_dom_95_kHz,
            "f_BW_eff_over_f_dom_95": bw_suhe,
            "SR_sim_99.9_V_per_us": info["sr_sim_999"],
            "eta_SR": info["eta_sr"],
        })

    return pd.DataFrame(read)


def prindi_latexi_dunaamilised_read(tabel):
    """Prindib dünaamiliste omaduste tabeli read LaTeXi jaoks."""
    print("\nLaTeXi tabeliread dünaamiliste omaduste jaoks:\n")

    for _, rida in tabel.iterrows():
        print(
            f"{rida['Op-võimendi']} & "
            f"{rida['epsilon_A_percent']:.3f} & "
            f"{rida['f_BW_eff_kHz']:.1f} & "
            f"{rida['f_BW_eff_over_f_dom_95']:.1f} & "
            f"{rida['SR_sim_99.9_V_per_us']:.3f} & "
            f"{rida['eta_SR']:.2f} \\\\"
        )


def leia_sr_R4(fail):
    match = re.search(r"r4_([0-9.]+)k", fail.lower())
    return float(match.group(1)) * 1000 if match else np.nan


def arvuta_sr_failidest(muster, opamp):
    read = []

    for fail in sorted(glob.glob(muster)):
        andmed = np.loadtxt(fail)

        t = andmed[:, 0]
        u1 = andmed[:, 1]
        u2 = andmed[:, 3]

        dt = np.diff(t)

        sr1 = np.abs(np.diff(u1) / dt) / 1e6
        sr2 = np.abs(np.diff(u2) / dt) / 1e6

        sr1 = sr1[np.isfinite(sr1)]
        sr2 = sr2[np.isfinite(sr2)]

        read.append({
            "fail": os.path.basename(fail),
            "R4": leia_sr_R4(fail),
            "SR_XU1_p999": np.percentile(sr1, 99.9),
            "SR_XU2_p999": np.percentile(sr2, 99.9),
            "SR_max_p999": max(np.percentile(sr1, 99.9), np.percentile(sr2, 99.9)),
            "SR_XU1_max": np.max(sr1),
            "SR_XU2_max": np.max(sr2),
            "SR_max_abs": max(np.max(sr1), np.max(sr2)),
        })

    df = pd.DataFrame(read).sort_values("R4")

    kokkuvote = pd.DataFrame([{
        "opamp": opamp,
        "SR_sim_p999_max": df["SR_max_p999"].max(),
        "SR_sim_abs_max": df["SR_max_abs"].max(),
        "SR_sim_p999_mean": df["SR_max_p999"].mean(),
    }])

    return df, kokkuvote


# Arvutused ja joonised

# IV-karakteristikud ja nendest saadud koondparameetrid
joonista_yks_iv(iv_failid["ad704"], "AD704")
joonista_koik_iv(iv_failid)

iv_tulemused, iv_koond, iv_sobitus = analyysi_iv_koondtabel(iv_failid)
pusipunktide_vahemikud = skaneeri_pusipunktid(iv_tulemused)

# Bifurkatsioon, näitedünaamika ja faasiruumide areng
joonista_bifurkatsioon("op07", "op07_r4_*.txt")
pohisagedused = joonista_dunaamika(dunaamika_failid)
joonista_arenguportreed(arengu_failid)

# Uus osa: domineerivad sagedused kaosevahemikes ja dünaamiliste piirangute tabel
fdom_andmed, fdom_kokkuvote = arvuta_fdom_kaosevahemikes(
    muster="*_fdom_r4_*.txt",
    fmin=500,
    fmax=10000,
)

print("\nDomineerivad sagedused iga R4 väärtuse korral:\n")
print(fdom_andmed.to_string(index=False))

print("\nf_dom 95-protsentiili kokkuvõte:\n")
print(fdom_kokkuvote.to_string(index=False))

amplituudid, amplituudide_kokkuvote = arvuta_amplituudid(amplituudi_vahemikud)
dunaamilised_omadused = arvuta_dunaamilised_omadused(fdom_kokkuvote)

print("\nUuendatud dünaamiliste piirangute tabel:\n")
print(dunaamilised_omadused.to_string(index=False))
prindi_latexi_dunaamilised_read(dunaamilised_omadused)

# SR arvutus konkreetsetest transient-failidest, siin näitena OP07
sr_tulemused, sr_kokkuvote = arvuta_sr_failidest("op07_sr_r4_*.txt", "OP07")

SR_makro = 0.595
SR_varutegur = SR_makro / sr_kokkuvote.loc[0, "SR_sim_p999_max"]

print("\nOP07 SR varutegur:\n")
print(SR_varutegur)
