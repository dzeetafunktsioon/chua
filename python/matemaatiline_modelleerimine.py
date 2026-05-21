import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Konstandid ja parameetrid

US = 9.0

C1 = 10e-9
C2 = 100e-9
L1 = 10e-3
L2 = 10e-3

R1 = 22e3
R2 = 220
R3 = 22e3
R5 = 3.3e3
R6 = 2.2e3
R7 = 220

B1 = (R5 / (R3 + R5)) * US
B2 = (R6 / (R7 + R6)) * US

m0 = -1 / R5 - 1 / R6
m1 = 1 / R1 - 1 / R6
m2 = 1 / R1 + 1 / R2

q1 = (m1 - m0) * B1
q2 = (m2 - m1) * B2 + q1

alpha = C2 / C1
b1 = B1 / US
b2 = B2 / US

fB1 = m0 * B1
fB2 = m1 * B2 - q1

R4_kr1 = -B1 / fB1
R4_kr2 = -B2 / fB2

# Funktsioonid

def chua_iv(u):
    return (
        m2 * u
        + 0.5 * (m1 - m2) * (np.abs(u + B2) - np.abs(u - B2))
        + 0.5 * (m0 - m1) * (np.abs(u + B1) - np.abs(u - B1))
    )

def chua_iv_scalar(u):
    if u < -B2:
        return m2 * u + q2
    if u < -B1:
        return m1 * u + q1
    if u <= B1:
        return m0 * u
    if u <= B2:
        return m1 * u - q1
    return m2 * u - q2

def beta(R4):
    return R4**2 * C2 / (L1 + L2)

def dimensioonitud_parameetrid(R4):
    g0 = R4 * m0
    g1 = R4 * m1
    g2 = R4 * m2
    h1 = (g1 - g0) * b1
    h2 = (g2 - g1) * b2 + h1
    return g0, g1, g2, h1, h2

def phi(x, R4):
    g0, g1, g2, h1, h2 = dimensioonitud_parameetrid(R4)
    y = np.empty_like(x)

    mask1 = x < -b2
    mask2 = (x >= -b2) & (x < -b1)
    mask3 = (x >= -b1) & (x <= b1)
    mask4 = (x > b1) & (x <= b2)
    mask5 = x > b2

    y[mask1] = g2 * x[mask1] + h2
    y[mask2] = g1 * x[mask2] + h1
    y[mask3] = g0 * x[mask3]
    y[mask4] = g1 * x[mask4] - h1
    y[mask5] = g2 * x[mask5] - h2

    return y

def jacobi_maatriks(R4, tous):
    return np.array([
        [-alpha * (1 + tous), alpha, 0.0],
        [1.0, -1.0, 1.0],
        [0.0, -beta(R4), 0.0]
    ])

def omavaartused(R4, tous):
    return np.linalg.eigvals(jacobi_maatriks(R4, tous))

def pusipunktid(R4, tol=1e-12):
    g0, g1, g2, h1, h2 = dimensioonitud_parameetrid(R4)

    punktid = [{
        "haru": "keskne",
        "x": 0.0,
        "y": 0.0,
        "tous": g0
    }]

    x_plus = None
    tous_plus = None

    if abs(1 + g1) > tol:
        x_keskmine = h1 / (1 + g1)
        if x_keskmine > b1 + tol and x_keskmine <= b2 + tol:
            x_plus = x_keskmine
            tous_plus = g1

    if x_plus is None and abs(1 + g2) > tol:
        x_valimine = h2 / (1 + g2)
        if x_valimine > b2 + tol:
            x_plus = x_valimine
            tous_plus = g2

    if x_plus is not None:
        punktid.append({
            "haru": "positiivne",
            "x": x_plus,
            "y": 0.0,
            "tous": tous_plus
        })
        punktid.append({
            "haru": "negatiivne",
            "x": -x_plus,
            "y": 0.0,
            "tous": tous_plus
        })

    return punktid

def chua_susteem(t, olek, R4):
    u1, u2, iL = olek
    du1 = ((u2 - u1) / R4 - chua_iv_scalar(u1)) / C1
    du2 = ((u1 - u2) / R4 + iL) / C2
    diL = -u2 / (L1 + L2)
    return [du1, du2, diL]

def chua_susteem_dimensioonitu(tau, olek, R4):
    x, y, z = olek
    dx = alpha * (y - x - phi(np.array([x]), R4)[0])
    dy = x - y + z
    dz = -beta(R4) * y
    return [dx, dy, dz]

def leia_hopf(andmed):
    R = andmed["R4"]
    sigma = andmed["sigma"]
    omega = andmed["omega"]
    kandidaadid = []

    mask = np.isfinite(sigma) & np.isfinite(omega) & (omega > 1e-6)
    Rv = R[mask]
    sv = sigma[mask]
    ov = omega[mask]

    for i in range(len(Rv) - 1):
        if sv[i] * sv[i + 1] < 0:
            Rh = Rv[i] - sv[i] * (Rv[i + 1] - Rv[i]) / (sv[i + 1] - sv[i])
            oh = 0.5 * (ov[i] + ov[i + 1])
            kandidaadid.append((Rh, oh))

    return kandidaadid

def kompleksse_paari_andmed(eig):
    eig = np.array(eig)
    indeksid = np.where(np.abs(np.imag(eig)) > 1e-9)[0]

    if len(indeksid) < 2:
        return np.nan, np.nan

    paar = eig[indeksid[:2]]
    sigma = np.mean(np.real(paar))
    omega = np.mean(np.abs(np.imag(paar)))
    return sigma, omega

def fft_spekter(aeg, signaal):
    aeg = np.asarray(aeg)
    signaal = np.asarray(signaal)

    aeg_uhtlane = np.linspace(aeg[0], aeg[-1], len(aeg))
    signaal_uhtlane = np.interp(aeg_uhtlane, aeg, signaal)
    signaal_uhtlane = signaal_uhtlane - np.mean(signaal_uhtlane)

    dt = aeg_uhtlane[1] - aeg_uhtlane[0]
    sagedused = np.fft.rfftfreq(len(signaal_uhtlane), d=dt)
    amplituudid = np.abs(np.fft.rfft(signaal_uhtlane)) / len(signaal_uhtlane)

    return sagedused, amplituudid

# Pinge-voolu karakteristik

u = np.linspace(-9, 9, 2000)
i = chua_iv(u)

plt.figure(figsize=(8, 5))
plt.plot(u, i, linewidth=0.7)
plt.xlabel(r"$u_1\;(\mathrm{V})$", fontsize=13)
plt.ylabel(r"$f(u_1)\;(\mathrm{A})$", fontsize=13)
plt.grid(True, linestyle="dotted", alpha=0.5)
plt.xlim(-9, 9)
plt.tight_layout()
plt.show()

# Dimensioonitud karakteristikud

x = np.linspace(-1.5, 1.5, 2000)
R4_vaartused = [800, 1300, 1800, 2500]

plt.figure(figsize=(8, 5))

for R4 in R4_vaartused:
    plt.plot(x, phi(x, R4), linewidth=1, label=fr"$R_4={R4}\,\Omega$")

for murdepunkt in [-b2, -b1, b1, b2]:
    plt.axvline(murdepunkt, linestyle="--", linewidth=0.8, alpha=0.8)

plt.xlabel(r"$x$", fontsize=13)
plt.ylabel(r"$\varphi(x)$", fontsize=13)
plt.legend()
plt.xlim(-1.0, 1.0)
plt.ylim(-1.5, 1.5)
plt.grid(True, linestyle="dotted", alpha=0.5)
plt.tight_layout()
plt.show()

# Püsipunktid ja stabiilsus

R4_vahemik = np.arange(500, 3001, 1)

andmed = {
    "keskne": {"R4": [], "u1": [], "max_re": [], "sigma": [], "omega": []},
    "positiivne": {"R4": [], "u1": [], "max_re": [], "sigma": [], "omega": []},
    "negatiivne": {"R4": [], "u1": [], "max_re": [], "sigma": [], "omega": []},
}

hyperboolsus = []

for R4 in R4_vahemik:
    vahed = []

    for punkt in pusipunktid(R4):
        eig = omavaartused(R4, punkt["tous"])
        sigma, omega = kompleksse_paari_andmed(eig)
        haru = punkt["haru"]

        andmed[haru]["R4"].append(R4)
        andmed[haru]["u1"].append(US * punkt["x"])
        andmed[haru]["max_re"].append(np.max(np.real(eig)))
        andmed[haru]["sigma"].append(sigma)
        andmed[haru]["omega"].append(omega)

        vahed.append(np.min(np.abs(np.real(eig))))

    hyperboolsus.append(np.min(vahed))

for haru in andmed:
    for voti in andmed[haru]:
        andmed[haru][voti] = np.array(andmed[haru][voti])

hyperboolsus = np.array(hyperboolsus)

hopf_keskne = leia_hopf(andmed["keskne"])
hopf_valimine = leia_hopf(andmed["positiivne"])

fig, teljed = plt.subplots(1, 2, figsize=(13.5, 4.8))

teljed[0].plot(andmed["keskne"]["R4"], andmed["keskne"]["u1"], linewidth=1, label=r"$P_0$")
teljed[0].plot(andmed["positiivne"]["R4"], andmed["positiivne"]["u1"], linewidth=1, label=r"$P_+$")
teljed[0].plot(andmed["negatiivne"]["R4"], andmed["negatiivne"]["u1"], linewidth=1, label=r"$P_-$")
teljed[0].axvline(R4_kr1, linestyle="--", linewidth=1, label=r"$R_{4,\mathrm{kr},1}$")
teljed[0].axvline(R4_kr2, linestyle="--", linewidth=1, label=r"$R_{4,\mathrm{kr},2}$")
teljed[0].set_xlabel(r"$R_4\,(\Omega)$", fontsize=13)
teljed[0].set_ylabel(r"$u_1^\ast\,(\mathrm{V})$", fontsize=13)
teljed[0].grid(True, linestyle="dotted", alpha=0.5)
teljed[0].legend()

teljed[1].plot(R4_vahemik, hyperboolsus, linewidth=1)
teljed[1].axhline(0, linestyle="--", linewidth=1)
teljed[1].axvline(R4_kr1, linestyle="--", linewidth=1)
teljed[1].axvline(R4_kr2, linestyle="--", linewidth=1)
teljed[1].set_xlabel(r"$R_4\,(\Omega)$", fontsize=13)
teljed[1].set_ylabel(r"$\min_i|\mathrm{Re}(\lambda_i)|$", fontsize=13)
teljed[1].grid(True, linestyle="dotted", alpha=0.5)

fig.tight_layout()
plt.show()

fig, teljed = plt.subplots(1, 2, figsize=(13.5, 4.8))

teljed[0].plot(andmed["keskne"]["R4"], andmed["keskne"]["max_re"], linewidth=1, label=r"$P_0$")
teljed[0].plot(andmed["positiivne"]["R4"], andmed["positiivne"]["max_re"], linewidth=1, label=r"$P_\pm$")
teljed[0].axhline(0, linestyle="--", linewidth=1)
teljed[0].axvline(R4_kr1, linestyle="--", linewidth=1)
teljed[0].axvline(R4_kr2, linestyle="--", linewidth=1)
teljed[0].set_xlabel(r"$R_4\,(\Omega)$", fontsize=13)
teljed[0].set_ylabel(r"$\max_i\mathrm{Re}(\lambda_i)$", fontsize=13)
teljed[0].grid(True, linestyle="dotted", alpha=0.5)
teljed[0].legend()

teljed[1].plot(andmed["keskne"]["R4"], andmed["keskne"]["sigma"], linewidth=1, label=r"$P_0$")
teljed[1].plot(andmed["positiivne"]["R4"], andmed["positiivne"]["sigma"], linewidth=1, label=r"$P_\pm$")
teljed[1].axhline(0, linestyle="--", linewidth=1)
teljed[1].axvline(R4_kr1, linestyle="--", linewidth=1)
teljed[1].axvline(R4_kr2, linestyle="--", linewidth=1)

for Rh, oh in hopf_keskne:
    teljed[1].plot(Rh, 0, "o", markersize=4)

for Rh, oh in hopf_valimine:
    teljed[1].plot(Rh, 0, "o", markersize=4)

teljed[1].set_xlabel(r"$R_4\,(\Omega)$", fontsize=13)
teljed[1].set_ylabel(r"$\sigma=\mathrm{Re}(\lambda_{1,2})$", fontsize=13)
teljed[1].grid(True, linestyle="dotted", alpha=0.5)
teljed[1].legend()

fig.tight_layout()
plt.show()

# Bifurkatsioonidiagramm

R4_vaartused = np.arange(500, 2501, 10)
t_max = 100e-3
punktide_arv = 30000
algolek = [0.1, 0.0, 0.0]

R4_punktid = []
voolu_punktid = []

for R4 in R4_vaartused:
    t_eval = np.linspace(0, t_max, punktide_arv)

    lahend = solve_ivp(
        chua_susteem,
        (0, t_max),
        algolek,
        args=(R4,),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    u1 = lahend.y[0]
    u2 = lahend.y[1]

    algus = int(0.6 * len(u1))
    u1 = u1[algus:]
    u2 = u2[algus:]

    indeksid = np.where(u1[:-1] * u1[1:] < 0)[0]

    if indeksid.size > 0:
        u10 = u1[indeksid]
        u11 = u1[indeksid + 1]
        u20 = u2[indeksid]
        u21 = u2[indeksid + 1]

        kaal = -u10 / (u11 - u10)
        u2_loikes = u20 + kaal * (u21 - u20)
        iR4_loikes = u2_loikes / R4

        R4_punktid.extend([R4] * len(iR4_loikes))
        voolu_punktid.extend(iR4_loikes.tolist())

plt.figure(figsize=(9, 5))
plt.plot(R4_punktid, voolu_punktid, ".", markersize=0.5)
plt.xlabel(r"$R_4\,(\Omega)$", fontsize=13)
plt.ylabel(r"$i_{R_4}|_{u_1=0}\;(\mathrm{A})$", fontsize=13)
plt.tight_layout()
plt.show()

# Koormussirged

R4_vaartused = [1320.0, 1950.0, 2030.0, 2075.0]
u = np.linspace(-10, 10, 3000)

for R4 in R4_vaartused:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(u, chua_iv(u), linewidth=0.8, label=r"$f(u_1)$")
    ax.plot(u, -u / R4, linestyle="--", linewidth=0.8, label=rf"$i=-u_1/R_4$")

    for punkt in pusipunktid(R4):
        u_taht = US * punkt["x"]
        i_taht = -u_taht / R4
        ax.plot(u_taht, i_taht, "o", markersize=5)

    ax.axhline(0, linewidth=0.6, alpha=0.5)
    ax.axvline(0, linewidth=0.6, alpha=0.5)
    ax.set_xlabel(r"$u_1\;(\mathrm{V})$", fontsize=13)
    ax.set_ylabel(r"$i\;(\mathrm{A})$", fontsize=13)
    ax.grid(True, linestyle="dotted", alpha=0.3)
    ax.set_xlim(-10, 10)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Dünaamilised režiimid

R4 = 1320.0
tau_max = 80.0
punktide_arv = 5000
algolek = [0.1, 0.0, 0.0]

tau_eval = np.linspace(0, tau_max, punktide_arv)

lahend = solve_ivp(
    chua_susteem_dimensioonitu,
    (0, tau_max),
    algolek,
    args=(R4,),
    t_eval=tau_eval,
    rtol=1e-8,
    atol=1e-10
)

tau = lahend.t
x = lahend.y[0]
y = lahend.y[1]

sagedused, amplituudid = fft_spekter(tau, x)

fig, teljed = plt.subplots(1, 3, figsize=(17, 5))

teljed[0].plot(x, y, linewidth=0.5)
teljed[0].set_xlabel(r"$x$", fontsize=16)
teljed[0].set_ylabel(r"$y$", fontsize=16)
teljed[0].grid(True, linestyle="dotted", alpha=0.5)

teljed[1].plot(tau, x, linewidth=0.7)
teljed[1].set_xlabel(r"$\tau$", fontsize=16)
teljed[1].set_ylabel(r"$x(\tau)$", fontsize=16)
teljed[1].grid(True, linestyle="dotted", alpha=0.5)

teljed[2].plot(sagedused, amplituudid, linewidth=0.7)
teljed[2].set_xlim(0, 3)
teljed[2].set_xlabel(r"$f$", fontsize=16)
teljed[2].set_ylabel(r"$|Y(f)|$", fontsize=16)
teljed[2].grid(True, linestyle="dotted", alpha=0.5)

plt.tight_layout()
plt.show()

# Faasiruumi areng

R4_vaartused = [2069.0, 2070.0, 2071.0, 2075.0, 2077.0, 2080.0, 2090.0, 2170.0, 3000.0]
tau_max = 50.0
punktide_arv = 5000
algolek = [0.1, 0.0, 0.0]

tulemused = []

for R4 in R4_vaartused:
    tau_eval = np.linspace(0, tau_max, punktide_arv)

    lahend = solve_ivp(
        chua_susteem_dimensioonitu,
        (0, tau_max),
        algolek,
        args=(R4,),
        t_eval=tau_eval,
        rtol=1e-8,
        atol=1e-10
    )

    tulemused.append({
        "R4": R4,
        "x": lahend.y[0],
        "y": lahend.y[1],
        "punktid": pusipunktid(R4)
    })

fig, teljed = plt.subplots(3, 3, figsize=(14, 12))
teljed = teljed.ravel()

for ax, tulemus in zip(teljed, tulemused):
    ax.plot(tulemus["x"], tulemus["y"], linewidth=0.5)

    for punkt in tulemus["punktid"]:
        ax.plot(punkt["x"], punkt["y"], "o", markersize=4)

    ax.set_title(rf"$R_4={tulemus['R4']:.0f}\,\Omega$")
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.grid(True, linestyle="dotted", alpha=0.5)

plt.tight_layout()
plt.show()
