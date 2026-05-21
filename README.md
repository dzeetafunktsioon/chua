# Chua vooluahela simulatsioonid

See repository sisaldab bakalaureusetöö **„Operatsioonivõimendite mõju kaose dünaamikale Chua vooluahelas“** lisamaterjale.

Töös uuritakse, kuidas operatsioonivõimendi valik mõjutab Chua dioodi voolu-pinge karakteristikut ja Chua vooluahela kaootilist dünaamikat. Repository sisaldab tööga seotud Python koode ja ngspice sisendfaile.

# Sisu

- `python/` – Python koodid matemaatiliseks modelleerimiseks ja simulatsiooniandmete analüüsiks
- `ngspice/` – ngspice sisendfailid Chua vooluahela ja operatsioonivõimendite omaduste simuleerimiseks

# Python koodid

Kaustas `python/` on töö arvutus- ja analüüsikoodid.

- `matemaatiline_modelleerimine.py` – Chua vooluahela matemaatilise mudeli analüüs
- `simulatsiooniandmete_analyys.py` – ngspice simulatsiooniandmete analüüs ja jooniste koostamine

Vajalikud Python paketid:

```bash
pip install numpy pandas matplotlib
```

Python faili käivitamiseks:

```bash
python3 python/matemaatiline_modelleerimine.py
```

või

```bash
python3 python/simulatsiooniandmete_analyys.py
```

või kopeerida Jupyteri.

# ngspice sisendfailid

Kaustas `ngspice/` on simulatsioonide sisendfailid.

- `chua_vooluahel.cir` – Chua vooluahela simulatsioon
- `iv_karakteristik.cir` – Chua dioodi voolu-pinge karakteristiku simulatsioon
- `beta_varieerimine.cir` – takistuse R4 varieerimine
- `avatud_ahela_voimendus.cir` – operatsioonivõimendi avatud ahela võimenduse määramine
- `gbw.cir` – võimenduse-ribalaiuse korrutise määramine
- `sisendnihkepinge.cir` – sisendnihkepinge määramine
- `sisendvool.cir` – sisendvoolu määramine
- `sr.cir` – väljundpinge maksimaalse kasvukiiruse määramine
- `valjundpinge_ulatus.cir` – väljundpinge ulatuse määramine

ngspice faili käivitamiseks:

```bash
ngspice ngspice/chua_vooluahel.cir
```

# Operatsioonivõimendite SPICE mudelfailid

Operatsioonivõimendite SPICE makromudeleid selles repository’s kaasas ei ole. Need tuleb eraldi alla laadida tootjate kodulehtedelt ja panna lokaalselt näiteks kausta `models/`.

Simulatsioonides kasutatud operatsioonivõimendid:

- AD704 – Analog Devices
- AD817A – Analog Devices
- LM324 – ON Semiconductor
- OP07 – Texas Instruments
- OPA445 – Texas Instruments

Mudelfailid on tavaliselt leitavad tootja tootelehelt.

Kui mudelfailid asuvad kaustas `models/`, tuleb `.cir` failides kasutada näiteks selliseid `.include` ridu:

```spice
.include ../models/AD704.lib
.include ../models/AD817A.lib
.include ../models/LM324.sub
.include ../models/OP07.lib
.include ../models/OPA445.lib
```

## Simulatsiooniandmed

Python analüüsikood eeldab, et ngspice simulatsioonidest saadud `.txt` väljundfailid on olemas ja nende nimed vastavad koodis kasutatud failinimedele. Kui failinimed või asukohad muutuvad, tuleb neid muuta ka Python koodis.

## Reprodutseerimiseks vajalik

Tulemuste uuesti saamiseks on vaja:

1. Pythonit koos pakettidega `numpy`, `pandas` ja `matplotlib`;
2. ngspice tarkvara;
3. tootjate operatsioonivõimendite SPICE makromudeleid;
4. simulatsioonidest saadud väljundandmeid.

## Autor

Annette Bogdanov  
Bakalaureusetöö  
Tartu Ülikool
