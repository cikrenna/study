# Patches alto contrasto (CAMELS LH, snapshot 090)

Variante "alto contrasto" del notebook standard `make_input_patches.ipynb` (NB1): genera
l'input per la U-Net basato sul rapporto SPH **T = I1/I2**, che sopprime il background
diffuso ed enfatizza i picchi (subhalo). Salva **solo** in
`input_data/sim_<RUN>_<TAG>_highcontrast/` e non tocca mai le cartelle originali `sim_<id>/`.

    snapshot + h (cache)
      → 25 fette lungo z
      → I1 = Σ m ρ W  (peso = m_dm · ρ_j)      I2 = Σ m W  (peso = m_dm)
      → T = I1/I2    (dove I2 > 0; 0 altrove)
      → normalizzazione log10 + z-score robusto
      → patches 256×256 → input_data/sim_<RUN>_<TAG>_highcontrast/

con $\rho_j = 32\,m_{\mathrm{DM}} / (\tfrac{4}{3}\pi h_j^3)$ densità SPH locale della
particella $j$ ($k = 32$ vicini, §4).

**Nota fisica critica**: T serve **solo per la detection**. Le masse restano calcolate su
I2 --- identico al rendering standard di NB1 --- quindi `mpu_per_slice.npy` è fisicamente
valido per la calibrazione di stadio 2 (NB3).

**Dipendenze dal notebook standard** (autonomo ma non autosufficiente):

-   `input_data/sim_<id>/Y_mask_patches.npy` --- target FoF, **copiato invariato**
    (mai rigenerato: garanzia di identità con NB1);
-   `input_data/sim_<id>/raw_slice_sample.npy` --- usato dalla §10 per verificare che il
    rendering I2 coincida con quello standard (garanzia di validità di `mpu`);
-   `smoothing_lengths_cache/` --- cache $h$ condivisa con NB1 (stessi $k=32$ e validazione MD5).

## Output per simulazione (7 file in `sim_<id>_highcontrast/`)

| File | Contenuto | Shape |
|---|---|---|
| `X_patches_HC.npy` | input: T normalizzata | (400, 1, 256, 256) float32 |
| `Y_mask_patches.npy` | copia del target FoF originale | (400, 1, 256, 256) uint8 |
| `norm_stats_HC.npy` | mediana e sigma della normalizzazione di T | (1, 2) float32 |
| `mpu_per_slice.npy` | massa per unità (da **I2**), calibrazione stadio 2 | (25,) float64 |
| `raw_slice_sample_HC.npy` | fetta `RAW_SAMPLE_SLICE`: T raw pre-normalizzazione | (1024, 1024) float32 |
| `raw_slice_sample_I2.npy` | fetta `RAW_SAMPLE_SLICE`: I2 raw (riferimento per §10) | (1024, 1024) float32 |
| `dataset_info_HC.json` | metadati completi (schema annidato, §7) | --- |

## Struttura del notebook

| Sez. | Contenuto |
|---|---|
| 1 | Setup e configurazione (solo flag HC) |
| 2 | Scoperta delle simulazioni |
| 3 | Funzioni base: MD5, snapshot, smoothing lengths |
| 4 | Densità locale SPH per particella (B2) |
| 5 | Rendering HC I1/I2/T + normalizzazione (B3--B5) |
| 6 | Estrazione patches + utility di ricostruzione |
| 7 | Pipeline HC per simulazione (dataset_info_HC annidato) |
| 8 | Esecuzione su tutte le simulazioni |
| 9 | Verifica dei file di output |
| 10 | Verifica end-to-end (B7): catena T e uguaglianza I2 |
| 11 | Visual check HC: T vs I2 vs diff con overlay FoF/Subhalo |


## 1 --- Setup: librerie, configurazione, comportamento di esecuzione

Config unica fonte di verità, **solo modalità HC**: niente flag morti, niente doppie
versioni. `PIPELINE_VERSION = "1.2.0-hc"` è assegnato una sola volta.

`SKIP_EXISTING = True` non riscrive mai un output completo: se `X_patches_HC.npy` e
`dataset_info_HC.json` esistono già con la versione corrente, la simulazione viene saltata.
Per rigenerare da zero: cancellare `input_data/sim_<id>_highcontrast/` e rieseguire.
Non esiste backfill in HC --- T non è denormalizzabile in I2, quindi i file non sono
incrementali e vanno rigenerati insieme.



```python
# ============================================================
# LIBRERIE + CONFIGURAZIONE (unica fonte di verità, solo HC)
# ============================================================
import os, gc, time, json, hashlib, traceback
from datetime import datetime

import numpy as np
import h5py
import hdf5plugin                                       # richiesto per gli snapshot compressi (bitshuffle)
import pandas as pd
import matplotlib.pyplot as plt

import sphviewer2                                       # type: ignore
from sphviewer2 import estimate_h, render               # type: ignore

# ---------------- versione e modalità ----------------
PIPELINE_VERSION = "1.2.0-hc"
MODE             = "HIGH_CONTRAST"

# ---------------- percorsi ----------------
DATA_DIR    = r"C:\Users\kekko\Downloads\Python\TESI\CAMELS_data"
SIM_ROOT    = os.path.join(DATA_DIR, "Simulations")
OUT_ROOT    = os.path.join(DATA_DIR, "input_data")
H_CACHE_DIR = os.path.join(DATA_DIR, "smoothing_lengths_cache")

# ---------------- selezione dati ----------------
SNAP_TAG = "090"                                        # snapshot a z≈0

# ---------------- rendering SPH ----------------
N_NEIGHBORS_H = 32                                      # vicini per lo smoothing length (e per ρ_j)
TARGET_CELLS  = 4                                       # celle per h nel rendering
R_MAX_LEVEL   = 10                                      # griglia: IMG_SIZE = 2**10 = 1024
IMG_SIZE      = 2 ** R_MAX_LEVEL
N_SLICES      = 25                                      # fette lungo z (Δz = 1 Mpc/h)

# ---------------- label / patches ----------------
N_MIN_PARTICLES = 100                                   # taglio FoF (solo metadato: Y è copiata dall'originale)
MIN_RADIUS_PX   = 1.5                                   # raggio minimo dei cerchi FoF nel visual check [px]
PATCH_SIZE      = 256
EMPTY_FILL      = -3.0                                  # valore dei pixel vuoti/non finiti

# ---------------- gate (solo per eventuale backfill futuro, non usato in HC) ----------------
REG_MIN_DIRECT = 2.0
REG_MIN_RATIO  = 3.0

SEED        = 42                                        # seed del campionamento in normalize_slices
NUM_THREADS = 8

# ---------------- comportamento di esecuzione ----------------
SKIP_EXISTING    = True     # se X_HC + info completi, salta la simulazione
RAW_SAMPLE_SLICE = 12       # fetta di cui salvare T e I2 raw (verifica end-to-end in §10)

# ---------------- suffisso cartelle HC ----------------
SIM_SUFFIX = "_highcontrast"

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(H_CACHE_DIR, exist_ok=True)

# ---------------- riepilogo configurazione ----------------
print(f"Pipeline v{PIPELINE_VERSION} | modalità {MODE} | snapshot {SNAP_TAG}")
print(f"  rendering : {N_SLICES} fette × {IMG_SIZE}² | h da k={N_NEIGHBORS_H} vicini | T = I1/I2")
print(f"  patches   : {PATCH_SIZE}² | griglia {IMG_SIZE // PATCH_SIZE}×{IMG_SIZE // PATCH_SIZE} per fetta")
print(f"  esecuzione: SKIP_EXISTING={SKIP_EXISTING} | raw sample: fetta {RAW_SAMPLE_SLICE}")
print(f"  output    : sim_<id>{SIM_SUFFIX}/ | percorsi: DATA_DIR={DATA_DIR}")

```

    Pipeline v1.2.0-hc | modalità HIGH_CONTRAST | snapshot 090
      rendering : 25 fette × 1024² | h da k=32 vicini | T = I1/I2
      patches   : 256² | griglia 4×4 per fetta
      esecuzione: SKIP_EXISTING=True | raw sample: fetta 12
      output    : sim_<id>_highcontrast/ | percorsi: DATA_DIR=C:\Users\kekko\Downloads\Python\TESI\CAMELS_data
    

## 2 --- Scoperta delle simulazioni

Percorsi costruiti direttamente sul layout fisso di CAMELS
(`<RUN>/snapshots_<RUN>/snapshot_090_<RUN>.hdf5` + `<RUN>/groups_<RUN>/groups_090_<RUN>.hdf5`).
Se un file manca, la simulazione viene saltata con un messaggio esplicito --- non
silenziosamente ignorata. L'ID segue la convenzione `<RUN>_<TAG>`, es. `LH_100_090`
(`RUN` = run cosmologica `LH_100`…`LH_106`, `TAG` = snapshot `090`, $z \approx 0$).

Per il riepilogo serve solo l'Header --- nessuna coordinata viene toccata, costo
trascurabile ($\sim\mathrm{ms}$ per file):

| Campo | Significato | Unità nel file | Conversione |
|---|---|---|---|
| `Redshift` | $z$ dell'output | --- | $\approx 0$ per lo snapshot 090 |
| `Omega0` | $\Omega_{\mathrm{matter}}$ | --- | --- |
| `HubbleParam` | $h = H_0/100$ | --- | --- |
| `BoxSize` | lato della simulazione | $\mathrm{ckpc}/h$ | $\div 1000 \rightarrow \mathrm{Mpc}/h$ |
| `MassTable[1]` | massa particella DM (PartType1) | $10^{10}\,M_\odot/h$ | $\times 10^{10} \rightarrow M_\odot/h$ |
| `NumPart_Total[1]` | numero particelle DM | --- | --- |



```python
# ============================================================
# SCOPERTA DELLE SIMULAZIONI
# ============================================================
def header_summary(snap_path):
    """Legge solo l'Header dello snapshot (nessuna coordinata).
    Ritorna un dict con i campi fisici rilevanti per il riepilogo."""
    with h5py.File(snap_path, "r") as f:
        hd = f["Header"].attrs
        get = lambda name: float(np.ravel(hd[name])[0]) if name in hd else float("nan")
        return {
            "redshift":  get("Redshift"),
            "omega0":    get("Omega0"),
            "hubble":    get("HubbleParam"),
            "box_mpc":   get("BoxSize") / 1000.0,
            "m_dm":      float(np.ravel(hd["MassTable"])[1]) * 1e10,   # 1e10 Msun/h -> Msun/h
            "n_dm":      int(np.ravel(hd["NumPart_Total"])[1]),
        }


# Costruisce il dizionario {sim_id: (snap_path, cat_path)} scansionando SIM_ROOT
SIMULATIONS = {}
for run in sorted(os.listdir(SIM_ROOT)):
    s = os.path.join(SIM_ROOT, run, f"snapshots_{run}", f"snapshot_{SNAP_TAG}_{run}.hdf5")
    c = os.path.join(SIM_ROOT, run, f"groups_{run}",    f"groups_{SNAP_TAG}_{run}.hdf5")
    if os.path.exists(s) and os.path.exists(c):
        SIMULATIONS[f"{run}_{SNAP_TAG}"] = (s, c)
    else:
        print(f"[skip] {run}: layout incompleto")

print(f"\nSimulazioni trovate: {len(SIMULATIONS)}\n")
print(pd.DataFrame([
    {"sim_id": k, **header_summary(s)} for k, (s, _) in SIMULATIONS.items()
]).to_string(index=False))

```

    
    Simulazioni trovate: 7
    
        sim_id     redshift  omega0  hubble  box_mpc         m_dm     n_dm
    LH_100_090 2.220446e-16  0.4590  0.6711     25.0 1.186462e+08 16777216
    LH_101_090 2.220446e-16  0.1430  0.6711     25.0 3.696386e+07 16777216
    LH_102_090 2.220446e-16  0.1194  0.6711     25.0 3.086353e+07 16777216
    LH_103_090 2.220446e-16  0.2486  0.6711     25.0 6.426026e+07 16777216
    LH_104_090 2.220446e-16  0.3138  0.6711     25.0 8.111371e+07 16777216
    LH_105_090 2.220446e-16  0.3726  0.6711     25.0 9.631284e+07 16777216
    LH_106_090 2.220446e-16  0.2850  0.6711     25.0 7.366924e+07 16777216
    

## 3 --- Funzioni base: MD5, caricamento snapshot, smoothing lengths

-   **`file_md5`**: hash del **contenuto** (blocchi da 4 MB, $\sim$500 MB per file) ---
    le run CAMELS hanno nomi file identici tra loro, il nome non basta a identificare
    uno snapshot.
-   **`load_snapshot`**: legge solo `PartType1/Coordinates` (ckpc/h $\rightarrow$ Mpc/h)
    e l'Header. Velocità e `ParticleIDs` **non** vengono letti: non servono alla pipeline
    2D e il file è già compresso (bitshuffle + filtro 32001).
-   **`get_smoothing_lengths`**: `estimate_h` costa $\sim$12 minuti per simulazione
    ($k = 32$ su $16.7\,\mathrm{M}$ particelle, 8 threads). Cache `.npz` per simulazione
    in `smoothing_lengths_cache/` con chiavi `h`, `md5`, `sim_id`, `created`. Validazione:
    MD5 ricalcolato dallo snapshot e confrontato con quello salvato + lunghezza $h$
    attesa; cache invalidata altrimenti. La cache è **condivisa** con NB1 standard
    (stessi parametri $k$, quindi stesso risultato atteso).



```python
# ============================================================
# FUNZIONI BASE: MD5, load_snapshot, get_smoothing_lengths
# ============================================================
def file_md5(path, chunk=1 << 22):
    """MD5 del CONTENUTO del file: le run CAMELS hanno nomi file identici tra loro,
    quindi il nome non basta a identificare uno snapshot."""
    m = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            m.update(block)
    return m.hexdigest()


def load_snapshot(snap_path):
    """Particelle di DM (PartType1 in CAMELS) + quantità utili dall'Header.
    Le coordinate sono in ckpc/h e vengono convertite in Mpc/h."""
    with h5py.File(snap_path, "r") as f:
        pos = f["PartType1/Coordinates"][:].astype(np.float64) / 1000.0   # ckpc/h -> Mpc/h
        hd = f["Header"].attrs
        return {
            "pos":      pos,
            "lbox":     float(np.ravel(hd["BoxSize"])[0]) / 1000.0,        # Mpc/h
            "m_dm":     float(np.ravel(hd["MassTable"])[1]) * 1e10,        # Msun/h
            "redshift": float(np.ravel(hd["Redshift"])[0]),
            "omega0":   float(np.ravel(hd["Omega0"])[0]),
            "hubble":   float(np.ravel(hd["HubbleParam"])[0]),
        }


def get_smoothing_lengths(sim_id, snap_path, pos, lbox):
    """Smoothing length SPH (raggio che contiene k=N_NEIGHBORS_H vicini),
    con cache per-simulazione validata via MD5 del contenuto dello snapshot."""
    cache = os.path.join(H_CACHE_DIR, f"smoothing_lengths_{sim_id}_k{N_NEIGHBORS_H}.npz")
    md5 = file_md5(snap_path)

    # --- tentativo di riuso della cache ---
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            if str(z["md5"]) == md5 and len(z["h"]) == len(pos):
                print(f"[{sim_id}] smoothing lengths: cache valida")
                return z["h"]

    # --- calcolo (lento) + salvataggio ---
    print(f"[{sim_id}] calcolo smoothing lengths (k={N_NEIGHBORS_H})...")
    t0 = time.time()
    x, y, z = (np.ascontiguousarray(pos[:, i], dtype=np.float64) for i in range(3))
    h = np.asarray(
        estimate_h(x, y, z, lbox, k=N_NEIGHBORS_H, num_threads=NUM_THREADS),
        dtype=np.float64,
    )
    np.savez(cache, h=h, md5=md5, sim_id=sim_id,
             created=datetime.now().isoformat(timespec="seconds"))
    print(f"  calcolati in {time.time() - t0:.0f} s")
    return h

```

## 4 --- Densità locale SPH per particella (B2)

$$
\boxed{\;
\rho_j \;=\; \frac{k \cdot m_{\mathrm{DM}}}{\tfrac{4}{3}\,\pi\, h_j^{3}},
\qquad k = \texttt{N\_NEIGHBORS\_H} = 32
\;}
$$

$h_j$ è il raggio che contiene $k$ vicini --- esattamente la quantità calcolata da
`estimate_h` in §3: la formula è quindi la stima SPH standard della densità locale di
ogni particella. Unità: $m_{\mathrm{DM}}$ in $M_\odot/h$, $h$ in $\mathrm{Mpc}/h$
$\rightarrow$ $\rho_j$ in $M_\odot/h\,/\,\mathrm{Mpc}^3$ (arbitrarie ma consistenti; in
$T = I_1/I_2$ l'unità si semplifica nel rapporto).

Il peso di I1 in §5 è $m_{\mathrm{DM}} \cdot \rho_j$: questa è la funzione che lo genera.



```python
# ============================================================
# B2 — DENSITÀ LOCALE SPH PER PARTICELLA (per I1)
# ============================================================
def compute_rho_particles(h, m_dm, k=N_NEIGHBORS_H):
    """Densità locale SPH di ogni particella:
       rho_j = k * m_j / ((4/3) * pi * h_j^3)
    Unità: m_dm in Msun/h, h in Mpc/h → rho_j in Msun/h / Mpc^3 (arbitrarie ma consistenti).
    """
    return (k * m_dm) / ((4.0 / 3.0) * np.pi * h**3)

```

## 5 --- Rendering alto contrasto e normalizzazione (B3--B5)

### 5.1 --- `render_slices_hc`: I1, I2, T = I1/I2

Per ogni fetta $k$ si selezionano le particelle con
$z_i \in [\,k\,\Delta z,\; (k+1)\,\Delta z\,)$. $\Delta z = L_{\text{box}}/25 = 1\,\mathrm{Mpc}/h$
e si proiettano sul piano $x$--$y$ **due** quantità, con lo stesso kernel SPH periodico
in $(x,y)$ e **gli stessi parametri** del rendering standard di NB1:

$$
I_1(x,y) = \sum_j m_j\,\rho_j\,W_j
\qquad\qquad
I_2(x,y) = \sum_j m_j\,W_j
$$

-   **I2** usa pesi `np.full(n, m_dm)`: è **identico** al rendering standard di NB1.
    Da qui $\mathrm{mpu}_k = n_k \cdot m_{\mathrm{DM}} / \sum_{(x,y)} I_2$
    ($M_\odot/h$ per unità di somma immagine, per fetta perché $m_{\mathrm{DM}}$ cambia
    tra le run LH): la fisica delle masse è invariata. L'uguaglianza I2_HC = I2_standard
    è verificata numericamente in §10 --- se fallisce, `mpu` non è valido.
-   **I1** usa pesi `m_dm * rho_j` (vettore di pesi come 5° argomento di `render`;
    il rendering è verificato lineare nei pesi).
-   **T = I1/I2** dove $I_2 > 0$, 0 altrove: $T$ è la media locale di $\rho_j$ pesata per
    massa --- nei picchi (subhalo, $\rho_j$ alto) emerge, nel background diffuso
    ($\rho_j$ basso) viene soppresso. È l'input della U-Net, **solo per detection**.
-   La direzione $z$ **non** è periodica: ogni particella appartiene a una sola fetta,
    mai le stesse due volte --- la somma delle masse per fetta è ben definita (NB3).

### 5.2 --- `normalize_slices`: log10 + z-score robusto

Il campo copre $\sim$6 decadi di gamma dinamica: una U-Net non può lavorarci direttamente.

$$
\text{norm} \;=\; \frac{\log_{10} T - \operatorname{med}(\log_{10} T)}
{\dfrac{P_{84}(\log_{10} T) - P_{16}(\log_{10} T)}{2}}
$$

mediana e semi-larghezza $16$--$84$ stimate su un campione casuale ($\leq 262144$ pixel,
seed fisso 42). Lo z-score è **robusto** (non classico) perché il campo ha una coda destra
lunga: media e deviazione standard sarebbero dominate da pochi pixel densissimi.

I pixel vuoti ($T \leq 0$, cioè $I_2 = 0$) o non finiti vanno a
`EMPTY_FILL = -3.0` ($\sim3\sigma$ sotto la mediana, distinguibile dal bulk $\sim 0$).
NB3-HC rimetterà $T = 0$ (pixel vuoto) con `X == EMPTY_FILL` prima di ogni calcolo di massa.



```python
# ============================================================
# B3 + B4 — RENDERING ALTO CONTRASTO: I1, I2, T = I1/I2
# ============================================================
def render_slices_hc(pos, h, rho_j, lbox, m_dm):
    """Proiezione SPH per fetta di I1 = Σ m ρ W e I2 = Σ m W, poi T = I1/I2.
    I1 e I2 usano gli STESSI parametri del rendering standard di NB1
    (l'uguaglianza I2_HC = I2_standard è verificata in §10).
    Ritorna (I1_slices, I2_slices, T_slices, mpu_per_slice)."""
    dz = lbox / N_SLICES
    I1_slices = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    I2_slices = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    mpu = np.full(N_SLICES, np.nan)

    for i in range(N_SLICES):
        sel = (pos[:, 2] >= i * dz) & (pos[:, 2] < (i + 1) * dz)
        n = int(sel.sum())
        if n == 0:
            continue
        x = np.ascontiguousarray(pos[sel, 0])
        y = np.ascontiguousarray(pos[sel, 1])
        z = np.zeros(n, dtype=np.float64)
        h_sel = np.ascontiguousarray(h[sel])

        # --- I1 = Σ m ρ W (peso = m_dm * rho_j) ---
        img1, _ = render(
            x, y, z, h_sel, np.ascontiguousarray(m_dm * rho_j[sel]),
            Lbox=lbox, extent=lbox, xc=lbox / 2, yc=lbox / 2, zc=0.0,
            azimuth=0.0, elevation=0.0, roll=0.0, periodic=True,
            r_max=R_MAX_LEVEL, target_cells_per_h=TARGET_CELLS, num_threads=NUM_THREADS,
        )
        I1_slices[i] = np.asarray(img1, dtype=np.float32)

        # --- I2 = Σ m W (identico al rendering standard di NB1) ---
        img2, _ = render(
            x, y, z, h_sel, np.full(n, m_dm),
            Lbox=lbox, extent=lbox, xc=lbox / 2, yc=lbox / 2, zc=0.0,
            azimuth=0.0, elevation=0.0, roll=0.0, periodic=True,
            r_max=R_MAX_LEVEL, target_cells_per_h=TARGET_CELLS, num_threads=NUM_THREADS,
        )
        I2_slices[i] = np.asarray(img2, dtype=np.float32)

        # mpu da I2 (fisica delle masse invariata)
        s2 = float(I2_slices[i].sum(dtype=np.float64))
        if s2 > 0:
            mpu[i] = n * m_dm / s2
        print(f"  fetta {i + 1}/{N_SLICES}", end="\r")
    print()

    # --- T = I1 / I2 (dove I2 > 0; altrimenti 0) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        T_slices = np.where(I2_slices > 0, I1_slices / I2_slices, 0.0).astype(np.float32)

    return I1_slices, I2_slices, T_slices, mpu


def normalize_slices(slices):
    """log10 + z-score robusto (mediana e semi-larghezza 16-84, campionate con seed fisso).
    I pixel vuoti o non finiti ricevono EMPTY_FILL."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_d = np.where(slices > 0, np.log10(slices), np.nan)
    vals = log_d[np.isfinite(log_d)]
    if len(vals) == 0:
        raise RuntimeError("Il campo di densità non contiene valori finiti.")

    rng = np.random.default_rng(SEED)
    sample = vals[rng.choice(len(vals), size=min(262144, len(vals)), replace=False)]
    p16, med, p84 = np.percentile(sample, [16, 50, 84])
    sig = (p84 - p16) / 2.0
    if not np.isfinite(sig) or sig <= 0:
        raise RuntimeError(f"Scala di normalizzazione non valida: sigma={sig}")

    norm = (log_d - med) / sig
    norm[~np.isfinite(norm)] = EMPTY_FILL
    return norm.astype(np.float32), float(med), float(sig)

```

## 6 --- Estrazione patches e utility di ricostruzione

Ogni fetta $1024^2$ è partizionata in una griglia $4 \times 4$ di patches $256^2$, senza
sovrapposizione:

$$
\text{indice} \;=\; s \cdot 16 + g_y \cdot 4 + g_x
\qquad s \in [0, 24],\; g_y, g_x \in [0, 3]
$$

L'ordine è **fisso e documentato**: le fette sono ricostruibili nei notebook successivi
senza metadati (la formula è invertibile). Totali: $25 \times 16 = 400$ patches per
simulazione. Shape: $X$ $(400, 1, 256, 256)$ float32; $Y$ $(400, 1, 256, 256)$ uint8 ---
canale esplicito perché è la shape attesa da `Conv2d(in_channels=1, ...)`, `uint8` per le
maschere binarie ($-4\times$ storage).

`tiles_to_slices_from_flat` è l'operazione inversa: ricostruisce le 25 fette da un array
$(N \cdot 16, 256, 256)$; usata dalla §10 (verifica) e dalla §11 (visual check).

In HC la funzione viene chiamata come `extract_patches(dens_HC, dens_HC)`: il secondo
argomento (labels) esiste per compatibilità di firma con NB1 ma qui non serve --- l'uscita
Y viene scartata perché il target è la **copia** dell'originale (§7).



```python
# ============================================================
# ESTRAZIONE PATCHES + UTILITY DI RICOSTRUZIONE
# ============================================================
def extract_patches(dens, labels):
    """Estrae patches 256² usando reshape/transpose (vettorizzato, no loop Python).
    In HC viene chiamata con extract_patches(dens_HC, dens_HC): l'uscita Y è scartata."""
    n_grid = IMG_SIZE // PATCH_SIZE

    # (25, 1024, 1024) -> (25, 4, 256, 4, 256) -> (25, 4, 4, 256, 256) -> (400, 256, 256)
    X = dens.reshape(N_SLICES, n_grid, PATCH_SIZE, n_grid, PATCH_SIZE) \
            .transpose(0, 1, 3, 2, 4) \
            .reshape(-1, PATCH_SIZE, PATCH_SIZE)

    Y = labels.reshape(N_SLICES, n_grid, PATCH_SIZE, n_grid, PATCH_SIZE) \
              .transpose(0, 1, 3, 2, 4) \
              .reshape(-1, PATCH_SIZE, PATCH_SIZE)

    # dimensione canale esplicita: (400, 1, 256, 256)
    return X[:, None, :, :], Y[:, None, :, :]


def tiles_to_slices_from_flat(tiles, patch, img):
    """Ricostruisce le fette da un array (N*16, patch, patch).
    Ordine: indice = slice*16 + gy*4 + gx (identico a nb1/nb2/nb3)."""
    n_grid = img // patch
    n = tiles.shape[0] // n_grid ** 2
    out = np.empty((n, img, img), dtype=tiles.dtype)
    for i in range(n):
        for k in range(n_grid ** 2):
            gy, gx = divmod(k, n_grid)
            out[i, gy*patch:(gy+1)*patch, gx*patch:(gx+1)*patch] = tiles[i*n_grid**2 + k]
    return out

```

## 7 --- `process_simulation_hc`: pipeline HC completa per una simulazione

Flusso:

1.  **Skip**: se `X_patches_HC.npy` + `dataset_info_HC.json` esistono e `PIPELINE_VERSION`
    corrisponde $\rightarrow$ skip (`SKIP_EXISTING`; il secondo run del loop non
    ricalcola nulla).
2.  `load_snapshot` $\rightarrow$ `get_smoothing_lengths` (cache, §3) $\rightarrow$ filtro
    $h \leq 0$ / non finiti (scartate conteggiate e salvate in `STATS`).
3.  `compute_rho_particles` (B2) $\rightarrow$ `render_slices_hc` (B3--B4):
    $I_1$, $I_2$, $T$, `mpu` (da $I_2$).
4.  Copia `T[RAW_SAMPLE_SLICE]` e `I2[RAW_SAMPLE_SLICE]` come raw samples (verifica §10),
    poi `del` degli array grandi.
5.  `normalize_slices(T)` (B5) $\rightarrow$ `dens_HC`, `med_hc`, `sig_hc`.
6.  **Y caricata dall'originale** `sim_<id>/Y_mask_patches.npy` (sola lettura): se manca,
    `FileNotFoundError` esplicito --- eseguire prima NB1 standard.
7.  `extract_patches(dens_HC, dens_HC)` $\rightarrow$ `X_HC` (l'uscita Y è scartata).
8.  Salvataggio dei 7 file in `sim_<id>_highcontrast/` (§0).

### `dataset_info_HC.json` --- schema annidato

| Blocco | Chiavi |
|---|---|
| (top-level) | `PIPELINE_VERSION`, `MODE`, `SIMULATION_ID` |
| `PHYSICS_AND_GRID` | `LBOX_MPC_H`, `M_DM_MSUNH`, `N_SLICES`, `IMG_SIZE`, `PATCH_SIZE`, `MASS_CUT_MSUNH`, `N_MIN_PARTICLES`, `REDSHIFT`, `OMEGA0`, `HUBBLE` |
| `INPUTS_FOR_UNET` | `X_FILE`, `X_CHANNEL`, `NORMALIZATION`{`MEDIAN`, `SIGMA`, `EMPTY_FILL`} |
| `TARGETS` | `Y_FILE`, `Y_FORMAT` |
| `CALIBRATION_FOR_MASS` | `MPU_FILE`, `MPU_MEDIAN`, `RAW_T_SAMPLE`, `RAW_DENSITY_SAMPLE`, `RAW_SAMPLE_SLICE` |
| `STATS` | `PATCHES_TOTAL`, `PATCHES_POSITIVE`, `PIXELS_TOTAL`, `PIXELS_POSITIVE`, `PIXELS_POSITIVE_FRAC`, `N_PARTICLES_DROPPED_H`, `SECONDS` |

Il percorso di snapshot e catalogo non è salvato: deriva da `SIMULATION_ID` tramite il
layout fisso di CAMELS (§2). `MASS_CUT_MSUNH` è un metadato informativo: il taglio è già
incorporato nella Y originale. Lo schema annidato separa fisica / input / target /
calibrazione / statistiche: i notebook successivi (NB2-HC, NB3-HC) leggono
`INPUTS_FOR_UNET.NORMALIZATION` e `CALIBRATION_FOR_MASS`, non più le chiavi piatte
`MED_D_HC` / `SIG_D_HC` dell'abbozzo v1.1.0.



```python
# ============================================================
# PIPELINE HC PER SIMULAZIONE (v1.2.0-hc, dataset_info_HC annidato)
# ============================================================
def process_simulation_hc(sim_id, snap_path, cat_path):
    """Pipeline HC completa per una simulazione: rendering I1/I2, T = I1/I2,
    normalizzazione, patches. Y è COPIATA dall'originale (mai rigenerata);
    le cartelle sim_<id>/ originali restano intatte (sola lettura).
    Ritorna un dict di riepilogo, o None se la simulazione viene saltata."""
    t0 = time.time()
    out_dir   = os.path.join(OUT_ROOT, f"sim_{sim_id}{SIM_SUFFIX}")
    x_path    = os.path.join(out_dir, "X_patches_HC.npy")
    info_path = os.path.join(out_dir, "dataset_info_HC.json")

    # ---------- 1. skip se completo ----------
    if os.path.exists(x_path) and os.path.exists(info_path):
        if SKIP_EXISTING:
            with open(info_path) as fh:
                existing = json.load(fh)
            if existing.get("PIPELINE_VERSION") == PIPELINE_VERSION:
                print(f"[{sim_id}] HC completo (v{PIPELINE_VERSION}) -> skip")
                return None

    # ---------- 2. snapshot + smoothing lengths ----------
    print(f"{'=' * 70}\n{sim_id} [HIGH_CONTRAST]\n{'=' * 70}")
    sim = load_snapshot(snap_path)
    lbox, m_dm = sim["lbox"], sim["m_dm"]
    mass_cut = N_MIN_PARTICLES * m_dm
    print(f"particelle {len(sim['pos']):,} | box {lbox:.2f} Mpc/h | m_DM {m_dm:.3e} Msun/h | "
          f"Omega0 {sim['omega0']:.4f} | fetta {lbox / N_SLICES:.3f} Mpc/h")

    pos = sim.pop("pos")
    h = get_smoothing_lengths(sim_id, snap_path, pos, lbox)
    good = np.isfinite(h) & (h > 0)
    n_dropped_h = int((~good).sum())
    pos, h = pos[good], h[good]
    print(f"  h: {n_dropped_h} particelle scartate "
          f"({100 * n_dropped_h / max(1, len(h) + n_dropped_h):.4f}%)")

    # ---------- 3. rho_j + rendering I1/I2/T ----------
    rho_j = compute_rho_particles(h, m_dm, k=N_NEIGHBORS_H)
    I1_slices, I2_slices, T_slices, mpu = render_slices_hc(pos, h, rho_j, lbox, m_dm)
    raw_slice_T  = T_slices[RAW_SAMPLE_SLICE].copy()      # §10: verifica catena T
    raw_slice_I2 = I2_slices[RAW_SAMPLE_SLICE].copy()     # §10: verifica uguaglianza I2 vs originale
    del pos, h, rho_j, I1_slices, I2_slices
    gc.collect()

    # ---------- 4. normalizzazione T ----------
    dens_HC, med_hc, sig_hc = normalize_slices(T_slices)
    del T_slices
    gc.collect()
    print(f"normalizzazione T: mediana={med_hc:.4f}, sigma={sig_hc:.4f}")

    # ---------- 5. Y caricata dall'originale (sola lettura, mai modificata) ----------
    y_orig_path = os.path.join(OUT_ROOT, f"sim_{sim_id}", "Y_mask_patches.npy")
    if not os.path.exists(y_orig_path):
        raise FileNotFoundError(
            f"Y originale mancante: {y_orig_path}. Esegui prima il notebook standard (NB1).")
    Y = np.load(y_orig_path)

    # ---------- 6. patches (X da dens_HC; Y scartata: è la copia dell'originale) ----------
    X, _ = extract_patches(dens_HC, dens_HC)
    n_pos = int((Y[:, 0].sum(axis=(1, 2)) > 0).sum())
    print(f"patches: {X.shape} | positive: {n_pos}/{len(X)} ({100 * n_pos / len(X):.1f}%)")

    # ---------- 7. salvataggio (solo in sim_<id>_highcontrast/) ----------
    os.makedirs(out_dir, exist_ok=True)
    np.save(x_path, X)
    np.save(os.path.join(out_dir, "Y_mask_patches.npy"), Y)
    np.save(os.path.join(out_dir, "norm_stats_HC.npy"),
            np.array([[med_hc, sig_hc]], dtype=np.float32))
    np.save(os.path.join(out_dir, "mpu_per_slice.npy"), mpu)
    np.save(os.path.join(out_dir, "raw_slice_sample_HC.npy"), raw_slice_T)
    np.save(os.path.join(out_dir, "raw_slice_sample_I2.npy"), raw_slice_I2)

    dataset_info_HC = {
        "PIPELINE_VERSION": PIPELINE_VERSION,
        "MODE":             MODE,
        "SIMULATION_ID":    sim_id,
        "PHYSICS_AND_GRID": {
            "LBOX_MPC_H":      lbox,
            "M_DM_MSUNH":      m_dm,
            "N_SLICES":        N_SLICES,
            "IMG_SIZE":        IMG_SIZE,
            "PATCH_SIZE":      PATCH_SIZE,
            "MASS_CUT_MSUNH":  mass_cut,
            "N_MIN_PARTICLES": N_MIN_PARTICLES,
            "REDSHIFT":        sim["redshift"],
            "OMEGA0":          sim["omega0"],
            "HUBBLE":          sim["hubble"],
        },
        "INPUTS_FOR_UNET": {
            "X_FILE":    "X_patches_HC.npy",
            "X_CHANNEL": "log10(I1/I2), robust z-score (high contrast)",
            "NORMALIZATION": {
                "MEDIAN":     med_hc,
                "SIGMA":      sig_hc,
                "EMPTY_FILL": EMPTY_FILL,
            },
        },
        "TARGETS": {
            "Y_FILE":   "Y_mask_patches.npy",
            "Y_FORMAT": "binary FoF mask (copiata dall'originale, invariata)",
        },
        "CALIBRATION_FOR_MASS": {
            "MPU_FILE":           "mpu_per_slice.npy",
            "MPU_MEDIAN":         float(np.nanmedian(mpu)),
            "RAW_T_SAMPLE":       "raw_slice_sample_HC.npy",
            "RAW_DENSITY_SAMPLE": "raw_slice_sample_I2.npy",
            "RAW_SAMPLE_SLICE":   RAW_SAMPLE_SLICE,
        },
        "STATS": {
            "PATCHES_TOTAL":         int(len(X)),
            "PATCHES_POSITIVE":      n_pos,
            "PIXELS_TOTAL":          int(N_SLICES * IMG_SIZE * IMG_SIZE),
            "PIXELS_POSITIVE":       int(Y.sum()),
            "PIXELS_POSITIVE_FRAC":  float(Y.sum()) / (N_SLICES * IMG_SIZE * IMG_SIZE),
            "N_PARTICLES_DROPPED_H": n_dropped_h,
            "SECONDS":               time.time() - t0,
        },
    }
    with open(info_path, "w") as fh:
        json.dump(dataset_info_HC, fh, indent=2)

    print(f"[{sim_id}] HC completato in {time.time() - t0:.1f} s -> {out_dir}")
    return {"sim_id": sim_id, "n_patches": len(X), "positive": n_pos,
            "seconds": time.time() - t0}

```

## 8 --- Esecuzione su tutte le simulazioni

Loop robusto a fallimenti singoli: se una simulazione solleva un'eccezione (Y originale
mancante, errore di rendering), viene catturata, registrata e la pipeline **continua**
con le altre. Alla fine: `DataFrame` di riepilogo + elenco esplicito dei `FAILED`.

**Test di fumo** (consigliato prima del run completo): limitare temporaneamente il loop a
una simulazione, es.

``` python
for sim_id, (snap_path, cat_path) in {"LH_105_090": SIMULATIONS["LH_105_090"]}.items():
```

verificare §9--§10 su quella simulazione, poi ripristinare il loop completo. Con $h$ in
cache, il costo per simulazione è dominato dal doppio render di 25 fette ($\sim2\times$
il rendering standard).



```python
# ============================================================
# ESECUZIONE SU TUTTE LE SIMULAZIONI (solo HC)
# ============================================================
summaries, failures = [], []

for sim_id, (snap_path, cat_path) in SIMULATIONS.items():
    try:
        result = process_simulation_hc(sim_id, snap_path, cat_path)
        if result is not None:
            summaries.append(result)
    except Exception as exc:
        failures.append((sim_id, repr(exc)))
        traceback.print_exc()
    gc.collect()

# --- riepilogo ---
print("\n" + "=" * 70)
if summaries:
    print(f"Simulazioni processate: {len(summaries)}\n")
    print(pd.DataFrame(summaries).to_string(index=False))
else:
    print("Nessuna simulazione processata (tutte skippate o fallite).")

if failures:
    print(f"\n⚠️  Simulazioni FALLITE: {len(failures)}")
    for sim_id, err in failures:
        print(f"  FAILED {sim_id}: {err}")
else:
    print("\n✓ Nessun fallimento.")

```

    [LH_100_090] HC completo (v1.2.0-hc) -> skip
    [LH_101_090] HC completo (v1.2.0-hc) -> skip
    [LH_102_090] HC completo (v1.2.0-hc) -> skip
    [LH_103_090] HC completo (v1.2.0-hc) -> skip
    [LH_104_090] HC completo (v1.2.0-hc) -> skip
    [LH_105_090] HC completo (v1.2.0-hc) -> skip
    [LH_106_090] HC completo (v1.2.0-hc) -> skip
    
    ======================================================================
    Nessuna simulazione processata (tutte skippate o fallite).
    
    ✓ Nessun fallimento.
    

## 9 --- Verifica dei file di output

Controllo indipendente che ogni simulazione in `SIMULATIONS` abbia i **7 file** attesi in
`input_data/sim_<id>_highcontrast/` con shape/dtype corretti (via `mmap_mode="r"`: non
ricarica i dati, verifica solo i metadati strutturali --- un controllo esaustivo sarebbe
costoso e ridondante con il caricamento che fa NB2-HC), e `PIPELINE_VERSION` corrente in
`dataset_info_HC.json` (chiave top-level, invariata nello schema annidato).

| File | Verifica |
|---|---|
| `X_patches_HC.npy` | shape $(400, 1, 256, 256)$, float32 |
| `Y_mask_patches.npy` | shape $(400, 1, 256, 256)$, uint8 |
| `raw_slice_sample_HC.npy` | shape $(1024, 1024)$, float32 |
| `raw_slice_sample_I2.npy` | shape $(1024, 1024)$, float32 |
| `mpu_per_slice.npy` | shape $(25,)$ |
| `norm_stats_HC.npy` | shape $(1, 2)$ |
| `dataset_info_HC.json` | `PIPELINE_VERSION == "1.2.0-hc"` |

Se anche una sola cella è `✗`, la §8 ha mancato qualcosa --- indagare prima di passare a
NB2-HC.



```python
# ============================================================
# VERIFICA OUTPUT HC PER TUTTE LE SIMULAZIONI
# ============================================================
def check_sim_outputs_hc(sim_id):
    """Verifica presenza + shape/dtype dei 7 file attesi in sim_<id>_highcontrast/.
    Ritorna un dict con i risultati per il riepilogo."""
    d = os.path.join(OUT_ROOT, f"sim_{sim_id}{SIM_SUFFIX}")
    expected_patches = (N_SLICES * (IMG_SIZE // PATCH_SIZE) ** 2, 1, PATCH_SIZE, PATCH_SIZE)
    res = {"sim_id": sim_id}

    for fname, shape, dtype, key in [
        ("X_patches_HC.npy",        expected_patches,     np.float32, "X_HC"),
        ("Y_mask_patches.npy",      expected_patches,     np.uint8,   "Y"),
        ("raw_slice_sample_HC.npy", (IMG_SIZE, IMG_SIZE), np.float32, "raw_T"),
        ("raw_slice_sample_I2.npy", (IMG_SIZE, IMG_SIZE), np.float32, "raw_I2"),
        ("mpu_per_slice.npy",       (N_SLICES,),          None,       "mpu"),
        ("norm_stats_HC.npy",       (1, 2),               None,       "norm_HC"),
    ]:
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            res[key] = "✗ manca"
            continue
        a = np.load(p, mmap_mode="r")
        ok = a.shape == shape and (dtype is None or a.dtype == dtype)
        res[key] = "✓" if ok else f"✗ {a.shape} {a.dtype}"
        del a


    # dataset_info_HC.json (schema annidato: PIPELINE_VERSION resta top-level)
    p = os.path.join(d, "dataset_info_HC.json")
    if os.path.exists(p):
        with open(p) as fh:
            info = json.load(fh)
        v = info.get("PIPELINE_VERSION", "?")
        res["version"] = "✓" if v == PIPELINE_VERSION else f"✗ {v}"
    else:
        res["version"] = "✗ manca"

    return res


rows = [check_sim_outputs_hc(s) for s in SIMULATIONS]
df_check_hc = pd.DataFrame(rows).set_index("sim_id")
print(f"Verifica output HC (atteso: pipeline v{PIPELINE_VERSION})\n")
print(df_check_hc.to_string())

n_ok = int((df_check_hc == "✓").all(axis=1).sum())
print(f"\nSimulazioni HC complete: {n_ok}/{len(df_check_hc)}")
if n_ok == len(df_check_hc):
    print("✓ Tutte le simulazioni hanno i file HC attesi.")
else:
    print("⚠️  Alcune simulazioni hanno file mancanti o non aggiornati — vedi tabella sopra.")

```

    Verifica output HC (atteso: pipeline v1.2.0-hc)
    
               X_HC  Y raw_T raw_I2 mpu norm_HC version
    sim_id                                             
    LH_100_090    ✓  ✓     ✓      ✓   ✓       ✓       ✓
    LH_101_090    ✓  ✓     ✓      ✓   ✓       ✓       ✓
    LH_102_090    ✓  ✓     ✓      ✓   ✓       ✓       ✓
    LH_103_090    ✓  ✓     ✓      ✓   ✓       ✓       ✓
    LH_104_090    ✓  ✓     ✓      ✓   ✓       ✓       ✓
    LH_105_090    ✓  ✓     ✓      ✓   ✓       ✓       ✓
    LH_106_090    ✓  ✓     ✓      ✓   ✓       ✓       ✓
    
    Simulazioni HC complete: 7/7
    ✓ Tutte le simulazioni hanno i file HC attesi.
    

## 10 --- Verifica end-to-end (B7): catena T e uguaglianza I2

Due check indipendenti per simulazione, adattati dalla verifica end-to-end di NB1
standard.

### (1) Catena T --- la stessa che NB3-HC userà per denormalizzare

$$
T_{\text{ricostruita}}(x,y) \;=\; 10^{\,X(x,y)\,\sigma_{HC} + \mu_{HC}},
\qquad
T_{\text{ricostruita}}[X = \texttt{EMPTY\_FILL}] = 0
$$

con $\mu_{HC}, \sigma_{HC}$ letti da `INPUTS_FOR_UNET.NORMALIZATION` di
`dataset_info_HC.json`, confrontata pixel-per-pixel con `raw_slice_sample_HC.npy` (T salvata
**prima** della normalizzazione, §7). Soglia: $\max$ errore relativo $< 10^{-4}$, 0
outlier, tutti i pixel EMPTY a 0 su entrambi i lati. La catena
$T \to \log_{10} \to$ z-score $\to X \to$ denorm introduce solo arrotondamenti
float32 ($\sim10^{-6}$ relativo atteso). Cattura bug di ordine patch, base del log,
off-by-one, `EMPTY_FILL` non ripristinato --- tutti invisibili a un semplice check di shape.

### (2) Uguaglianza I2 --- validità di `mpu`

`raw_slice_sample_I2.npy` (render HC) confrontato con
`sim_<id>/raw_slice_sample.npy` (originale NB1, **sola lettura**). Stessa soglia:
$\max$ errore relativo $< 10^{-4}$, 0 outlier.

**Se (2) fallisce**: i parametri di rendering HC differiscono da quelli standard
$\rightarrow$ `mpu_per_slice.npy` non è valido $\rightarrow$ le masse di NB3 sarebbero
sbagliate di un fattore costante, con fit biased. Indagare i parametri di rendering
**prima** di procedere.



```python
# ============================================================
# VERIFICA END-TO-END HC: catena T <-> raw_slice_sample_HC, I2_HC <-> I2_orig (B7)
# ============================================================
def verify_chain_HC(sim_id):
    """Due check indipendenti per simulazione:
    (1) catena T: 10^(X*sigma_HC+mu_HC) ricostruisce raw_slice_sample_HC.npy;
    (2) uguaglianza I2: raw_slice_sample_I2.npy == sim_<id>/raw_slice_sample.npy.
    Se (2) fallisce i parametri di rendering HC differiscono dall'originale
    e mpu_per_slice.npy non è valido per le masse di NB3."""
    d_hc = os.path.join(OUT_ROOT, f"sim_{sim_id}{SIM_SUFFIX}")
    d_or = os.path.join(OUT_ROOT, f"sim_{sim_id}")

    # --- metadati (schema annidato) ---
    with open(os.path.join(d_hc, "dataset_info_HC.json")) as fh:
        info = json.load(fh)
    norm       = info["INPUTS_FOR_UNET"]["NORMALIZATION"]
    med, sig   = float(norm["MEDIAN"]), float(norm["SIGMA"])
    empty_fill = float(norm["EMPTY_FILL"])
    raw_slice  = int(info["CALIBRATION_FOR_MASS"]["RAW_SAMPLE_SLICE"])

    # ---------- check 1: catena T ----------
    Xn = np.asarray(np.load(os.path.join(d_hc, "X_patches_HC.npy"), mmap_mode="r")[:, 0],
                    dtype=np.float32)
    Xn_slice = tiles_to_slices_from_flat(Xn, PATCH_SIZE, IMG_SIZE)[raw_slice]
    del Xn

    with np.errstate(over="ignore", invalid="ignore"):
        T_rec = (10.0 ** (Xn_slice.astype(np.float64) * sig + med)).astype(np.float32)
    T_rec[Xn_slice == empty_fill] = 0.0

    T_gt = np.load(os.path.join(d_hc, "raw_slice_sample_HC.npy"))

    valid   = Xn_slice != empty_fill
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"sim_id": sim_id, "status": "⚠️ nessun pixel valido",
                "max_rel_T": np.nan, "n_out_T": 0,
                "max_rel_I2": np.nan, "n_out_I2": 0}

    rel_T     = np.abs(T_rec[valid] - T_gt[valid]) / np.maximum(T_gt[valid], 1e-30)
    max_rel_T = float(rel_T.max())
    n_out_T   = int((rel_T > 1e-4).sum())
    empty_ok  = bool(np.all(T_rec[~valid] == 0.0) and np.all(T_gt[~valid] == 0.0))

    # ---------- check 2: uguaglianza I2 (HC vs originale, sola lettura) ----------
    I2_hc = np.load(os.path.join(d_hc, "raw_slice_sample_I2.npy")).astype(np.float64)
    I2_or = np.load(os.path.join(d_or, "raw_slice_sample.npy")).astype(np.float64)

    rel_I2     = np.abs(I2_hc - I2_or) / np.maximum(np.abs(I2_or), 1e-30)
    max_rel_I2 = float(rel_I2.max())
    n_out_I2   = int((rel_I2 > 1e-4).sum())

    ok = (max_rel_T < 1e-4 and n_out_T == 0 and empty_ok
          and max_rel_I2 < 1e-4 and n_out_I2 == 0)
    return {
        "sim_id":    sim_id,
        "status":    "✓ OK" if ok else "✗ FAIL",
        "max_rel_T": max_rel_T, "n_out_T": n_out_T,
        "max_rel_I2": max_rel_I2, "n_out_I2": n_out_I2,
    }


chain_results = [verify_chain_HC(s) for s in SIMULATIONS]
df_chain = pd.DataFrame(chain_results).set_index("sim_id")
print(f"Verifica end-to-end HC (RAW_SAMPLE_SLICE = {RAW_SAMPLE_SLICE})\n")
print(df_chain.to_string(float_format=lambda v: f"{v:.2e}" if isinstance(v, float) else v))

n_ok = int((df_chain["status"] == "✓ OK").sum())
print(f"\nSimulazioni con catena HC verificata: {n_ok}/{len(df_chain)}")
if n_ok == len(df_chain):
    print("✓ Catena T e uguaglianza I2 verificate: NB3-HC può fidarsi di denorm e mpu.")
else:
    print("⚠️  Alcune simulazioni hanno fallito — indagare i parametri di rendering prima di procedere.")

```

    Verifica end-to-end HC (RAW_SAMPLE_SLICE = 12)
    
               status  max_rel_T  n_out_T  max_rel_I2  n_out_I2
    sim_id                                                     
    LH_100_090   ✓ OK   2.32e-06        0    0.00e+00         0
    LH_101_090   ✓ OK   1.62e-06        0    0.00e+00         0
    LH_102_090   ✓ OK   1.48e-06        0    0.00e+00         0
    LH_103_090   ✓ OK   2.29e-06        0    0.00e+00         0
    LH_104_090   ✓ OK   2.36e-06        0    0.00e+00         0
    LH_105_090   ✓ OK   1.56e-06        0    0.00e+00         0
    LH_106_090   ✓ OK   1.75e-06        0    0.00e+00         0
    
    Simulazioni con catena HC verificata: 7/7
    ✓ Catena T e uguaglianza I2 verificate: NB3-HC può fidarsi di denorm e mpu.
    

## 11 --- Visual check HC: T vs I2 vs diff con overlay FoF/Subhalo

Quattro pannelli orizzontali sulla finestra di zoom della fetta `RAW_SAMPLE_SLICE`
(`SLICE_TO_SHOW` è **forzata** ad essa: i raw sample esistono solo per quella fetta):

-   **(a)** $\log_{10}(T)$ --- l'input della U-Net --- overlay: dischi FoF
    (ciano, $R_{200c}$) + subhalo (giallo, $2\,R_{1/2}$, minimo $3$ px);
-   **(b)** $\log_{10}(I_2)$ --- la densità di massa (stesso overlay): confronto diretto
    con l'input, da cui $T$ deriva;
-   **(c)** maschera $Y$ (copia dell'originale), ricostruita a fetta da
    `tiles_to_slices_from_flat`;
-   **(d)** $\log_{10}(T) - \log_{10}(I_2)$, colormap `RdBu_r` centrata su 0.
    **Atteso: rosso sui picchi** (T enfatizza le zone dense), **bianco/blu sul
    background**. I pixel vuoti ($I_2 = 0 \Rightarrow T = 0$) sono in bianco (NaN
    mascherati).

Config in cima alla cella: `SIM_ID`, `SLICE_TO_SHOW` (forzata), `ZOOM_SIZE` (256 default),
`ZOOM_CENTER` (`None` = zoom automatico sull'alone FoF più massiccio della slice),
`N_MIN_SUB = 20` (filtro sui subhalo in **particelle DM**, `SubhaloLenType[:, 1]`).
Zoom con wrap periodico: cerchi a cavallo del bordo della finestra vengono gestiti con
candidati $\pm L_{\text{box}}$. FoF e Subhalo sono letti inline dal catalogo (sola
lettura) e assegnati alla slice con la stessa regola della pipeline (`floor(z/Δz)` sui
centri; per i FoF anche il taglio in massa).

Criterio di accettazione: pannello (d) con rosso sui picchi e blu/bianco sul background;
contorni ciano/giallo sui picchi di (a), non su zone vuote.



```python
# ============================================================
# 11 — VISUAL CHECK HC: T vs I2 vs diff, overlay FoF + Subhalo
# (Layout 2x4: Riga superiore CON overlay, Riga inferiore SENZA)
# ============================================================
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- configurazione ---
SIM_ID        = "LH_105_090"
SLICE_TO_SHOW = RAW_SAMPLE_SLICE
ZOOM_SIZE, ZOOM_CENTER, N_MIN_SUB = 256, None, 20

if SLICE_TO_SHOW != RAW_SAMPLE_SLICE:
    print(f"[visual check] SLICE_TO_SHOW forzato a RAW_SAMPLE_SLICE = {RAW_SAMPLE_SLICE}")
    SLICE_TO_SHOW = RAW_SAMPLE_SLICE

d_hc = os.path.join(OUT_ROOT, f"sim_{SIM_ID}{SIM_SUFFIX}")

# --- immagini + maschera (solo la slice richiesta) ---
T_slice  = np.load(os.path.join(d_hc, "raw_slice_sample_HC.npy"))
I2_slice = np.load(os.path.join(d_hc, "raw_slice_sample_I2.npy"))
n_grid   = IMG_SIZE // PATCH_SIZE
ng2      = n_grid * n_grid
Y_slice  = (np.asarray(np.load(os.path.join(d_hc, "Y_mask_patches.npy"), mmap_mode="r")
                       [SLICE_TO_SHOW*ng2:(SLICE_TO_SHOW+1)*ng2, 0])
            .reshape(n_grid, n_grid, PATCH_SIZE, PATCH_SIZE)
            .transpose(0, 2, 1, 3).reshape(IMG_SIZE, IMG_SIZE))

# --- cataloghi ---
run = SIM_ID.rsplit("_", 1)[0]
cat = os.path.join(SIM_ROOT, run, f"groups_{run}", f"groups_{SNAP_TAG}_{run}.hdf5")
with h5py.File(cat, "r") as f:
    lbox_v = float(np.ravel(f["Header"].attrs["BoxSize"])[0]) / 1000.0
    with h5py.File(SIMULATIONS[SIM_ID][0], "r") as sf:
        mt = sf["Header"].attrs.get("MassTable")
        if mt is None:
            raise KeyError(f"'MassTable' not found in snapshot header: {SIMULATIONS[SIM_ID][0]}")
        m_dm_v = float(np.ravel(mt)[1]) * 1e10
    grp, sub = f["Group"], f["Subhalo"]
    fof_pos  = grp["GroupPos"][:].astype(np.float64) / 1000.0
    fof_r200 = grp["Group_R_Crit200"][:].astype(np.float64) / 1000.0
    fof_mass = grp["GroupMass"][:].astype(np.float64) * 1e10
    sub_pos   = sub["SubhaloPos"][:].astype(np.float64) / 1000.0
    sub_rhalf = sub["SubhaloHalfmassRad"][:].astype(np.float64) / 1000.0
    sub_n_dm  = sub["SubhaloLenType"][:, 1].astype(np.int64)

PX_MPC, dz_v, px_grid = lbox_v / IMG_SIZE, lbox_v / N_SLICES, IMG_SIZE / lbox_v

# --- oggetti nella slice ---
slc_fof = np.floor(fof_pos[:, 2] / dz_v).astype(int) == SLICE_TO_SHOW
slc_sub = np.floor(sub_pos[:, 2] / dz_v).astype(int) == SLICE_TO_SHOW
in_fof  = slc_fof & (fof_mass >= N_MIN_PARTICLES * m_dm_v) & (fof_r200 > 0)
in_sub  = slc_sub & (sub_n_dm >= N_MIN_SUB)

fof_x = fof_pos[in_fof, 0] * px_grid;  fof_y = fof_pos[in_fof, 1] * px_grid
fof_r = np.maximum(fof_r200[in_fof] * px_grid, MIN_RADIUS_PX)
sub_x = sub_pos[in_sub, 0] * px_grid;  sub_y = sub_pos[in_sub, 1] * px_grid
sub_r = np.maximum(2.0 * sub_rhalf[in_sub] * px_grid, 3.0 * PX_MPC)

# --- zoom ---
if ZOOM_CENTER is None:
    if in_fof.any():
        k = int(np.argmax(fof_mass[in_fof]))
        ZC, zoom_note = (float(fof_x[k]), float(fof_y[k])), \
            f"zoom auto: FoF più massiccio (M = {fof_mass[in_fof][k]:.2e} Msun/h)"
    else:
        ZC, zoom_note = (IMG_SIZE / 2, IMG_SIZE / 2), "zoom: centro del box"
else:
    ZC, zoom_note = (float(ZOOM_CENTER[0]), float(ZOOM_CENTER[1])), "zoom: centro manuale"

shift_x = int(round(ZC[0])) - ZOOM_SIZE // 2
shift_y = int(round(ZC[1])) - ZOOM_SIZE // 2

# --- finestra con wrap (1 fancy-index, no roll doppio) ---
rows = (np.arange(ZOOM_SIZE) + shift_y) % IMG_SIZE
cols = (np.arange(ZOOM_SIZE) + shift_x) % IMG_SIZE
def zoom_window(img): return img[np.ix_(rows, cols)]

T_win, I2_win, Y_win = zoom_window(T_slice), zoom_window(I2_slice), zoom_window(Y_slice)

# --- overlay con wrap VETTORIZZATO (9 offset in un colpo) ---
def overlay_in_window(cx, cy, r):
    cx, cy, r = map(np.asarray, (cx, cy, r))
    if cx.size == 0:
        return np.array([]), np.array([]), np.array([])
    wx0 = (cx - shift_x) % IMG_SIZE
    wy0 = (cy - shift_y) % IMG_SIZE
    off = np.array([-IMG_SIZE, 0, IMG_SIZE])
    DX, DY = (a.ravel()[:, None] for a in np.meshgrid(off, off, indexing="ij"))
    wx = wx0[None, :] + DX
    wy = wy0[None, :] + DY
    R  = r[None, :]
    m  = (wx > -R) & (wx < ZOOM_SIZE + R) & (wy > -R) & (wy < ZOOM_SIZE + R)
    R3 = np.broadcast_to(R, wx.shape)
    return wx[m], wy[m], R3[m]

fof_wx, fof_wy, fof_wr = (a * PX_MPC for a in overlay_in_window(fof_x, fof_y, fof_r))
sub_wx, sub_wy, sub_wr = (a * PX_MPC for a in overlay_in_window(sub_x, sub_y, sub_r))

print(f"{SIM_ID} | fetta {SLICE_TO_SHOW} | {zoom_note}")
print(f"  FoF: {int(in_fof.sum())} ({len(fof_wx)} in finestra) | "
      f"Subhalo (>= {N_MIN_SUB}): {int(in_sub.sum())} ({len(sub_wx)} in finestra)")

# --- log10 + diff (errstate condiviso) ---
with np.errstate(divide="ignore", invalid="ignore"):
    T_log    = np.log10(np.where(T_win  > 0, T_win,  np.nan))
    I2_log   = np.log10(np.where(I2_win > 0, I2_win, np.nan))
    diff_log = T_log - I2_log

vmin_T,  vmax_T  = np.nanpercentile(T_log,  [0.5, 99.5])
vmin_I2, vmax_I2 = np.nanpercentile(I2_log, [0.5, 99.5])
valid_d  = diff_log[np.isfinite(diff_log)]
vmax_d   = float(np.percentile(np.abs(valid_d), 99)) if valid_d.size else 1.0

# --- pannelli ---
panels = [
    ("(a) T = I1/I2 (log10)",      T_log,    "magma",   vmin_T,  vmax_T,  1.00, "log10(T)",      "black"),
    ("(b) I2 = densità (log10)",   I2_log,   "magma",   vmin_I2, vmax_I2, 1.00, "log10(I2)",     "black"),
    ("(c) Maschera FoF (Y)",       Y_win,    "Blues",   0,       1,       0.85, "maschera (0/1)", None),
    ("(d) Δ log10(T) - log10(I2)", diff_log, "RdBu_r", -vmax_d,  vmax_d,  1.00, "Δ log10",       "white"),
]

# --- tick precalcolati (usati 8×) ---
Lwin = ZOOM_SIZE * PX_MPC
xt   = np.linspace(0, Lwin, 5)
xtl  = [f"{v:.1f}" for v in xt]

# --- figure ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
extent = [0, Lwin, 0, Lwin]
fig.suptitle(f"{SIM_ID} — visual check HC | fetta {SLICE_TO_SHOW} | finestra {ZOOM_SIZE} px "
             f"({Lwin:.1f} Mpc/h) | {zoom_note}", fontsize=13, fontweight="bold")

def draw_overlays(ax):
    if len(fof_wx):
        ax.add_collection(EllipseCollection(2*fof_wr, 2*fof_wr, 0, units="xy",
            offsets=np.c_[fof_wx, fof_wy], transOffset=ax.transData,
            facecolors="none", edgecolors="cyan", linewidths=1.1, alpha=0.9))
        ax.plot(fof_wx, fof_wy, "+", color="cyan", ms=5, mew=1.0)
    if len(sub_wx):
        ax.add_collection(EllipseCollection(2*sub_wr, 2*sub_wr, 0, units="xy",
            offsets=np.c_[sub_wx, sub_wy], transOffset=ax.transData,
            facecolors="none", edgecolors="yellow", linewidths=0.9, alpha=0.7))
        ax.plot(sub_wx, sub_wy, "+", color="yellow", ms=4, mew=0.8)
    handles = [
        Patch(facecolor="none", edgecolor="cyan",   label=r"FoF $R_{200c}$"),
        Line2D([], [], marker="+", color="cyan",   lw=0, label="Centro FoF"),
        Patch(facecolor="none", edgecolor="yellow", label=r"Subhalo $2R_{1/2}$"),
        Line2D([], [], marker="+", color="yellow", lw=0, label="Centro subhalo"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, facecolor="black",
              labelcolor="white", framealpha=0.7)

def style_ax(ax, title, cbar_label, cmap_bad):
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, Lwin); ax.set_ylim(0, Lwin)
    ax.set_xticks(xt); ax.set_yticks(xt)
    ax.set_xticklabels(xtl, fontsize=7); ax.set_yticklabels(xtl, fontsize=7)
    ax.set_xlabel("x [Mpc/h]", fontsize=8); ax.set_ylabel("y [Mpc/h]", fontsize=8)
    if cbar_label and ax.images:
        im = ax.images[-1]
        if cmap_bad is not None:
            cm = im.get_cmap().copy(); cm.set_bad(cmap_bad); im.set_cmap(cm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

for j, (title, data, cmap, vmin, vmax, alpha, cbl, cbad) in enumerate(panels):
    kw = dict(origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    axes[0, j].imshow(data, **kw)
    draw_overlays(axes[0, j])
    style_ax(axes[0, j], f"{title} — CON overlay", cbl, cbad)
    axes[1, j].imshow(data, **kw)
    style_ax(axes[1, j], f"{title} — SENZA overlay", cbl, cbad)

plt.show()
```

    LH_105_090 | fetta 12 | zoom auto: FoF più massiccio (M = 6.39e+12 Msun/h)
      FoF: 161 (23 in finestra) | Subhalo (>= 20): 490 (96 in finestra)
    


    
![png](make_input_patches_HC_files/make_input_patches_HC_22_1.png)
    

