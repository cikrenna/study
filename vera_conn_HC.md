## Quadro completo del Notebook 3-HC (conn_HC)

- **Titolo — Notebook 3-HC: dal catalogo di probabilità alto contrasto al catalogo 3D, con confronto head-to-head contro la baseline** (versione `conn_HC v1.0.0`, modalità HC)
- **Sezione 1 — Setup: percorsi duali, impianto fotometrico e verifica della catena**
    - 1.1 Due directory di input (`sim_<id>` baseline e `sim_<id>_highcontrast`) e due run di probabilità (`unet/`, `unet_HC/`)
    - 1.2 Impianto fotometrico: detection HC, massa da I2 (perché $T = I_1/I_2$ non ha unità di massa)
    - 1.3 Caricamento: densità I2, `mpu`, probabilità BASE/HC, cataloghi Group/Subhalo
    - 1.4 `verify_chain_HC` su LH_105 (catena T e uguaglianza I2) e geometria derivata
- **Sezione 2 — Algoritmi core (invariati, parametrizzati sul set di probabilità)**
    - Cache per $(\mathrm{mode}, t)$; `extract_blobs`, `link_blobs`, `build_catalog`, `match_catalogs`, `classify_fps`
- **Sezione 3 — Calibrazione della massa e diagnosi del bias (HC vs baseline)**
    - 3.1 Pipeline su entrambe le detection e fit log-log a $(0.95, 4)$
    - 3.2 Errore relativo di massa binnato (entrambe le detection)
    - 3.3 Autopsia dei falsi negativi: seen vs not seen (entrambe le detection)
- **Sezione 4 — Diagnosi dei falsi positivi e transizione delle soglie (HC vs baseline)**
    - 4.1 Fotografia statica: classificazione FP e purity fisica
    - 4.2 Transizione dinamica $0.99 \to 0.95$ (BASE e HC affiancati)
    - 4.3 Ispezione visiva del rumore (fetta campionata corretta per oggetti cross-slice)
    - 4.4 Conclusione
- **Sezione 5 — Griglia $4\times3$ head-to-head (baseline vs HC)**
    - 5.1 Esecuzione della doppia griglia (fit di massa in ogni cella)
    - 5.2 Tabelle affiancate e $\Delta C$ per cella
    - 5.3 Grafici trade-off sovrapposti
    - 5.4 Scelta della configurazione operativa HC (regola deterministica)
    - 5.5 Confronto visivo delle soglie su una fetta (HC)
- **Sezione 6 — HMF e completezza per massa (confronto)**
    - 6.1 HMF: FoF, predetti BASE/HC, matchati
    - 6.2 Completeness e purity in funzione di $M_{\mathrm{thr}}$
- **Sezione 7 — Diagnostica specifica HC**
    - 7.1 Contaminazione delle zone vuote (`EMPTY_FILL`)
    - 7.2 Robustness: background proiettato con detection HC
- **Sezione 8 — Riepilogo finale**
    - 8.1 Tabella riassuntiva BASE vs HC (configurazione operativa)
    - 8.2 Scrittura di `chosen_threshold.json` in `unet_HC/`

### Cosa cambia rispetto al Notebook 3 baseline (`vera_conn`)

- **Impianto fotometrico (decisione confermata).** La detection usa le probabilità HC di `unet_HC/val_prob_slices.npy`; la **massa resta una quantità I2**, ricostruita da `X_patches.npy` baseline con `MED_D`/`SIG_D` e calibrata con `mpu_per_slice.npy`. Il rapporto $T = I_1/I_2$ è adimensionale: da $T$ denormalizzato non si può ricavare massa fisica, e `mpu` è definito solo per la densità I2. Il confronto HC vs baseline è così **iso-fotometrico**: stessa fotometria, stessa verità FoF, cambia solo la detection. La fotometria I2 della sim HC è garantita dal check `verify_chain_HC` (§1.4).
- **Griglia 4×3 head-to-head (decisione confermata).** La stessa griglia (soglie $\times$ aree) gira su entrambi i set di probabilità: tabelle affiancate, $\Delta C$ per cella, e fit di massa in **ogni** cella.
- **Correzioni ereditate dal review del NB3 originale** (l'originale resta intatto; qui la prosa e la diagnostica sono già corrette):
    - *M1*: il linking collega due blob se la distanza XY è inferiore al **massimo** dei due raggi equivalenti (non alla "somma" come scritto nel testo originale), con floor di 3 px; il fattore di smorzamento **0.7** per i link a due passi ($s \to s+2$) è ora documentato.
    - *M2*: la completezza $C$ **non è monotona** in $t$: a soglie basse i blob si fondono (merging) e l'accoppiamento 1-a-1 perde aloni, a soglie alte frammentano. Il testo descrive ora questo comportamento.
    - *M3*: l'ispezione visiva del rumore campiona la **fetta del blob a probabilità massima** (non `floor(z/DZ)`): per oggetti collegati a salto la media pesata in $z$ può cadere su una fetta senza blob, e la diagnostica originale mostrava $p_{\max} \approx 0$ per oggetti reali.
    - *M4*: la tabella riassuntiva non ha righe duplicate e i riferimenti di sezione sono corretti.
    - *M5*: il fit di massa ($a$, $\sigma_{\mathrm{MAD}}$) è calcolato in ogni cella della griglia (colonne `a`, `sigma_MAD`): la scelta operativa legge la qualità della massa da dati, non da claim.
- **Contratti rispettati.** `dataset_info_HC.json` (schema annidato di NB1-HC) per la geometria e la normalizzazione T; `candidate_thresholds.json` di NB2-HC letto per le soglie candidate (con provenance del checkpoint); `chosen_threshold.json` scritto **solo** alla fine, dentro `unet_HC/`, senza toccare quello della baseline in `unet/`. Nessun RNG: il notebook è deterministico.
- **Output non inclusi.** Le celle non sono eseguite: figure e numeri si rigenerano rieseguendo il notebook. I numeri della **baseline** citati nel testo (es. $C = 0.818$, $P_{\mathrm{phys}} = 0.9968$, $a = 0.884$, $\sigma_{\mathrm{MAD}} = 0.133$ dex, ceiling $99.4\%$) vengono dal run originale di `vera_conn` su LH_105 e servono da ancoraggio per il confronto.


## 1. Setup e Caricamento Dati (HC)

L'obiettivo è lo stesso del Notebook 3 baseline: trasformare le mappe di probabilità della U-Net in un **catalogo fisico 3D** di aloni, con coordinate $(x, y, z)$ e massa stimata $M$. La differenza è che ora esistono **due run di probabilità** prodotte dallo stesso protocollo di training di NB2: la baseline (input standard, in `unet/`) e l'alto contrasto (input HC, in `unet_HC/`). Il notebook esegue la pipeline di catalogazione su entrambe e le confronta a parità di tutto il resto, così ogni differenza nei cataloghi è attribuibile alla sola detection.

L'impianto è **duale** sul lato input e asimmetrico nei ruoli. Le due directory dei dati (`sim_<id>` baseline e `sim_<id>_highcontrast`, entrambe sotto `input_data/`) non vengono mai scritte: la directory HC fornisce i metadati (`dataset_info_HC.json`, schema annidato) e una copia di controllo di `mpu_per_slice.npy`; la directory baseline fornisce la **fotometria** (densità ricostruita dalle patch standard) e la calibrazione di massa. Entrambe le directory di run (`unet/`, `unet_HC/`) sono lette in sola lettura tranne una: `unet_HC/` riceve il `chosen_threshold.json` finale (§8).

### 1.1 Due directory di input e due run di probabilità



```python
# ============================================================
# 1 — SETUP, PERCORSI DUALI E CONFIGURAZIONE (HC)
# ============================================================
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import h5py
import hdf5plugin  # (side-effect: registra i filtri di compressione HDF5 per h5py)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- percorsi (stesso schema di NB2-HC v2) ----------------
try:
    from google.colab import drive                                #type: ignore
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/CAMELS_data'
except ImportError:
    DATA_DIR = r"C:\Users\kekko\Downloads\Python\TESI\CAMELS_data"

IN_ROOT      = os.path.join(DATA_DIR, "input_data")
SIM_ROOT     = os.path.join(DATA_DIR, "Simulations")
RUN_DIR      = os.path.join(DATA_DIR, "unet_HC")   # probabilità HC (NB2-HC)
RUN_DIR_BASE = os.path.join(DATA_DIR, "unet")      # probabilità baseline (NB2), solo lettura

INFO_FILENAME = "dataset_info_HC.json"             # schema annidato di NB1-HC
DATA_SUFFIX   = "_highcontrast"

# ---------------- simulazione di validation ----------------
RUN_NAME = "LH_105"
SNAP_TAG = "090"
SIM_ID   = f"{RUN_NAME}_{SNAP_TAG}"
D_BASE   = os.path.join(IN_ROOT, f"sim_{SIM_ID}")                # baseline NB1 (sola lettura)
D_HC     = os.path.join(IN_ROOT, f"sim_{SIM_ID}{DATA_SUFFIX}")   # alto contrasto (sola lettura)

# ---------------- configurazione ----------------
N_SLICES, IMG_SIZE, PATCH_SIZE = 25, 1024, 256
RHO_CRIT0 = 2.7754e11   # densità critica z=0, unità (M_sun/h)/(Mpc/h)^3
                        # (a z=0.090 rho_c è ~9% più alta: R200_est è solo una
                        #  tolleranza di matching, l'effetto ~3% è trascurabile)

SOGLIE_CANDIDATE = [0.85, 0.90, 0.95, 0.99]   # default; la cella §1.2 le allinea a candidate_thresholds.json
AREE_CANDIDATE   = [4, 6, 12]                 # filtri d'area 3D [px]

# ---------------- algoritmo (identici a NB3 baseline) ----------------
MIN_AREA_2D_PX = 1        # filtro di sicurezza 2D (tiene tutto)
LINK_FLOOR_PX  = 3        # floor linking [px]
MIN_TOL_PX     = 3        # floor matching [px]
LINK_NEIGHBORS = 2        # collega s->s+1 (scala 1.0) e s->s+2 (scala 0.7)

# ---------------- i due set di probabilità ----------------
MODES = ["BASE", "HC"]    # BASE = unet/ (baseline), HC = unet_HC/ (alto contrasto)

print(f"NB3-HC (conn_HC) | sim {SIM_ID}")
print(f"  input : {D_BASE}  +  {D_HC}")
print(f"  probs : BASE <- {RUN_DIR_BASE} | HC <- {RUN_DIR}")

```

    NB3-HC (conn_HC) | sim LH_105_090
      input : C:\Users\kekko\Downloads\Python\TESI\CAMELS_data\input_data\sim_LH_105_090  +  C:\Users\kekko\Downloads\Python\TESI\CAMELS_data\input_data\sim_LH_105_090_highcontrast
      probs : BASE <- C:\Users\kekko\Downloads\Python\TESI\CAMELS_data\unet | HC <- C:\Users\kekko\Downloads\Python\TESI\CAMELS_data\unet_HC
    

### 1.2 Impianto fotometrico: perché la massa resta I2

**La detection è HC, la massa è I2.** Nel ramo HC l'input della rete è $T = I_1/I_2$: un rapporto tra due proiezioni, quindi **adimensionale**. La normalizzazione per-sim (`MEDIAN`, `SIGMA` in `dataset_info_HC.json`) rende $T$ comparabile tra simulazioni, ma non introduce nessuna unità di massa: da $10^{X_{HC}\cdot\sigma_{HC}+\mu_{HC}}$ si ricostruisce $T$, non $\rho$. Il fattore `mpu` (massa per unità di mappa), ricavato in NB1 dalla conservazione della massa totale della fetta, è calibrato **sulla densità I2** ed è quindi l'unico ponte valido verso i $M_\odot/h$.

Di conseguenza la massa di ogni blob resta quella del NB3 baseline
$$
M_{\mathrm{blob}} = \Big(\sum_{p \in \mathrm{blob}} \rho^{I2}_p\Big) \cdot \mathrm{mpu}_{\mathrm{slice}},
\qquad
\rho^{I2}(x,y) = 10^{\,X_{base}(x,y)\cdot \mathrm{SIG\_D} + \mathrm{MED\_D}},
$$
mentre i **pixel** che compongono i blob arrivano dalla binarizzazione delle probabilità HC. I pixel vuoti hanno $\rho^{I2}=0$ per costruzione (cutoff su `EMPTY_FILL` baseline), quindi contribuiscono massa zero anche se la rete HC vi accendesse una probabilità spuria; il rischio residuo — detezioni interamente in zona vuota che gonfiano il rumore — è misurato direttamente in §7.1. La cella seguente legge entrambi i file di metadati (schema piatto per il baseline, annidato per l'HC), verifica che la griglia coincida, e allinea le soglie candidate al contratto con NB2-HC (`candidate_thresholds.json`, con provenance del checkpoint).



```python
# ============================================================
# 1.2 — METADATI DUALI E SOGLIE CANDIDATE (contratto con NB2-HC)
# ============================================================
with open(os.path.join(D_BASE, "dataset_info.json")) as fh:
    info_base = json.load(fh)
with open(os.path.join(D_HC, INFO_FILENAME)) as fh:
    info_hc = json.load(fh)

# --- fotometria: normalizzazione I2 dal baseline (chiavi piatte di NB1) ---
LBOX     = float(info_base["LBOX_MPC_H"])
MASS_CUT = float(info_base["MASS_CUT_MSUNH"])
SIG_D    = float(info_base["SIG_D"])
MED_D    = float(info_base["MED_D"])

# --- HC: schema annidato di NB1-HC (geometria + normalizzazione T) ---
p_hc = info_hc["PHYSICS_AND_GRID"]
norm = info_hc["INPUTS_FOR_UNET"]["NORMALIZATION"]
EMPTY_FILL_HC = float(norm["EMPTY_FILL"])

# La griglia HC deve coincidere con quella baseline (NB1-HC lo garantisce; qui lo verifichiamo)
assert LBOX == float(p_hc["LBOX_MPC_H"]),     "LBOX diverso tra baseline e HC"
assert N_SLICES == int(p_hc["N_SLICES"]),     "N_SLICES diverso tra baseline e HC"
assert IMG_SIZE == int(p_hc["IMG_SIZE"]),     "IMG_SIZE diverso tra baseline e HC"
assert PATCH_SIZE == int(p_hc["PATCH_SIZE"]), "PATCH_SIZE diverso tra baseline e HC"

# --- soglie candidate: si legge candidate_thresholds.json scritto da NB2-HC ---
CAND_PATH = os.path.join(RUN_DIR, "candidate_thresholds.json")
with open(CAND_PATH) as fh:
    cand = json.load(fh)
SOGLIE_CANDIDATE = [float(t) for t in cand["candidates"]]
prov = cand.get("provenance", {})

print(f"candidate_thresholds.json: soglie candidate = {SOGLIE_CANDIDATE}")
print(f"  provenance: epoca checkpoint = {prov.get('checkpoint_epoch')} | "
      f"val AP = {prov.get('checkpoint_val_ap')} | torch = {prov.get('torch_version')}")
print(f"normalizzazioni: I2 med={MED_D:.4f} sig={SIG_D:.4f} | "
      f"T_HC med={float(norm['MEDIAN']):.4f} sig={float(norm['SIGMA']):.4f} | "
      f"EMPTY_FILL={EMPTY_FILL_HC}")
print(f"MASS_CUT = {MASS_CUT:.3e} M_sun/h | LBOX = {LBOX:.1f} Mpc/h")

```

    candidate_thresholds.json: soglie candidate = [0.85, 0.9, 0.95, 0.99]
      provenance: epoca checkpoint = 17 | val AP = 0.8549388389214116 | torch = 2.14.1+cpu
    normalizzazioni: I2 med=10.0451 sig=0.5409 | T_HC med=10.2925 sig=0.6470 | EMPTY_FILL=-3.0
    MASS_CUT = 9.631e+09 M_sun/h | LBOX = 25.0 Mpc/h
    

### 1.3 Caricamento: densità I2, `mpu`, probabilità e cataloghi

La densità I2 si ricostruisce dalle patch baseline invertendo lo z-score robusto di NB1, esattamente come nel NB3 originale (cutoff $X \le -2.9$ sui pixel vuoti, robusto a errori float32): questo è il vincolo di iso-fotometria, non un dettaglio. Le probabilità arrivano dai due run di NB2 (`unet/val_prob_slices.npy` e `unet_HC/val_prob_slices.npy`, attese $(25, 1024, 1024)$ come la densità). I cataloghi di verità sono gli stessi per entrambe le detection: Group FoF (con e senza `MASS_CUT`) e Subfind, con la stessa pulizia del notebook originale.



```python
# ============================================================
# 1.3 — CARICAMENTO: DENSITÀ I2 (FOTOMETRIA), MPU, PROBABILITÀ, CATALOGHI
# ============================================================
def tiles_to_slices(tiles, patch, img):
    """Ricostruisce fette 1024². Ordine: indice = slice*16 + gy*4 + gx."""
    n_grid = img // patch
    n = tiles.shape[0] // n_grid ** 2
    out = np.empty((n, img, img), dtype=tiles.dtype)
    for i in range(n):
        for k in range(n_grid ** 2):
            gy, gx = divmod(k, n_grid)
            out[i, gy*patch:(gy+1)*patch, gx*patch:(gx+1)*patch] = tiles[i*n_grid**2 + k]
    return out

# ---------------- densità I2: IDENTICA a NB3 baseline (iso-fotometria) ----------------
Xn = tiles_to_slices(
        np.asarray(np.load(os.path.join(D_BASE, "X_patches.npy"), mmap_mode="r")[:, 0], dtype=np.float32),
        PATCH_SIZE, IMG_SIZE)
# Pixel vuoti baseline riempiti a -3.0 da NB1: cutoff <= -2.9, robusto a errori float32
raw = np.where(Xn <= -2.9, 0.0, 10.0 ** (Xn * SIG_D + MED_D)).astype(np.float32)
del Xn

# mpu: calibrazione di massa da I2. La copia HC deve coincidere (check esteso in §1.4)
mpu    = np.load(os.path.join(D_BASE, "mpu_per_slice.npy"))
mpu_hc = np.load(os.path.join(D_HC,  "mpu_per_slice.npy"))
assert np.allclose(mpu, mpu_hc, rtol=1e-6), "mpu_per_slice.npy divergente tra baseline e HC"

# ---------------- probabilità U-Net: due run separate ----------------
probs_by_mode = {
    "BASE": np.load(os.path.join(RUN_DIR_BASE, "val_prob_slices.npy")).astype(np.float32),
    "HC":   np.load(os.path.join(RUN_DIR,      "val_prob_slices.npy")).astype(np.float32),
}
for m in MODES:
    assert probs_by_mode[m].shape == raw.shape, \
        f"probs {m}: {probs_by_mode[m].shape} != raw {raw.shape}"

# ---------------- cataloghi (Group + Subhalo) ----------------
CAT_PATH = os.path.join(SIM_ROOT, RUN_NAME, f"groups_{RUN_NAME}", f"groups_{SNAP_TAG}_{RUN_NAME}.hdf5")
with h5py.File(CAT_PATH, "r") as f:
    g  = f["Group"]
    sh = f["Subhalo"]
    fof_full = {
        "pos":  g["GroupPos"][:].astype(np.float64) / 1000.0,
        "r200": g["Group_R_Crit200"][:].astype(np.float64) / 1000.0,
        "mass": g["GroupMass"][:].astype(np.float64) * 1e10,
    }
    sub_full = {
        "pos":  sh["SubhaloPos"][:].astype(np.float64) / 1000.0,
        "mass": sh["SubhaloMass"][:].astype(np.float64) * 1e10,
        "r200": np.maximum(2.0 * sh["SubhaloHalfmassRad"][:].astype(np.float64) / 1000.0,
                           3.0 * (LBOX / IMG_SIZE)),
    }

# Pulizia cataloghi (identica a NB3 baseline)
keep = np.isfinite(fof_full["pos"]).all(1) & (fof_full["r200"] > 0)
for k in fof_full: fof_full[k] = fof_full[k][keep]
keep = np.isfinite(sub_full["pos"]).all(1) & (sub_full["mass"] > 0)
for k in sub_full: sub_full[k] = sub_full[k][keep]

# Catalogo tagliato (target di training)
sel = fof_full["mass"] >= MASS_CUT
fof = {k: v[sel] for k, v in fof_full.items()}

print(f"mappe  : raw I2 {raw.shape} | probs {probs_by_mode['BASE'].shape}")
print(f"Group  : {len(fof_full['mass']):,} totali -> {len(fof['mass']):,} sopra MASS_CUT={MASS_CUT:.2e}")
print(f"Subhalo: {len(sub_full['mass']):,}")

```

    mappe  : raw I2 (25, 1024, 1024) | probs (25, 1024, 1024)
    Group  : 12,304 totali -> 4,701 sopra MASS_CUT=9.63e+09
    Subhalo: 16,223
    

### 1.4 Verifica della catena HC e geometria derivata

Prima di calcolare qualunque massa, rieseguiamo qui i due check di `verify_chain_HC` di NB1-HC sulla sim di validation: (1) la catena $T$ si ricostruisce da $X_{HC}$ con i parametri per-sim (`MEDIAN`, `SIGMA`, `EMPTY_FILL`); (2) la fetta I2 salvata nella directory HC è **identica** a quella baseline. Il check (2) è la garanzia che il rendering HC abbia usato gli stessi parametri dell'originale e che `mpu_per_slice.npy` sia valido per la fotometria di questo notebook: senza di esso, qualunque differenza nelle masse sarebbe sospetta. Seguono la geometria derivata (pixel, fette, floor di matching) e il conteggio dei pixel vuoti, che la §7.1 userà come maschera.



```python
# ============================================================
# 1.4 — VERIFY_CHAIN_HC (da NB1-HC, rieseguito qui su LH_105) E GEOMETRIA
# ============================================================
def verify_chain_HC_val():
    """Due check indipendenti (quelli di NB1-HC, rieseguiti qui da NB3-HC, sola lettura):
    (1) catena T: 10^(X_HC*sig_hc+med_hc) ricostruisce raw_slice_sample_HC.npy;
    (2) uguaglianza I2: raw_slice_sample_I2.npy == sim_<id>/raw_slice_sample.npy.
    Il check (2) garantisce che il rendering HC abbia usato gli stessi parametri
    dell'originale: senza di esso mpu non sarebbe valido per la fotometria di questo notebook."""
    raw_slice = int(info_hc["CALIBRATION_FOR_MASS"]["RAW_SAMPLE_SLICE"])
    med_hc, sig_hc = float(norm["MEDIAN"]), float(norm["SIGMA"])

    Xh = tiles_to_slices(
        np.asarray(np.load(os.path.join(D_HC, "X_patches_HC.npy"), mmap_mode="r")[:, 0], dtype=np.float32),
        PATCH_SIZE, IMG_SIZE)[raw_slice]
    with np.errstate(over="ignore", invalid="ignore"):
        T_rec = (10.0 ** (Xh.astype(np.float64) * sig_hc + med_hc)).astype(np.float32)
    T_rec[Xh == EMPTY_FILL_HC] = 0.0

    T_gt = np.load(os.path.join(D_HC, "raw_slice_sample_HC.npy"))
    valid = Xh != EMPTY_FILL_HC
    rel_T = np.abs(T_rec[valid] - T_gt[valid]) / np.maximum(T_gt[valid], 1e-30)

    I2_hc = np.load(os.path.join(D_HC, "raw_slice_sample_I2.npy")).astype(np.float64)
    I2_or = np.load(os.path.join(D_BASE, "raw_slice_sample.npy")).astype(np.float64)
    rel_I2 = np.abs(I2_hc - I2_or) / np.maximum(np.abs(I2_or), 1e-30)

    ok = (rel_T.max() < 1e-4 and int((rel_T > 1e-4).sum()) == 0
          and rel_I2.max() < 1e-4 and int((rel_I2 > 1e-4).sum()) == 0
          and bool(np.all(T_rec[~valid] == 0.0)))
    print(f"verify_chain_HC ({RUN_NAME}, fetta {raw_slice}): {'✓ OK' if ok else '✗ FAIL'}")
    print(f"  T round-trip: max rel err = {rel_T.max():.2e} | fuori tolleranza: {int((rel_T > 1e-4).sum())}")
    print(f"  I2 identico : max rel err = {rel_I2.max():.2e} | fuori tolleranza: {int((rel_I2 > 1e-4).sum())}")
    assert ok, "catena HC rotta: la fotometria I2 non è garantita per questa sim"

verify_chain_HC_val()

# ---------------- geometria derivata ----------------
NS             = raw.shape[0]
IMG            = raw.shape[1]
PX_MPC         = LBOX / IMG
DZ             = LBOX / NS
MIN_TOL_MPC    = MIN_TOL_PX * PX_MPC
LINK_FLOOR_MPC = LINK_FLOOR_PX * PX_MPC
STRUCT8        = np.ones((3, 3), bool)

print(f"\nSimulazione {SIM_ID}: LBOX={LBOX:.1f} Mpc/h | DZ={DZ:.2f} Mpc/h | PX={PX_MPC:.4f} Mpc/h")
print(f"pixel vuoti I2: {int((raw == 0).sum()):,} ({100*(raw == 0).mean():.2f}% della mappa)")

```

    verify_chain_HC (LH_105, fetta 12): ✓ OK
      T round-trip: max rel err = 1.56e-06 | fuori tolleranza: 0
      I2 identico : max rel err = 0.00e+00 | fuori tolleranza: 0
    
    Simulazione LH_105_090: LBOX=25.0 Mpc/h | DZ=1.00 Mpc/h | PX=0.0244 Mpc/h
    pixel vuoti I2: 0 (0.00% della mappa)
    

## 2. Algoritmi della Pipeline

Le funzioni sono **identiche** a quelle del NB3 baseline: l'unica differenza è che il set di probabilità (`BASE` o `HC`) viene passato esplicitamente, quindi ogni differenza nei cataloghi è attribuibile alla detection e il confronto della §5 usa esattamente lo stesso codice per entrambe. La cache dei blob diventa per $(\mathrm{mode}, t)$: la doppia griglia $4\times3$ riusa le estrazioni senza ricalcoli identici.

1. **Estrazione Blob 2D** (`extract_blobs`): binarizzazione a soglia $t$ con labeling su mosaico $3\times 3$ per la periodicità XY completa (8-connesso). Per ogni blob si calcolano: centroide circolare robusto al wrap (media di $\sin$/$\cos$ con `arctan2`), massa $m_{\rm blob} = (\sum_p \rho^{I2}_p)\cdot \mathrm{mpu}_{\rm slice}$, raggio equivalente $r_{\rm eq} = \sqrt{A/\pi}$, probabilità di picco e indice di fetta.
2. **Linking 3D** (`link_blobs`): due blob in fette $s \to s+1$ o $s \to s+2$ si collegano se la loro distanza proiettata in XY è inferiore al **massimo tra i due raggi equivalenti** (con un floor di $3\,\mathrm{px}$ per evitare frammentazioni spurie). Per il collegamento a due passi ($s \to s+2$, utile quando la fetta intermedia non produce blob) la soglia è smorzata del fattore **0.7**. *(Nota: il testo del NB3 originale parlava di "somma dei raggi" mentre il codice usa il massimo dei due; la descrizione qui è allineata al codice effettivamente eseguito — correzione M1.)*
3. **Costruzione Catalogo** (`build_catalog`): i blob connessi diventano un unico oggetto 3D, con massa somma dei blob, coordinate $(x,y,z)$ come media circolare pesata per la massa e $R_{200}^{\mathrm{est}} = (3M/4\pi\cdot 200\,\rho_c)^{1/3}$ come tolleranza di matching. $\rho_c$ è valutata a $z=0$: a $z=0.090$ sarebbe $\sim 9\%$ più alta ($R$ $\sim 3\%$ più piccolo) — effetto trascurabile su una tolleranza.

**Matching** (`match_catalogs`): greedy 1-a-1 con KD-tree periodico in XY; due oggetti matchano se $d_{XY} \le \max(R_{200}^{\mathrm{FoF}},\, 3\,\mathrm{px})$ e $|\Delta z| \le dz$. **Classificazione FP** (`classify_fps`): le predizioni non accoppiate sono classificate in *sotto cut*, *doppia*, *subhalo*, *rumore*, e la purity fisica $P_{\mathrm{phys}} = (N_{\mathrm{match}} + N_{\mathrm{below}} + N_{\mathrm{sub}})/N_{\mathrm{pred}}$ conta come validi anche gli aloni reali invisibili al target.



```python
# ============================================================
# 2 — FUNZIONI CORE (identiche a NB3 baseline, parametrizzate sul set di probs)
# ============================================================
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree

# ---- cache blob per (mode, soglia): la doppia griglia 4x3 non ricalcola nulla ----
_BLOB_CACHE = {}

def get_blobs(mode, t):
    k = (mode, round(float(t), 4))
    if k not in _BLOB_CACHE:
        _BLOB_CACHE[k] = extract_blobs(probs_by_mode[mode], raw, mpu, LBOX, t=k[1],
                                       min_area=MIN_AREA_2D_PX)
    return _BLOB_CACHE[k]

def get_linked(mode, t):
    k = ("linked", mode, round(float(t), 4))
    if k not in _BLOB_CACHE:
        _BLOB_CACHE[k] = link_blobs(get_blobs(mode, t), LBOX, n_neighbors=LINK_NEIGHBORS)
    return _BLOB_CACHE[k]

```


```python
# ---- distanza periodica XY (come NB3 baseline) ----
def pdist_periodic(a, b, lbox):
    dx = (a[:, None, 0] - b[None, :, 0] + lbox/2) % lbox - lbox/2
    dy = (a[:, None, 1] - b[None, :, 1] + lbox/2) % lbox - lbox/2
    return np.hypot(dx, dy)

def extract_blobs(probs, raw_map, mpu_arr, lbox, t, min_area=1):
    """Estrae blob 2D a soglia t. Labeling su mosaico 3x3 per periodicità XY completa (8-conn)."""
    B = {k: [] for k in ("x", "y", "mass", "area", "r_eq", "prob", "slice")}
    for s in range(NS):
        lab_big, _ = nd_label(np.tile(probs[s] > t, (3, 3)), STRUCT8)
        core = lab_big[IMG:2*IMG, IMG:2*IMG]
        ys, xs = np.nonzero(core)
        if len(ys) == 0:
            continue
        lb = core[ys, xs]
        uniq, cid = np.unique(lb, return_inverse=True)
        k = len(uniq)

        area = np.bincount(cid, minlength=k).astype(np.float64)
        mass = np.bincount(cid, weights=raw_map[s][ys, xs].astype(np.float64),
                           minlength=k) * float(mpu_arr[s])
        ax_ = 2 * np.pi * (xs + 0.5) / IMG
        ay_ = 2 * np.pi * (ys + 0.5) / IMG
        xc = (np.arctan2(np.bincount(cid, np.sin(ax_), minlength=k),
                         np.bincount(cid, np.cos(ax_), minlength=k)) / (2*np.pi)) % 1.0 * lbox
        yc = (np.arctan2(np.bincount(cid, np.sin(ay_), minlength=k),
                         np.bincount(cid, np.cos(ay_), minlength=k)) / (2*np.pi)) % 1.0 * lbox
        r_eq = np.sqrt(area / np.pi) * PX_MPC

        # peak via np.maximum.reduceat (più rapido di ndimage.maximum, stesso risultato)
        order = np.argsort(cid, kind="stable")
        starts = np.searchsorted(cid[order], np.arange(k))
        peak = np.maximum.reduceat(probs[s][ys[order], xs[order]], starts)

        good = area >= min_area
        B["x"] += xc[good].tolist();      B["y"] += yc[good].tolist()
        B["mass"] += mass[good].tolist(); B["area"] += area[good].tolist()
        B["r_eq"] += r_eq[good].tolist(); B["prob"] += peak[good].tolist()
        B["slice"] += [s] * int(good.sum())
    return {k: np.asarray(v) for k, v in B.items()}

def link_blobs(B, lbox, n_neighbors=LINK_NEIGHBORS):
    """Collega blob in fette s->s+k (k=1: scala 1.0; k=2: scala 0.7). La distanza XY deve
    essere <= max(r_eq dei due blob, floor 3 px): il MASSIMO dei due raggi, non la somma (M1)."""
    n = len(B["x"])
    edges = []
    for k in range(1, n_neighbors + 1):
        scale = 1.0 if k == 1 else 0.7
        for s in range(NS):
            s2 = (s + k) % NS
            ia = np.flatnonzero(B["slice"] == s)
            ib = np.flatnonzero(B["slice"] == s2)
            if len(ia) == 0 or len(ib) == 0:
                continue
            d = pdist_periodic(np.c_[B["x"][ia], B["y"][ia]],
                               np.c_[B["x"][ib], B["y"][ib]], lbox)
            link_len = scale * np.maximum(
                np.maximum(B["r_eq"][ia][:, None], B["r_eq"][ib][None, :]), LINK_FLOOR_MPC)
            ii, jj = np.nonzero(d <= link_len)
            edges.append(np.c_[ia[ii], ib[jj]])
    edges = np.concatenate(edges) if edges else np.zeros((0, 2), int)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return comp

```


```python
# ---- build_catalog: vettorizzato con bincount (come NB3 baseline) ----
def build_catalog(B, comp, lbox, dz, min_area_3d):
    z_c = (np.arange(NS) + 0.5) * dz
    m, a = B["mass"], B["area"]
    uniq, cid = np.unique(comp, return_inverse=True)
    n = len(uniq)
    M = np.bincount(cid, weights=m, minlength=n)
    A = np.bincount(cid, weights=a, minlength=n)
    th  = 2 * np.pi * z_c[B["slice"]] / lbox
    thx = 2 * np.pi * B["x"] / lbox
    thy = 2 * np.pi * B["y"] / lbox
    z = (np.arctan2(np.bincount(cid, m*np.sin(th),  minlength=n),
                    np.bincount(cid, m*np.cos(th),  minlength=n)) / (2*np.pi)) % 1.0 * lbox
    x = (np.arctan2(np.bincount(cid, m*np.sin(thx), minlength=n),
                    np.bincount(cid, m*np.cos(thx), minlength=n)) / (2*np.pi)) % 1.0 * lbox
    y = (np.arctan2(np.bincount(cid, m*np.sin(thy), minlength=n),
                    np.bincount(cid, m*np.cos(thy), minlength=n)) / (2*np.pi)) % 1.0 * lbox
    r_est = (3 * M / (4 * np.pi * 200 * RHO_CRIT0)) ** (1/3)
    keep = A >= min_area_3d
    return {"x": x[keep], "y": y[keep], "z": z[keep],
            "mass": M[keep], "r200_est": r_est[keep], "area_tot": A[keep]}

# ---- match_catalogs: cKDTree periodico (come NB3 baseline) ----
def match_catalogs(pred, ref, lbox, dz, tol_scale=1.0):
    n_pred, n_ref = len(pred["mass"]), len(ref["mass"])
    if n_pred == 0 or n_ref == 0:
        return np.zeros((0, 2), int), np.zeros(n_pred, bool), np.zeros(n_ref, bool)
    r_match = np.maximum(ref["r200"], MIN_TOL_MPC) * tol_scale
    rmax = float(r_match.max()) * 1.01
    tree = cKDTree(ref["pos"][:, :2], boxsize=lbox)
    xy_pred = np.column_stack([pred["x"], pred["y"]])
    z_pred  = pred["z"]
    rxy, rz = ref["pos"][:, :2], ref["pos"][:, 2]

    ci, cj, cd = [], [], []
    for i, js in enumerate(tree.query_ball_point(xy_pred, r=rmax)):
        if not js:
            continue
        js = np.asarray(js, dtype=int)
        dx = (xy_pred[i, 0] - rxy[js, 0] + lbox/2) % lbox - lbox/2
        dy = (xy_pred[i, 1] - rxy[js, 1] + lbox/2) % lbox - lbox/2
        dxy = np.hypot(dx, dy)
        ok = dxy <= r_match[js]
        if not ok.any():
            continue
        dzp = (z_pred[i] - rz[js] + lbox/2) % lbox - lbox/2
        ok = ok & (np.abs(dzp) <= dz)
        if not ok.any():
            continue
        js_ok = js[ok]
        ci += [i] * len(js_ok); cj += js_ok.tolist(); cd += dxy[ok].tolist()

    if not ci:
        return np.zeros((0, 2), int), np.zeros(n_pred, bool), np.zeros(n_ref, bool)

    ci, cj, cd = np.asarray(ci, int), np.asarray(cj, int), np.asarray(cd)
    order = np.lexsort((cj, ci, cd))
    pm, fm = np.zeros(n_pred, bool), np.zeros(n_ref, bool)
    pairs = []
    for k in order:
        i, j = ci[k], cj[k]
        if not (pm[i] or fm[j]):
            pairs.append((i, j)); pm[i] = fm[j] = True
    return np.asarray(pairs, int).reshape(-1, 2), pm, fm

# ---- classify_fps (come NB3 baseline) ----
def classify_fps(pred, pm, fof_full_ref, sub_full_ref, lbox, dz):
    fp_idx = np.flatnonzero(~pm)
    if len(fp_idx) == 0:
        return {"n_below": 0, "n_double": 0, "n_sub": 0, "n_noise": 0, "P_phys": 1.0}
    pred_fp = {k: pred[k][fp_idx] for k in pred}
    pairs_g, pm_g, _ = match_catalogs(pred_fp, fof_full_ref, lbox, dz)
    _,       pm_s, _ = match_catalogs(pred_fp, sub_full_ref, lbox, dz)
    mass_g = np.full(len(fp_idx), np.nan)
    for i, j in pairs_g:
        mass_g[i] = fof_full_ref["mass"][j]
    cat = np.full(len(fp_idx), "", dtype=object)
    cat[pm_g & (mass_g <  MASS_CUT)] = "sotto cut"
    cat[pm_g & (mass_g >= MASS_CUT)] = "doppia"
    cat[(~pm_g) & pm_s]              = "subhalo"
    cat[(~pm_g) & (~pm_s)]           = "rumore"
    n_below  = int((cat == "sotto cut").sum())
    n_double = int((cat == "doppia").sum())
    n_sub    = int((cat == "subhalo").sum())
    n_noise  = int((cat == "rumore").sum())
    P_phys   = (pm.sum() + n_below + n_sub) / len(pred["mass"])
    return {"n_below": n_below, "n_double": n_double, "n_sub": n_sub,
            "n_noise": n_noise, "P_phys": float(P_phys)}

```


```python
# ---- metriche di massa e pipeline completa per un set di probabilità ----
def mass_stats(pairs, pred, fof_ref):
    lp = np.log10(pred["mass"][pairs[:, 0]])
    lt = np.log10(fof_ref["mass"][pairs[:, 1]])
    a, b = np.polyfit(lp, lt, 1)
    res_fit = lt - (a * lp + b)
    dlog = lt - lp
    sig_mad = 1.4826 * np.median(np.abs(res_fit - np.median(res_fit)))
    return {"lp": lp, "lt": lt, "dlog": dlog, "a": float(a), "b": float(b),
            "sigma_mad": float(sig_mad), "median_dlog": float(np.median(dlog))}

def run_pipeline(mode, t_val, area3d):
    """Pipeline completa per un set di probabilità (cache su (mode, t))."""
    B    = get_blobs(mode, t_val)
    comp = get_linked(mode, t_val)
    pred = build_catalog(B, comp, LBOX, DZ, min_area_3d=area3d)
    if len(pred["mass"]) == 0:
        return None
    pairs, pm, fm = match_catalogs(pred, fof, LBOX, DZ)
    fp_res = classify_fps(pred, pm, fof_full, sub_full, LBOX, DZ)
    n_pred = len(pred["mass"])
    return {"mode": mode, "t": float(t_val), "area": int(area3d),
            "n_pred": int(n_pred), "n_match": int(len(pairs)),
            "C": float(fm.sum() / len(fof["mass"])),
            "P_raw": float(pm.sum() / n_pred),
            "P_phys": fp_res["P_phys"],
            "n_below": fp_res["n_below"], "n_double": fp_res["n_double"],
            "n_sub": fp_res["n_sub"], "n_noise": fp_res["n_noise"],
            "pred": pred, "pairs": pairs, "pm": pm, "fm": fm}

```

## 3. Calibrazione della massa e diagnosi del bias

Prima di lanciare la doppia griglia verifichiamo che la fisica della pipeline funzioni **con entrambe le detection**, alla configurazione di riferimento della baseline $(t = 0.95,\ \mathrm{area} = 4)$. La massa è la stessa fotometria I2 per entrambe; cambia solo l'insieme dei pixel che compongono i blob, quindi anche il loro aggregato 3D. Il fit log-log
$$
\log_{10} M_{\mathrm{FoF}} = a \cdot \log_{10} M_{\mathrm{pred}} + b
$$
misura così direttamente quanto la *detection* HC — quanti pixel raccoglie, come li aggrega in fette adiacenti — modifica la stima di massa rispetto alla baseline. Nel run originale della baseline su LH_105 il fit dava $a = 0.884$, $\sigma_{\mathrm{MAD}} = 0.133$ dex, con $C = 0.818$ e $P_{\mathrm{phys}} = 0.9968$: sono i numeri di riferimento da confermare o battere.

### 3.1 Pipeline su entrambe le detection e fit di massa



```python
# ============================================================
# 3.1 — PIPELINE SU ENTRAMBE LE DETECTION E FIT DI MASSA (t=0.95, area=4)
# ============================================================
T_BASE, AREA_BASE = 0.95, 4
res = {m: run_pipeline(m, T_BASE, AREA_BASE) for m in MODES}
ms  = {m: mass_stats(res[m]["pairs"], res[m]["pred"], fof) for m in MODES}

for m in MODES:
    r, s = res[m], ms[m]
    print(f"[{m:>4}] t={T_BASE}, area={AREA_BASE}: n_pred={r['n_pred']:,} | "
          f"match={r['n_match']:,} | C={r['C']:.3f} | P_phys={r['P_phys']:.4f} | "
          f"fit a={s['a']:.3f}, sigma_MAD={s['sigma_mad']:.3f} dex")

# --- plot comparativo a due pannelli (rosso = HC, blu = baseline; convenzione di NB2-HC v2) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

all_v = np.concatenate([ms["HC"]["lp"], ms["HC"]["lt"], ms["BASE"]["lp"], ms["BASE"]["lt"]])
lims = 10.0 ** np.array([np.floor(all_v.min()), np.ceil(all_v.max())])
xx = np.log10(lims)

panels = [("HC", "crimson", "HC (detection alto contrasto)"),
          ("BASE", "tab:blue", "Baseline (detection standard)")]
for ax, (m, color, title) in zip(axes, panels):
    s = ms[m]
    ax.scatter(10**s["lp"], 10**s["lt"], s=8, alpha=0.3, color=color,
               edgecolors="none", rasterized=True, label=f"{len(res[m]['pairs']):,} coppie")
    ax.plot(lims, lims, "--", color="0.4", lw=1.4, label="1:1")
    ax.plot(lims, 10**(s["a"]*xx + s["b"]), color="k", lw=2,
            label=rf"Fit: (a={s['a']:.3f}, $\sigma_{{MAD}}$={s['sigma_mad']:.3f})")
    ax.axhline(MASS_CUT, color="0.2", ls=":", lw=1.5, alpha=0.6, label=f"MASS_CUT = {MASS_CUT:.1e}")
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
           xlabel=r"$M_{\mathrm{pred}}\ [M_\odot/h]$",
           ylabel=r"$M_{\mathrm{FoF}}\ [M_\odot/h]$",
           title=f"{title}\nC={res[m]['C']:.3f}, P_phys={res[m]['P_phys']:.4f}")
    ax.grid(True, which="major", ls=":", color="0.75")
    ax.legend(frameon=False, loc="lower right", fontsize=9)

plt.suptitle(f"Fit di massa a t={T_BASE}, area={AREA_BASE}: "
             "fotometria I2 identica, cambia solo la detection", fontsize=13)
plt.show()

```

    [BASE] t=0.95, area=4: n_pred=5,962 | match=3,846 | C=0.818 | P_phys=0.9968 | fit a=0.884, sigma_MAD=0.133 dex
    [  HC] t=0.95, area=4: n_pred=5,747 | match=3,999 | C=0.851 | P_phys=0.9972 | fit a=0.866, sigma_MAD=0.129 dex
    


    
![png](vera_conn_HC_files/vera_conn_HC_15_1.png)
    


### 3.2 Errore relativo di massa binnato

Come nel NB3 originale, dividiamo l'errore relativo $(M_{\mathrm{pred}} - M_{\mathrm{true}})/M_{\mathrm{true}}$ in bin di massa vera, con mediana (bias) e scatter $1\sigma = (P_{84}-P_{16})/2$ (incertezza), richiedendo almeno 5 oggetti per bin. Nel run baseline la mediana era vicina a zero a tutte le masse, con barre larghe sotto $10^{10}\,M_\odot/h$ (fino a $\pm 0.5$) e strette per gli aloni massicci ($\pm 0.10$ sopra $10^{11.5}$), più una leggera sovrastima sistematica dei massicci (materia diffusa attorno all'alone inclusa nella fotometria). Qui le due detection sono sovrapposte sullo stesso grafico: se HC raccoglie pixel diversi attorno ai nuclei, la curva rossa si sposta (bias diverso) o si restringe (incertezza minore) rispetto a quella blu.



```python
# ============================================================
# 3.2 — ERRORE RELATIVO DI MASSA BINNATO (BASE vs HC)
# ============================================================
bins = np.logspace(np.log10(MASS_CUT), np.log10(max(fof["mass"].max(), 10**13)), 12)
bin_centers = np.sqrt(bins[:-1] * bins[1:])

fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
ax.axhline(0, color="black", lw=1.2, ls="--", label="Errore nullo (ideale)")

for m, color, marker in [("HC", "crimson", "o"), ("BASE", "tab:blue", "s")]:
    pairs_m = res[m]["pairs"]
    m_true = fof["mass"][pairs_m[:, 1]]
    m_pred = res[m]["pred"]["mass"][pairs_m[:, 0]]
    rel_err = (m_pred - m_true) / m_true

    medians, p16s, p84s = [], [], []
    for i in range(len(bins) - 1):
        mask = (m_true >= bins[i]) & (m_true < bins[i+1])
        if mask.sum() > 5:
            medians.append(np.median(rel_err[mask]))
            p16s.append(np.percentile(rel_err[mask], 16))
            p84s.append(np.percentile(rel_err[mask], 84))
        else:
            medians.append(np.nan); p16s.append(np.nan); p84s.append(np.nan)
    medians, p16s, p84s = np.array(medians), np.array(p16s), np.array(p84s)
    scatter = (p84s - p16s) / 2.0

    ax.errorbar(bin_centers, medians, yerr=scatter, fmt=marker, color=color, ecolor="gray",
                capsize=4, elinewidth=1.5, markeredgecolor="black", markersize=6,
                label=f"{m}: mediana ± scatter 1σ")

ax.set_xscale("log")
ax.set_xlabel(r"$M_{\mathrm{true}}\ (\mathrm{FoF})\ [M_\odot/h]$", fontsize=12)
ax.set_ylabel(r"$\frac{M_{\mathrm{pred}} - M_{\mathrm{true}}}{M_{\mathrm{true}}}$", fontsize=12)
ax.set_title("Errore relativo di massa: detection HC vs baseline (fotometria I2 identica)", fontsize=12)
ax.grid(True, which="major", ls=":", color="0.8")
ax.legend(frameon=True, loc="upper right")
plt.show()

```


    
![png](vera_conn_HC_files/vera_conn_HC_17_0.png)
    


### 3.3 Autopsia dei falsi negativi: seen vs not seen

L'analisi controfattuale del NB3 originale separa le responsabilità: per ogni FoF non accoppiato campioniamo la probabilità massima della rete sul disco di raggio $\max(R_{200}, 3\,\mathrm{px})$ nella fetta dell'alone. *Seen* ($p_{\max} \ge t$) significa che la rete lo vedeva e il post-processing l'ha perso (filtri o fusione); *not seen* è il limite fisico della rete, recuperabile solo riaddestrando. Nella baseline il quadro era netto: 855 aloni mancanti su 4701, di cui 829 seen e solo 26 not seen — ceiling di completezza 99.4%, margine algoritmico 17.6 pp, limite fisico 0.55%. Qui ripetiamo la decomposizione per entrambe le detection alla stessa $(0.95, 4)$: il confronto dice se l'alto contrasto riduce il margine algoritmico (meno aloni persi da filtri e linking) o il limite della rete (meno "ciechi"), o entrambi.



```python
# ============================================================
# 3.3 — DECOMPOSIZIONE SEEN / NOT SEEN DEI FoF NON ACCOPPIATI (BASE vs HC)
# ============================================================
def seen_not_seen(res_m, probs_m, t_thr):
    """Per ogni FoF non accoppiato: probabilità massima della rete sul disco di
    raggio max(R200, 3 px) nella fetta dell'alone (come NB3 baseline)."""
    unmatched_idx = np.flatnonzero(~res_m["fm"])
    h_slc = np.floor(fof["pos"][:, 2] / DZ).astype(int) % NS
    h_xpx = fof["pos"][:, 0] / PX_MPC
    h_ypx = fof["pos"][:, 1] / PX_MPC
    h_rpx = np.maximum(fof["r200"] / PX_MPC, MIN_TOL_PX)

    prob_peak = np.zeros(len(unmatched_idx))
    for k, j in enumerate(unmatched_idx):
        s = h_slc[j]
        cx, cy, R = h_xpx[j], h_ypx[j], h_rpx[j]
        Ri = int(np.ceil(R))
        xs = (np.arange(int(cx) - Ri, int(cx) + Ri + 1)) % IMG
        ys = (np.arange(int(cy) - Ri, int(cy) + Ri + 1)) % IMG
        patch = probs_m[s][np.ix_(ys, xs)]
        yy, xx = np.meshgrid(ys - cy, xs - cx, indexing="ij")
        m = yy**2 + xx**2 <= R**2
        prob_peak[k] = patch[m].max() if m.any() else 0.0

    seen = prob_peak >= t_thr
    return {"prob_peak": prob_peak, "seen": seen, "not_seen": ~seen,
            "n_unmatched": len(unmatched_idx),
            "comp_now": float(1.0 - len(unmatched_idx) / len(fof["mass"])),
            "comp_ceiling": float(1.0 - int((~seen).sum()) / len(fof["mass"]))}

DEC_095 = {m: seen_not_seen(res[m], probs_by_mode[m], T_BASE) for m in MODES}

for m in MODES:
    d = DEC_095[m]
    n_unm = max(d["n_unmatched"], 1)
    print(f"[{m:>4}] FoF non accoppiati: {d['n_unmatched']:,} / {len(fof['mass']):,} "
          f"(C = {100*d['comp_now']:.1f}%)")
    print(f"       seen (p >= {T_BASE})    : {int(d['seen'].sum()):>4} "
          f"({100*d['seen'].sum()/n_unm:.1f}%)")
    print(f"       not seen (p < {T_BASE}) : {int(d['not_seen'].sum()):>4} "
          f"({100*d['not_seen'].sum()/n_unm:.1f}%)")
    print(f"       ceiling = {100*d['comp_ceiling']:.1f}% | margine algoritmico = "
          f"{100*(d['comp_ceiling']-d['comp_now']):.1f} pp | limite rete = "
          f"{100*(1-d['comp_ceiling']):.2f} pp")

# Istogrammi affiancati
fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True, sharey=True)
bins_pk = np.linspace(0, 1, 41)
for ax, m in zip(axes, MODES):
    d = DEC_095[m]
    ax.hist(d["prob_peak"][d["seen"]], bins=bins_pk, alpha=0.7, color="tab:green",
            label=f"seen (n={int(d['seen'].sum())})")
    ax.hist(d["prob_peak"][d["not_seen"]], bins=bins_pk, alpha=0.7, color="tab:red",
            label=f"not seen (n={int(d['not_seen'].sum())})")
    ax.axvline(T_BASE, color="k", ls="--", lw=1.5, label=f"t = {T_BASE}")
    ax.set_yscale("log")
    ax.set_xlabel("Probabilità massima U-Net nell'intorno del FoF")
    ax.set_title(f"{m}: perché la completezza non è al 100%?")
    ax.legend(frameon=False)
axes[0].set_ylabel("Numero di FoF non matchati")
plt.show()

```

    [BASE] FoF non accoppiati: 855 / 4,701 (C = 81.8%)
           seen (p >= 0.95)    :  829 (97.0%)
           not seen (p < 0.95) :   26 (3.0%)
           ceiling = 99.4% | margine algoritmico = 17.6 pp | limite rete = 0.55 pp
    [  HC] FoF non accoppiati: 702 / 4,701 (C = 85.1%)
           seen (p >= 0.95)    :  679 (96.7%)
           not seen (p < 0.95) :   23 (3.3%)
           ceiling = 99.5% | margine algoritmico = 14.4 pp | limite rete = 0.49 pp
    


    
![png](vera_conn_HC_files/vera_conn_HC_19_1.png)
    


## 4. Diagnosi dei Falsi Positivi e Transizione delle Soglie

Le categorie di falsi positivi sono le quattro del NB3 originale: *sotto cut* (aloni reali con $M < \mathrm{MASS\_CUT}$, invisibili al target), *doppia* (frammentazione di un alone massivo già trovato), *subhalo* (sottostrutture Subfind reali) e *rumore* (nessun corrispettivo fisico). La purity fisica conta come validi anche i primi e i subhalo; nella baseline valeva $0.9968$ a $(0.95, 4)$, con rumore vero $12/5962 = 0.20\%$ e il 98.5% dei FP apparenti composto da aloni sotto cut. Qui ripetiamo la fotografia statica e la transizione dinamica $0.99 \to 0.95$ **per entrambe le detection**: il confronto diretto mostra se l'alto contrasto paga in recuperi (nuovi FoF trovati) o se costa in frammentazione e rumore.

### 4.1 La fotografia statica: classificazione FP e purity fisica



```python
# ============================================================
# 4.1 — CLASSIFICAZIONE FALSI POSITIVI (STATICA) — HC vs BASE
# ============================================================
for m in MODES:
    r = res[m]
    n_pred = r["n_pred"]
    n_fp = r["n_below"] + r["n_double"] + r["n_sub"] + r["n_noise"]
    print(f"\n[{m}] t={T_BASE}: {n_pred:,} oggetti | {r['n_match']:,} match | {n_fp} FP apparenti")
    if m == "HC":
        print(f"{'categoria':<12} | {'n':>4} | {'% FP':>6} | spiegazione")
        print("-" * 88)
        for name, n, desc in [
            ("sotto cut", r["n_below"],  "aloni reali, invisibili al target per costruzione"),
            ("doppia",    r["n_double"], "frammentazione di aloni massivi"),
            ("subhalo",   r["n_sub"],    "sottostrutture Subfind, non in Group"),
            ("rumore",    r["n_noise"],  "nessun corrispettivo fisico -> veri falsi positivi"),
        ]:
            print(f"{name:<12} | {n:>4} | {100*n/max(n_fp,1):>5.1f}% | {desc}")
    print(f"  P_raw={r['P_raw']:.4f} | P_phys={r['P_phys']:.4f} | "
          f"rumore vero = {r['n_noise']}/{n_pred} = {100*r['n_noise']/n_pred:.2f}%")

```

    
    [BASE] t=0.95: 5,962 oggetti | 3,846 match | 2116 FP apparenti
      P_raw=0.6451 | P_phys=0.9968 | rumore vero = 12/5962 = 0.20%
    
    [HC] t=0.95: 5,747 oggetti | 3,999 match | 1748 FP apparenti
    categoria    |    n |   % FP | spiegazione
    ----------------------------------------------------------------------------------------
    sotto cut    | 1701 |  97.3% | aloni reali, invisibili al target per costruzione
    doppia       |   12 |   0.7% | frammentazione di aloni massivi
    subhalo      |   31 |   1.8% | sottostrutture Subfind, non in Group
    rumore       |    4 |   0.2% | nessun corrispettivo fisico -> veri falsi positivi
      P_raw=0.6958 | P_phys=0.9972 | rumore vero = 4/5747 = 0.07%
    

### 4.2 La transizione dinamica: da $0.99$ a $0.95$

Quando la soglia scende da $0.99$ a $0.95$ il numero di oggetti cresce. Da dove vengono i nuovi? Il matching differenziale tra i due cataloghi predetti (`match_pred_pred`, greedy 1-a-1 con tolleranza su $R_{200}^{\mathrm{est}}$) li classifica in 5 categorie: *sotto cut*, *recupero* (matcha un FoF sopra cut **non trovato** a $0.99$: guadagno reale di completezza), *duplicato* (frammentazione di un FoF già trovato), *subhalo*, *rumore*. Nella baseline l'abbassamento era stato un arricchimento: 2421 nuovi oggetti, di cui 524 recuperi veri contro 7 duplicati e 10 rumori. Se per l'HC i recuperi superano i duplicati anche qui, la soglia 0.95 resta la scelta giusta; se invece la rete HC produce più frammenti duplicati, il compromesso si sposta verso soglie più alte — è esattamente ciò che la griglia della §5 quantifica cella per cella.



```python
# ============================================================
# 4.2 — DIFF TRA CATALOGHI (0.99 -> 0.95) E CLASSIFICAZIONE A 5 CATEGORIE
# ============================================================
def match_pred_pred(A, B, lbox, dz):
    """Matching greedy 1-a-1 tra due cataloghi predetti (tolleranza su R200_est)."""
    nA, nB = len(A["mass"]), len(B["mass"])
    if nA == 0 or nB == 0:
        return np.zeros((0, 2), int), np.zeros(nA, bool), np.zeros(nB, bool)
    rA, rB = A["r200_est"], B["r200_est"]
    rmax = float(max(rA.max(), rB.max(), MIN_TOL_MPC)) * 1.01
    tree = cKDTree(np.column_stack([B["x"], B["y"]]), boxsize=lbox)
    xyA, zA = np.column_stack([A["x"], A["y"]]), A["z"]

    ci, cj, cd = [], [], []
    for i, js in enumerate(tree.query_ball_point(xyA, r=rmax)):
        if not js:
            continue
        js = np.asarray(js, dtype=int)
        dx = (xyA[i, 0] - B["x"][js] + lbox/2) % lbox - lbox/2
        dy = (xyA[i, 1] - B["y"][js] + lbox/2) % lbox - lbox/2
        dxy = np.hypot(dx, dy)
        tol = np.maximum(np.maximum(rA[i], rB[js]), MIN_TOL_MPC)
        ok = dxy <= tol
        if not ok.any():
            continue
        dzp = (zA[i] - B["z"][js] + lbox/2) % lbox - lbox/2
        ok = ok & (np.abs(dzp) <= dz)
        if not ok.any():
            continue
        js_ok = js[ok]
        ci += [i] * len(js_ok); cj += js_ok.tolist(); cd += dxy[ok].tolist()

    if not ci:
        return np.zeros((0, 2), int), np.zeros(nA, bool), np.zeros(nB, bool)
    ci, cj, cd = np.asarray(ci, int), np.asarray(cj, int), np.asarray(cd)
    order = np.lexsort((cj, ci, cd))
    pmA, pmB = np.zeros(nA, bool), np.zeros(nB, bool)
    pairs = []
    for k in order:
        i, j = ci[k], cj[k]
        if not (pmA[i] or pmB[j]):
            pairs.append((i, j)); pmA[i] = pmB[j] = True
    return np.asarray(pairs, int).reshape(-1, 2), pmA, pmB


def diff_transition(mode):
    """Classificazione a 5 categorie dei nuovi oggetti di `mode` a 0.95 vs 0.99."""
    res_099 = run_pipeline(mode, 0.99, 4)
    res_095 = res[mode]                      # già eseguita a (0.95, 4) in §3.1
    _, _, pm_095_in_099 = match_pred_pred(res_099["pred"], res_095["pred"], LBOX, DZ)
    new_obj = np.flatnonzero(~pm_095_in_099)
    cat_new = np.array([], dtype=object)
    if len(new_obj) > 0:
        pred_new = {k: res_095["pred"][k][new_obj] for k in res_095["pred"]}
        pairs_g_new, pm_g_new, _ = match_catalogs(pred_new, fof_full, LBOX, DZ)
        _,           pm_s_new, _ = match_catalogs(pred_new, sub_full, LBOX, DZ)

        mass_g_new = np.full(len(new_obj), np.nan)
        for i, j in pairs_g_new:
            mass_g_new[i] = fof_full["mass"][j]

        groups_matched_at_099 = set(res_099["pairs"][:, 1].tolist())
        group_idx_new = np.full(len(new_obj), -1, dtype=int)
        for i, j in pairs_g_new:
            group_idx_new[i] = j
        already_matched = np.array([(gj >= 0) and (gj in groups_matched_at_099)
                                    for gj in group_idx_new])

        cat_new = np.full(len(new_obj), "", dtype=object)
        cat_new[pm_g_new & (mass_g_new < MASS_CUT)]                     = "sotto_cut"
        cat_new[pm_g_new & (mass_g_new >= MASS_CUT) & ~already_matched] = "recupero"
        cat_new[pm_g_new & (mass_g_new >= MASS_CUT) &  already_matched] = "duplicato"
        cat_new[(~pm_g_new) & pm_s_new]                                 = "subhalo"
        cat_new[(~pm_g_new) & (~pm_s_new)]                              = "rumore"
    return len(new_obj), cat_new


labels5 = ["sotto_cut", "recupero", "duplicato", "subhalo", "rumore"]
names5  = ["Sotto cut\n(reali, M<cut)", "Recupero\n(FoF nuovi!)", "Duplicato\n(frammentazione)",
           "Subhalo", "Rumore\n(FALSI)"]
colors5 = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:gray"]

diff_res = {m: diff_transition(m) for m in MODES}
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
for ax, m in zip(axes, MODES):
    n_new, cat_new = diff_res[m]
    if n_new == 0:
        ax.set_title(f"{m}: nessun nuovo oggetto a 0.95")
        ax.axis("off")
        continue
    counts = [int((cat_new == c).sum()) for c in labels5]
    bars = ax.bar(names5, counts, color=colors5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{c}\n({100*c/n_new:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"n. nuovi oggetti a t=0.95 (n={n_new})")
    ax.set_title(f"{m}: origine dei nuovi oggetti (t=0.95 vs t=0.99)")
    ax.grid(True, axis="y", ls=":", color="0.8")
    ax.set_ylim(0, max(counts) * 1.2 + 1)
plt.show()

for m in MODES:
    n_new, cat_new = diff_res[m]
    counts = {c: int((cat_new == c).sum()) for c in labels5}
    n_phys = sum(counts[c] for c in labels5[:-1])
    print(f"[{m:>4}] nuovi a 0.95: {n_new} | fisici {n_phys} ({100*n_phys/max(n_new,1):.1f}%) | "
          f"recuperi {counts['recupero']} | duplicati {counts['duplicato']} | "
          f"rumore {counts['rumore']}")

```


    
![png](vera_conn_HC_files/vera_conn_HC_23_0.png)
    


    [BASE] nuovi a 0.95: 2421 | fisici 2411 (99.6%) | recuperi 524 | duplicati 7 | rumore 10
    [  HC] nuovi a 0.95: 1844 | fisici 1842 (99.9%) | recuperi 464 | duplicati 5 | rumore 2
    

### 4.3 Ispezione visiva degli oggetti di rumore (con fetta campionata corretta)

L'ispezione visiva del NB3 originale campionava la fetta `floor(z/DZ)`, dove $z$ è la media pesata in massa dell'oggetto 3D. Per gli oggetti collegati a salto ($s \to s+2$) quella media può cadere su una fetta **senza blob**: la diagnostica originale mostrava allora $p_{\max} \approx 0$ per oggetti reali (in quel run, gli oggetti 4322 e 5320: area 7 e 105 px con $p_{\max} = 0.006$ e $0.002$). Qui campioniamo la **fetta del blob a probabilità massima** dell'oggetto (`best_slice_for_pred`, correzione M3): vale sia per la figura sia per la diagnostica numerica della cella successiva. La classificazione *rumore* resta quella di `classify_fps`, che non usa mai la fetta campionata: la correzione riguarda solo come ispezioniamo questi oggetti, non come li etichettiamo.



```python
# ============================================================
# 4.3 — ISPEZIONE VISIVA DEI FP "RUMORE" (HC) — FIX CROSS-SLICE (M3)
# ============================================================
def best_slice_for_pred(probs_map, cx, cy, r_px):
    """Fetta con probabilità massima nell'intorno del pixel (cx, cy).
    Fix vs NB3 originale: per oggetti collegati a salto (s -> s+2) la media pesata in z
    può cadere su una fetta senza blob, e campionare floor(z/DZ) mostrava p_max ~ 0
    per oggetti reali. Qui si campiona la fetta del blob a probabilità massima."""
    Ri = int(np.ceil(r_px))
    xs = (np.arange(int(cx) - Ri, int(cx) + Ri + 1)) % IMG
    ys = (np.arange(int(cy) - Ri, int(cy) + Ri + 1)) % IMG
    yy, xx = np.meshgrid(ys - cy, xs - cx, indexing="ij")
    disc = yy**2 + xx**2 <= r_px**2
    peaks = np.array([probs_map[s][np.ix_(ys, xs)][disc].max() if disc.any() else 0.0
                      for s in range(NS)])
    return int(peaks.argmax()), float(peaks.max())

# Falsi positivi HC e loro classificazione (identica a classify_fps)
fp_idx_hc = np.flatnonzero(~res["HC"]["pm"])
pred_fp = {k: res["HC"]["pred"][k][fp_idx_hc] for k in res["HC"]["pred"]}
pairs_g, pm_g, _ = match_catalogs(pred_fp, fof_full, LBOX, DZ)
_,       pm_s, _ = match_catalogs(pred_fp, sub_full, LBOX, DZ)
mass_g = np.full(len(fp_idx_hc), np.nan)
for i, j in pairs_g:
    mass_g[i] = fof_full["mass"][j]
cat_fp = np.full(len(fp_idx_hc), "rumore", dtype=object)
cat_fp[pm_g & (mass_g < MASS_CUT)]  = "sotto cut"
cat_fp[pm_g & (mass_g >= MASS_CUT)] = "doppia"
cat_fp[(~pm_g) & pm_s]              = "subhalo"
noise_indices = fp_idx_hc[cat_fp == "rumore"]
print(f"[HC] trovati {len(noise_indices)} oggetti di rumore a t={T_BASE}.")

# Ispezione visiva: fetta del blob a probabilità massima (fix cross-slice)
n_show = min(12, len(noise_indices))
fig, axes = plt.subplots(6, 4, figsize=(16, 22), constrained_layout=True)
axes = axes.ravel()

for k, idx in enumerate(noise_indices[:n_show]):
    x, y = res["HC"]["pred"]["x"][idx], res["HC"]["pred"]["y"][idx]
    r_px = max(MIN_TOL_PX, np.sqrt(res["HC"]["pred"]["area_tot"][idx] / np.pi))
    s, p_max_local = best_slice_for_pred(probs_by_mode["HC"], x / PX_MPC, y / PX_MPC, r_px)
    cx, cy = int(x / PX_MPC), int(y / PX_MPC)
    size = 64
    x0, x1 = max(0, cx-size), min(IMG, cx+size)
    y0, y1 = max(0, cy-size), min(IMG, cy+size)

    prob_patch = probs_by_mode["HC"][s][y0:y1, x0:x1]
    rho_patch  = raw[s][y0:y1, x0:x1]
    rho_vmax = np.percentile(raw[s][raw[s] > 0], 99)

    ap, ar = axes[2*k], axes[2*k+1]
    ap.imshow(prob_patch, cmap="viridis", origin="lower", extent=[x0, x1, y0, y1], vmin=0, vmax=1)
    ap.contour(rho_patch, levels=[np.percentile(raw[s], 90)], colors="red", linewidths=0.5,
               extent=[x0, x1, y0, y1])
    ap.contour(prob_patch, levels=[T_BASE], colors="blue", linewidths=1.2, extent=[x0, x1, y0, y1])
    ap.plot(cx, cy, "o", color="cyan", ms=8, mew=2, mfc="none")
    ap.set_title(f"Obj {idx} — Prob (fetta {s}, p_max={p_max_local:.3f})")

    ar.imshow(rho_patch, cmap="magma", origin="lower", extent=[x0, x1, y0, y1], vmin=0, vmax=rho_vmax)
    ar.plot(cx, cy, "o", color="cyan", ms=8, mew=2, mfc="none")
    ar.set_title(f"Obj {idx} — Simulazione (I2)")
    for a in (ap, ar):
        a.set_xticks([]); a.set_yticks([])

for a in axes[2*n_show:]:
    a.axis("off")
plt.suptitle(f"Ispezione visiva dei falsi positivi 'rumore' (HC, t={T_BASE}) — "
             "fetta del blob a p_max", fontsize=15)
plt.show()

```

    [HC] trovati 4 oggetti di rumore a t=0.95.
    


    
![png](vera_conn_HC_files/vera_conn_HC_25_1.png)
    



```python
# Diagnostica dettagliata sugli oggetti di rumore (fetta corretta, tolleranza 2x)
noise_pred = {k: res["HC"]["pred"][k][noise_indices] for k in res["HC"]["pred"]}
_, pm_g_wide, _ = match_catalogs(noise_pred, fof_full, LBOX, 2*DZ, tol_scale=2.0)
_, pm_s_wide, _ = match_catalogs(noise_pred, sub_full, LBOX, 2*DZ, tol_scale=2.0)

for k, idx in enumerate(noise_indices):
    x, y = noise_pred["x"][k], noise_pred["y"][k]
    r_px = max(MIN_TOL_PX, np.sqrt(noise_pred["area_tot"][k] / np.pi))
    s, p_max_local = best_slice_for_pred(probs_by_mode["HC"], x / PX_MPC, y / PX_MPC, r_px)
    cx, cy = int(x / PX_MPC) % IMG, int(y / PX_MPC) % IMG
    rho_ratio = raw[s, cy, cx] / np.median(raw[s][raw[s] > 0])
    status = "vicino a Group/Subhalo (2x tol)" if (pm_g_wide[k] or pm_s_wide[k]) \
             else "isolato (da check 3D)"
    print(f"Obj {idx:>4} | area={noise_pred['area_tot'][k]:>5.0f} px | fetta {s:>2} | "
          f"p_max={p_max_local:.3f} | rho/med={rho_ratio:>5.1f}x | {status}")

```

    Obj 2006 | area=    9 px | fetta 10 | p_max=0.972 | rho/med=1309.3x | vicino a Group/Subhalo (2x tol)
    Obj 4163 | area=   10 px | fetta 18 | p_max=0.997 | rho/med=108.6x | vicino a Group/Subhalo (2x tol)
    Obj 4627 | area=  174 px | fetta 19 | p_max=1.000 | rho/med= 84.8x | vicino a Group/Subhalo (2x tol)
    Obj 5158 | area=   69 px | fetta 21 | p_max=1.000 | rho/med=128.0x | vicino a Group/Subhalo (2x tol)
    

#### 4.4 Conclusione

La lettura è la stessa del NB3 originale, ora applicata alla detection HC: la purity fisica misura se la rete segmenta materia reale; i "falsi positivi" sono quasi interamente aloni sotto cut e subhalo — un catalogo arricchito, non inquinato — e il rumore vero è l'unico da guardare, con l'ispezione visiva (ora sulla fetta giusta) che dice se i residui sono fluttuazioni SPH o artefatti. Il punto nuovo è il confronto: per l'alto contrasto contano il **rapporto** recuperi/duplicati nella transizione $0.99 \to 0.95$ e la quota di rumore rispetto alla baseline. Questi numeri, insieme a $C$ e al fit di massa, arrivano dalla griglia della sezione successiva, che li produce per tutte le 12 configurazioni su entrambe le detection.


## 5. Esplorazione della Griglia $4\times3$ head-to-head (baseline vs HC)

Ora la scelta operativa passa da un confronto sistematico: la stessa griglia (4 soglie $\times$ 3 aree) gira su entrambi i set di probabilità, con lo stesso codice, la stessa fotometria I2 e la stessa verità FoF. Per ogni cella misuriamo le metriche del NB3 originale **più il fit di massa** ($a$, $\sigma_{\mathrm{MAD}}$), che nel notebook originale era riservato alla sola configurazione di riferimento: così anche la qualità della massa entra nella scelta operativa con dati, non con claim non verificate (correzione M5).

Due parametri, due effetti opposti sulla completezza — e $C$ **non è monotona** in $t$ (correzione M2 rispetto al testo originale): a soglie basse i blob crescono e si fondono, e l'accoppiamento 1-a-1 perde gli aloni inglobati in oggetti multipli (merging); a soglie alte i blob frammentano e gli aloni piccoli scendono sotto il filtro d'area. Nella baseline il massimo era a $t = 0.95$ ($C = 0.818$), con calo su entrambi i lati ($0.764$ a $0.85$, $0.747$ a $0.99$). La griglia HC mostra dove si trova il massimo dell'alto contrasto e quanto è alto rispetto alla baseline.

### 5.1 Esecuzione della doppia griglia (fit di massa in ogni cella)



```python
# ============================================================
# 5.1 — ESECUZIONE DOPPIA GRIGLIA 4x3 (BASE vs HC, fit di massa in ogni cella)
# ============================================================
grid_rows = []
for mode in MODES:
    for t_val in SOGLIE_CANDIDATE:
        for area in AREE_CANDIDATE:
            r = run_pipeline(mode, t_val, area)
            if r is None:
                continue
            s_fit = mass_stats(r["pairs"], r["pred"], fof)
            grid_rows.append({
                "mode": mode, "t": r["t"], "area": r["area"],
                "n_pred": r["n_pred"], "n_match": r["n_match"],
                "C": r["C"], "P_raw": r["P_raw"], "P_phys": r["P_phys"],
                "a": s_fit["a"], "sigma_MAD": s_fit["sigma_mad"],
                "n_below": r["n_below"], "n_double": r["n_double"],
                "n_sub": r["n_sub"], "n_noise": r["n_noise"],
            })

df_grid = pd.DataFrame(grid_rows)
print(f"Griglia completata: {len(df_grid)} celle "
      f"({len(SOGLIE_CANDIDATE)} soglie × {len(AREE_CANDIDATE)} aree × {len(MODES)} detection)")
print("\nHC (alto contrasto):")
print(df_grid[df_grid["mode"] == "HC"].drop(columns="mode")
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nBASE (baseline, riferimento del run originale):")
print(df_grid[df_grid["mode"] == "BASE"].drop(columns="mode")
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

```

    Griglia completata: 24 celle (4 soglie × 3 aree × 2 detection)
    
    HC (alto contrasto):
        t  area  n_pred  n_match     C  P_raw  P_phys     a  sigma_MAD  n_below  n_double  n_sub  n_noise
    0.850     4    6523     3818 0.812  0.585   0.997 0.863      0.126     2678         9      9        9
    0.850     6    6178     3812 0.811  0.617   0.997 0.864      0.126     2340         9      9        8
    0.850    12    2902     2740 0.583  0.944   0.996 0.883      0.114      145         3      6        8
    0.900     4    6252     3884 0.826  0.621   0.997 0.862      0.128     2339        10     12        7
    0.900     6    5828     3858 0.821  0.662   0.997 0.867      0.128     1941        10     12        7
    0.900    12    2629     2517 0.535  0.957   0.996 0.891      0.112       97         4      5        6
    0.950     4    5747     3999 0.851  0.696   0.997 0.866      0.129     1701        12     31        4
    0.950     6    5303     3925 0.835  0.740   0.997 0.876      0.129     1332        11     31        4
    0.950    12    2246     2191 0.466  0.976   0.996 0.909      0.109       37         6     10        2
    0.990     4    4302     3733 0.794  0.868   0.992 0.862      0.136      400        34    133        2
    0.990     6    3744     3421 0.728  0.914   0.993 0.897      0.133      191        26    104        2
    0.990    12    1498     1442 0.307  0.963   0.993 0.923      0.097       14        11     31        0
    
    BASE (baseline, riferimento del run originale):
        t  area  n_pred  n_match     C  P_raw  P_phys     a  sigma_MAD  n_below  n_double  n_sub  n_noise
    0.850     4    7482     3593 0.764  0.480   0.986 0.862      0.134     3771        10     15       93
    0.850     6    7326     3592 0.764  0.490   0.989 0.863      0.134     3639        10     15       70
    0.850    12    6674     3578 0.761  0.536   0.993 0.865      0.134     3038         5     14       39
    0.900     4    7015     3690 0.785  0.526   0.992 0.866      0.135     3259         9     11       46
    0.900     6    6806     3688 0.785  0.542   0.994 0.867      0.135     3063         9     11       35
    0.900    12    6069     3663 0.779  0.604   0.997 0.871      0.133     2374         4     11       17
    0.950     4    5962     3846 0.818  0.645   0.997 0.884      0.133     2084         7     13       12
    0.950     6    5729     3831 0.815  0.669   0.997 0.890      0.131     1870         6     13        9
    0.950    12    4945     3735 0.795  0.755   0.999 0.904      0.129     1191         3     12        4
    0.990     4    3854     3510 0.747  0.911   0.994 0.864      0.149      255        21     66        2
    0.990     6    3654     3387 0.720  0.927   0.994 0.904      0.139      183        19     63        2
    0.990    12    3051     2904 0.618  0.952   0.995 0.949      0.129       79        14     53        1
    

### 5.2 Tabelle affiancate e $\Delta C$ per cella

La pivot affianca le due detection cella per cella per ogni metrica; la tabella $\Delta C = \mathrm{HC} - \mathrm{BASE}$ è la misura diretta del guadagno (o perdita) di completezza dell'alto contrasto a configurazione costante. Un $\Delta C > 0$ a parità di $P_{\mathrm{phys}}$ e rumore significa che l'alto contrasto trova più aloni reali con la stessa pulizia; un $\Delta C < 0$ accompagnato da più rumore segnala il prezzo da pagare. Le colonne `a` e `sigma_MAD` completano il quadro: se la detection HC raccogliesse sistematicamente più materia diffusa, si vedrebbe come pendenza del fit diversa **senza** che la fotometria sia cambiata.



```python
# ============================================================
# 5.2 — PIVOT CONFRONTATA E DELTA PER CELLA
# ============================================================
cols_show = ["C", "P_phys", "n_noise", "a", "sigma_MAD", "n_pred"]
piv = df_grid.pivot(index=["t", "area"], columns="mode", values=cols_show)

for col in cols_show:
    print(f"\n--- {col} ---")
    print(piv[col].to_string(float_format=lambda v: f"{v:.3f}"))

dC = df_grid.pivot(index=["t", "area"], columns="mode", values="C")
dC["delta_C"] = dC["HC"] - dC["BASE"]
print("\n--- ΔC = HC - BASE (per cella) ---")
print(dC.to_string(float_format=lambda v: f"{v:+.3f}"))

```

    
    --- C ---
    mode       BASE    HC
    t    area            
    0.85 4    0.764 0.812
         6    0.764 0.811
         12   0.761 0.583
    0.90 4    0.785 0.826
         6    0.785 0.821
         12   0.779 0.535
    0.95 4    0.818 0.851
         6    0.815 0.835
         12   0.795 0.466
    0.99 4    0.747 0.794
         6    0.720 0.728
         12   0.618 0.307
    
    --- P_phys ---
    mode       BASE    HC
    t    area            
    0.85 4    0.986 0.997
         6    0.989 0.997
         12   0.993 0.996
    0.90 4    0.992 0.997
         6    0.994 0.997
         12   0.997 0.996
    0.95 4    0.997 0.997
         6    0.997 0.997
         12   0.999 0.996
    0.99 4    0.994 0.992
         6    0.994 0.993
         12   0.995 0.993
    
    --- n_noise ---
    mode        BASE    HC
    t    area             
    0.85 4    93.000 9.000
         6    70.000 8.000
         12   39.000 8.000
    0.90 4    46.000 7.000
         6    35.000 7.000
         12   17.000 6.000
    0.95 4    12.000 4.000
         6     9.000 4.000
         12    4.000 2.000
    0.99 4     2.000 2.000
         6     2.000 2.000
         12    1.000 0.000
    
    --- a ---
    mode       BASE    HC
    t    area            
    0.85 4    0.862 0.863
         6    0.863 0.864
         12   0.865 0.883
    0.90 4    0.866 0.862
         6    0.867 0.867
         12   0.871 0.891
    0.95 4    0.884 0.866
         6    0.890 0.876
         12   0.904 0.909
    0.99 4    0.864 0.862
         6    0.904 0.897
         12   0.949 0.923
    
    --- sigma_MAD ---
    mode       BASE    HC
    t    area            
    0.85 4    0.134 0.126
         6    0.134 0.126
         12   0.134 0.114
    0.90 4    0.135 0.128
         6    0.135 0.128
         12   0.133 0.112
    0.95 4    0.133 0.129
         6    0.131 0.129
         12   0.129 0.109
    0.99 4    0.149 0.136
         6    0.139 0.133
         12   0.129 0.097
    
    --- n_pred ---
    mode          BASE       HC
    t    area                  
    0.85 4    7482.000 6523.000
         6    7326.000 6178.000
         12   6674.000 2902.000
    0.90 4    7015.000 6252.000
         6    6806.000 5828.000
         12   6069.000 2629.000
    0.95 4    5962.000 5747.000
         6    5729.000 5303.000
         12   4945.000 2246.000
    0.99 4    3854.000 4302.000
         6    3654.000 3744.000
         12   3051.000 1498.000
    
    --- ΔC = HC - BASE (per cella) ---
    mode        BASE     HC  delta_C
    t    area                       
    0.85 4    +0.764 +0.812   +0.048
         6    +0.764 +0.811   +0.047
         12   +0.761 +0.583   -0.178
    0.90 4    +0.785 +0.826   +0.041
         6    +0.785 +0.821   +0.036
         12   +0.779 +0.535   -0.244
    0.95 4    +0.818 +0.851   +0.033
         6    +0.815 +0.835   +0.020
         12   +0.795 +0.466   -0.328
    0.99 4    +0.747 +0.794   +0.047
         6    +0.720 +0.728   +0.007
         12   +0.618 +0.307   -0.311
    

### 5.3 Grafici trade-off sovrapposti

I tre pannelli del NB3 originale ($C$, $P_{\mathrm{phys}}$, $n_{\mathrm{pred}}$ in funzione di $t$) con le curve delle due detection sovrapposte: HC a linea continua, baseline tratteggiata, una coppia per ogni area. Il quarto pannello mostra $\Delta C$ per cella d'area: è la firma dell'alto contrasto, positiva dove la detection HC recupera aloni che la baseline perdeva, negativa dove merging o frammentazione la penalizzano. Attenzione alla scala del pannello $P_{\mathrm{phys}}$: come nella baseline, valori sotto $0.99$ indicano l'ingresso di detezioni meno pulite a soglie basse.



```python
# ============================================================
# 5.3 — GRAFICI TRADE-OFF (C, P_phys, n_pred, ΔC) — HC vs BASE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
area_colors = {4: "tab:blue", 6: "tab:green", 12: "tab:orange"}

for area in AREE_CANDIDATE:
    sub = df_grid[df_grid["area"] == area].sort_values("t")
    b = sub[sub["mode"] == "BASE"]; h = sub[sub["mode"] == "HC"]
    axes[0, 0].plot(h["t"], h["C"], "o-", color=area_colors[area], label=f"HC area={area}")
    axes[0, 0].plot(b["t"], b["C"], "o--", color=area_colors[area], alpha=0.55,
                    label=f"BASE area={area}")
    axes[0, 1].plot(h["t"], h["P_phys"], "s-", color=area_colors[area])
    axes[0, 1].plot(b["t"], b["P_phys"], "s--", color=area_colors[area], alpha=0.55)
    axes[1, 0].plot(h["t"], h["n_pred"], "^-", color=area_colors[area])
    axes[1, 0].plot(b["t"], b["n_pred"], "^--", color=area_colors[area], alpha=0.55)
    axes[1, 1].plot(h["t"], h["C"].values - b["C"].values, "d-", color=area_colors[area],
                    label=f"area={area}")

axes[0, 0].set(title="Completezza C", xlabel="soglia t", ylabel="C")
axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
axes[0, 1].set(title="Purity fisica", xlabel="soglia t", ylabel="P_phys")
axes[0, 1].axhline(1.0, color="0.6", ls=":", lw=1)
axes[1, 0].set(title="Numero di oggetti predetti", xlabel="soglia t", ylabel="n_pred")
axes[1, 0].set_yscale("log")
axes[1, 1].axhline(0, color="0.6", ls=":", lw=1)
axes[1, 1].set(title=r"$\Delta C$ = HC $-$ BASE", xlabel="soglia t", ylabel=r"$\Delta C$")
axes[1, 1].legend(frameon=False)

for ax in axes.ravel():
    ax.grid(True, ls=":", color="0.8")
plt.suptitle("Trade-off soglia/area: HC (linea continua) vs baseline (tratteggiata)", fontsize=13)
plt.show()

```


    
![png](vera_conn_HC_files/vera_conn_HC_33_0.png)
    


### 5.4 Scelta della configurazione operativa HC

La scelta segue una regola esplicita e deterministica: tra le celle HC con purity fisica $\ge 0.99$ e rumore vero $\le 0.5\%$ degli oggetti predetti si prende quella con completezza massima; se nessuna cella rispetta i vincoli, si ripete la ricerca su tutte le celle HC senza vincoli (con avviso). La regola rende la decisione riproducibile e legata alle metriche. Accanto alla configurazione scelta viene stampata la variante conservativa $(0.99, 12)$ — la candidata di precisione del NB3 originale — per la comparazione. Il salvataggio della scelta in `chosen_threshold.json` avviene **solo** in §8, dopo il riepilogo.



```python
# ============================================================
# 5.4 — CONFIGURAZIONE OPERATIVA HC (regola deterministica)
# ============================================================
ok_cells = df_grid[(df_grid["mode"] == "HC")
                   & (df_grid["P_phys"] >= 0.99)
                   & (df_grid["n_noise"] / df_grid["n_pred"] <= 0.005)]
if len(ok_cells) == 0:      # fallback: nessuna cella rispetta i vincoli
    ok_cells = df_grid[df_grid["mode"] == "HC"]
    print("[AVVISO] nessuna cella HC rispetta i vincoli (P_phys>=0.99, rumore<=0.5%): "
          "scelta sulla C massima senza vincoli.")
row_final = ok_cells.loc[ok_cells["C"].idxmax()]
T_FINAL_HC, AREA_FINAL_HC = float(row_final["t"]), int(row_final["area"])

print(f"Configurazione operativa HC scelta: t = {T_FINAL_HC}, area3d = {AREA_FINAL_HC}")
print(row_final[["n_pred", "n_match", "C", "P_raw", "P_phys", "a", "sigma_MAD",
                 "n_below", "n_double", "n_sub", "n_noise"]].to_string())

cons = df_grid[(df_grid["mode"] == "HC") & (df_grid["t"] == 0.99) & (df_grid["area"] == 12)]
if len(cons):
    row_cons = cons.iloc[0]
    print(f"\nVariante conservativa HC (t=0.99, area=12): C={row_cons['C']:.3f}, "
          f"P_phys={row_cons['P_phys']:.4f}, a={row_cons['a']:.3f}, "
          f"sigma_MAD={row_cons['sigma_MAD']:.3f}, rumore={int(row_cons['n_noise'])}")
else:
    print("\n(Variante conservativa (0.99, 12) non presente nella griglia corrente.)")

print("\n(Il chosen_threshold.json della baseline in unet/ NON viene toccato: "
      "la scelta HC sarà scritta solo in unet_HC/ alla §8.)")

```

    Configurazione operativa HC scelta: t = 0.95, area3d = 4
    n_pred           5747
    n_match          3999
    C             0.85067
    P_raw        0.695841
    P_phys       0.997216
    a            0.866293
    sigma_MAD    0.129087
    n_below          1701
    n_double           12
    n_sub              31
    n_noise             4
    
    Variante conservativa HC (t=0.99, area=12): C=0.307, P_phys=0.9927, a=0.923, sigma_MAD=0.097, rumore=0
    
    (Il chosen_threshold.json della baseline in unet/ NON viene toccato: la scelta HC sarà scritta solo in unet_HC/ alla §8.)
    

### 5.5 Confronto visivo delle soglie (HC)

Come nel NB3 originale, una figura unica $3\times4$ su una fetta di esempio: riga 0 il contesto (densità I2 lineare e logaritmica con gli aloni FoF), riga 1 la mappa di probabilità **HC** con il contorno alla soglia, riga 2 la densità con i blob predetti (ciano) e la verità FoF (arancione). La fetta è la 12, la stessa usata dai visual check di NB1-HC e NB2-HC (`RAW_SAMPLE_SLICE`), così le tre ispezioni visive della pipeline guardano sempre lo stesso campo. L'effetto atteso è il ritmo del NB3 baseline: a soglie basse compaiono più isole (anche piccole e fondute), a soglie alte solo i nuclei più densi.



```python
# ============================================================
# 5.5 — CONFRONTO VISIVO DELLE 4 SOGLIE (HC, fetta 12)
# ============================================================
from matplotlib.collections import EllipseCollection

SLICE_DEMO = 12    # = RAW_SAMPLE_SLICE di NB1-HC (stessa fetta dei visual check precedenti)
rho_s, p_s = raw[SLICE_DEMO], probs_by_mode["HC"][SLICE_DEMO]
extent = [0, IMG, 0, IMG]

vmax = np.percentile(rho_s[rho_s > 0], 99)
rho_log = np.log10(rho_s, where=rho_s > 0, out=np.full_like(rho_s, -np.inf))
vmin_log, vmax_log = np.percentile(rho_log[np.isfinite(rho_log)], [1, 99.5])

h_slice = np.floor(fof["pos"][:, 2] / DZ).astype(int) % NS
fmask = h_slice == SLICE_DEMO
hxd, hyd = fof["pos"][fmask, :2].T / PX_MPC
h_r = np.maximum(fof["r200"][fmask] / PX_MPC, MIN_TOL_PX)

def rings(ax, x, y, r, color, alpha=0.8):
    if len(x):
        ax.add_collection(EllipseCollection(
            2*r, 2*r, 0, units="xy",
            offsets=np.c_[x, y], transOffset=ax.transData,
            facecolors="none", edgecolors=color,
            linewidths=1.2, alpha=alpha
        ))

fig = plt.figure(figsize=(18, 13), constrained_layout=True)
gs = fig.add_gridspec(3, 4)

ax00 = fig.add_subplot(gs[0, 0:2])
ax00.imshow(rho_s, origin="lower", cmap="magma", vmin=0, vmax=vmax, extent=extent)
ax00.set_title(f"Simulazione — Densità I2 originale (Slice {SLICE_DEMO})")
ax00.set(xticks=[], yticks=[])

ax01 = fig.add_subplot(gs[0, 2:4])
ax01.imshow(rho_log, origin="lower", cmap="magma", vmin=vmin_log, vmax=vmax_log, extent=extent)
ax01.scatter(hxd, hyd, s=4, c="white", marker="x", alpha=0.4)
ax01.set_title(r"Densità reale ($\log_{10}\rho$) — riferimento")
ax01.set(xticks=[], yticks=[])

for j, t in enumerate(SOGLIE_CANDIDATE):
    B = get_blobs("HC", t)
    m = B["slice"] == SLICE_DEMO
    bx, by = B["x"][m] / PX_MPC, B["y"][m] / PX_MPC
    br = B["r_eq"][m] / PX_MPC

    ax_top = fig.add_subplot(gs[1, j])
    ax_top.imshow(p_s, origin="lower", cmap="viridis", vmin=0, vmax=1, extent=extent)
    ax_top.contour(p_s, levels=[t], colors="red", linewidths=0.8, extent=extent)
    ax_top.set_title(f"Probabilità HC — t={t}")
    ax_top.set(xticks=[], yticks=[])

    ax_bot = fig.add_subplot(gs[2, j])
    ax_bot.imshow(rho_s, origin="lower", cmap="magma", vmin=0, vmax=vmax, extent=extent)
    rings(ax_bot, bx, by, br, "cyan")
    ax_bot.scatter(bx, by, s=6, c="cyan", label=f"Predetti ({m.sum()})")
    rings(ax_bot, hxd, hyd, h_r, "orange", alpha=0.6)
    ax_bot.scatter(hxd, hyd, s=8, c="orange", marker="+", label=f"FoF verità ({fmask.sum()})")
    ax_bot.set_title(f"Densità + Blob HC (n={m.sum()}) — t={t}")
    ax_bot.legend(loc="upper right", fontsize=8, facecolor="white", framealpha=0.9)
    ax_bot.set(xticks=[], yticks=[])

fig.suptitle(f"Impatto della soglia sulla detection HC (Slice {SLICE_DEMO})",
             fontsize=16, fontweight="bold")
plt.show()

```


    
![png](vera_conn_HC_files/vera_conn_HC_37_0.png)
    


## 6. Funzione di Massa (HMF) e Completezza per Massa

La HMF è la metrica cosmologica di sintesi: confronta la distribuzione di massa dei cataloghi predetti con quella della verità FoF. Il pannello di sinistra sovrappone quattro curve: la verità (grigio), le predizioni della baseline (blu) e dell'HC (rosso) — con l'eccesso atteso a basse masse per aloni sotto cut e subhalo — e i matchati HC (tratteggiata scura), che deve seguire fedelmente la verità. Il pannello di destra mostra la completezza per bin di massa delle due detection: il confronto dice a quale scala di massa l'alto contrasto guadagna (tipicamente le masse intermedie, dove il contrasto aiuta a separare i piccoli aloni dal fondo proiettato) e dove invece i due cataloghi coincidono.

### 6.1 HMF: FoF, predetti BASE/HC, matchati



```python
# ============================================================
# 6.1 — HMF: CONFRONTO FoF / PRED BASE / PRED HC / MATCHATI
# ============================================================
LOG_M_MIN, LOG_M_MAX = 9.5, 13.5
N_BINS = 16
edges   = np.logspace(LOG_M_MIN, LOG_M_MAX, N_BINS + 1)
centers = np.sqrt(edges[:-1] * edges[1:])

mf_fof, _ = np.histogram(fof["mass"], bins=edges)
mf_pred = {m: np.histogram(res[m]["pred"]["mass"], bins=edges)[0] for m in MODES}
mf_det  = {m: np.histogram(fof["mass"][res[m]["pairs"][:, 1]], bins=edges)[0] for m in MODES}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

ax1.plot(centers, mf_fof, "o-", color="0.35", lw=1.8, ms=5, label="FoF (verità)")
ax1.plot(centers, mf_pred["BASE"], "s--", color="tab:blue", lw=1.6, ms=5, label="Predetti BASE")
ax1.plot(centers, mf_pred["HC"], "s-", color="crimson", lw=1.8, ms=5, label="Predetti HC")
ax1.plot(centers, mf_det["HC"], "^:", color="0.2", lw=1.4, ms=4, label="Matchati (HC, FoF mass)")
ax1.axvline(MASS_CUT, color="tab:red", ls=":", lw=1.2, label=f"MASS_CUT = {MASS_CUT:.2e}")
ax1.set(xscale="log", yscale="log", xlabel=r"$M$ [$M_\odot/h$]", ylabel="N oggetti per bin",
        title="Funzione di massa degli aloni (HMF)")
ax1.legend(loc="lower left", frameon=False, fontsize=9)
ax1.grid(True, which="major", ls=":", color="0.8")

valid = mf_fof > 0
ax2.plot(centers[valid], 100 * mf_det["BASE"][valid] / mf_fof[valid], "o--", color="tab:blue",
         lw=1.5, ms=5, label="completezza BASE")
ax2.plot(centers[valid], 100 * mf_det["HC"][valid] / mf_fof[valid], "o-", color="crimson",
         lw=1.8, ms=5, label="completezza HC")
ax2.axvline(MASS_CUT, color="tab:red", ls=":", lw=1.2, label="MASS_CUT")
ax2.set(xscale="log", ylim=(0, 105), xlabel=r"$M$ [$M_\odot/h$]", ylabel="completezza [%]",
        title="Completezza per bin di massa")
ax2.legend(loc="lower left", frameon=False, fontsize=9)
ax2.grid(True, which="major", ls=":", color="0.8")

plt.show()

```


    
![png](vera_conn_HC_files/vera_conn_HC_39_0.png)
    


### 6.2 Completezza e Purity per Aloni Ben Risolti

Come nel NB3 originale, per soglie di massa crescenti $M_{\mathrm{thr}}$ filtriamo entrambi i cataloghi, rifacciamo il matching e misuriamo $C(M_{\mathrm{thr}}) = N_{\mathrm{match}}/N_{\mathrm{FoF}}(>M_{\mathrm{thr}})$ e $P(M_{\mathrm{thr}}) = N_{\mathrm{match}}/N_{\mathrm{pred}}(>M_{\mathrm{thr}})$. Per gli aloni massicci ($> 10^{11}\,M_\odot/h$) la baseline era vicina al 100%: qui le curve affiancate mostrano se l'alto contrasto sposta verso il basso la massa alla quale la pipeline raggiunge quelle performance. È un guadagno pratico concreto: sposta il limite di risoluzione del catalogo, cioè la massa sotto la quale le misure smettono di essere affidabili.



```python
# ============================================================
# 6.2 — COMPLETENESS E PURITY IN FUNZIONE DELLA MASSA (BASE vs HC)
# ============================================================
mass_thresholds = np.logspace(np.log10(MASS_CUT), np.log10(10**12.5), 10)
cp_vals = {}
for m in MODES:
    C_vals, P_vals = [], []
    for m_thr in mass_thresholds:
        fof_mask = fof["mass"] >= m_thr
        pred_mask = res[m]["pred"]["mass"] >= m_thr
        fof_thr = {k: v[fof_mask] for k, v in fof.items()}
        pred_thr = {k: v[pred_mask] for k, v in res[m]["pred"].items()}
        if len(fof_thr["mass"]) == 0 or len(pred_thr["mass"]) == 0:
            C_vals.append(np.nan); P_vals.append(np.nan)
            continue
        pairs_thr, pm_thr, fm_thr = match_catalogs(pred_thr, fof_thr, LBOX, DZ)
        C_vals.append(fm_thr.sum() / len(fof_thr["mass"]))
        P_vals.append(pm_thr.sum() / len(pred_thr["mass"]))
    cp_vals[m] = (np.array(C_vals), np.array(P_vals))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for m, color, ls in [("HC", "crimson", "-"), ("BASE", "tab:blue", "--")]:
    C_vals, P_vals = cp_vals[m]
    ax1.plot(mass_thresholds, C_vals * 100, "o" + ls, color=color, lw=2,
             label=f"Completeness {m}")
    ax2.plot(mass_thresholds, P_vals * 100, "s" + ls, color=color, lw=2, label=f"Purity {m}")
ax1.set_xscale("log"); ax2.set_xscale("log")
ax1.set(xlabel=r"Soglia di massa $M_{\mathrm{thr}}$ [$M_\odot/h$]", ylabel="Completeness [%]",
        title="Completeness per aloni sopra una data massa", ylim=(0, 105))
ax2.set(xlabel=r"Soglia di massa $M_{\mathrm{thr}}$ [$M_\odot/h$]", ylabel="Purity grezza [%]",
        title="Purity grezza per aloni sopra una data massa", ylim=(0, 105))
for ax in (ax1, ax2):
    ax.grid(True, which="major", ls=":", color="0.8")
    ax.legend(frameon=True)
plt.show()

```


    
![png](vera_conn_HC_files/vera_conn_HC_41_0.png)
    


## 7. Diagnostica specifica HC

Due test che nel NB3 baseline non c'erano (o non avevano questo senso) e qui sono diretti. Il primo misura la **contaminazione delle zone vuote**: le probabilità HC nei pixel dove $\rho^{I2}=0$ devono essere $\sim 0$, perché la rete ha visto patch con `EMPTY_FILL` costante e target nullo. Poiché la massa di quei pixel è nulla per costruzione, il rischio non è sul fit di massa ma sul rumore: una detezione interamente in zona vuota sopravvive al filtro d'area (che conta pixel, non massa) e finisce tra i falsi positivi "rumore" — la §4.3 ne mostrerebbe l'ispezione visiva. Il secondo test ripete il check del background proiettato del NB3 originale con la detection HC: sottrae il fondo I2 stimato nei pixel quieti ($p < 1\%$) e rifà il fit di massa — se $a$ non cambia, il fondo proiettato non è la causa di bias nemmeno per il catalogo HC.

### 7.1 Contaminazione delle zone vuote (`EMPTY_FILL`)



```python
# ============================================================
# 7.1 — CONTAMINAZIONE DELLE ZONE VUOTE (EMPTY_FILL)
# ============================================================
empty_mask = (raw == 0.0)
n_empty = int(empty_mask.sum())
print(f"pixel vuoti: {n_empty:,} ({100*empty_mask.mean():.2f}% della mappa)")

rows_empty = []
for mode in MODES:
    probs_m = probs_by_mode[mode]
    if n_empty == 0:
        row = {
            "mode": mode,
            "P99_empty": np.nan,
            "P999_empty": np.nan,
            "n>0.95_empty": np.nan,
            "empty%_0.95": np.nan,
        }
    else:
        p_empty = probs_m[empty_mask]
        detected = probs_m > 0.95
        row = {
            "mode": mode,
            "P99_empty": float(np.percentile(p_empty, 99)),
            "P999_empty": float(np.percentile(p_empty, 99.9)),
            "n>0.95_empty": int((p_empty > 0.95).sum()),
            "empty%_0.95": 100.0 * int((detected & empty_mask).sum()) / max(int(detected.sum()), 1),
        }
    rows_empty.append(row)

df_empty = pd.DataFrame(rows_empty).set_index("mode")
if n_empty == 0:
    print("[NOTA] Nessun pixel vuoto in questa simulazione: la diagnostica sulla "
          "contaminazione delle zone vuote non è applicabile (mostrata come N/A).")
else:
    print("\nContaminazione delle zone vuote:")
    print(df_empty.to_string())
```

    pixel vuoti: 0 (0.00% della mappa)
    [NOTA] Nessun pixel vuoto in questa simulazione: la diagnostica sulla contaminazione delle zone vuote non è applicabile (mostrata come N/A).
    

### 7.2 Robustness: background proiettato con detection HC

Stesso test controllato del NB3 originale, con la detection HC: il fondo è stimato per fetta come mediana della densità nei pixel "quieti" ($p_{\mathrm{HC}} < 1\%$, con fallback alla mediana della mappa se i pixel quieti sono meno dello $0.1\%$), le masse sono ricalcolate con $\rho \to \max(\rho - \mathrm{bkg}, 0)$ e il fit di massa rifatto alla configurazione di riferimento $(0.95, 4)$. La detection è invariante per costruzione (la binarizzazione usa le probabilità, non la densità), quindi gli indici degli oggetti restano validi e confrontiamo direttamente $a$ prima e dopo. Nel run baseline la conclusione era "fondo assolto" ($\Delta a = -0.0018$): qui si verifica lo stesso per il catalogo HC.



```python
# ============================================================
# 7.2 — STIMA DEL BACKGROUND E RICALCOLO MASSE (detection HC)
# ============================================================
P_BKG = 0.01

def subtract_background(raw_map, probs_ref, p_bkg=P_BKG):
    """Fondo per fetta = mediana della densità nei pixel con probabilità < p_bkg
    (fallback: metà mappa se i pixel quieti sono meno dello 0.1%). Come NB3 baseline."""
    n_s = raw_map.shape[0]
    bkg = np.zeros(n_s)
    for s in range(n_s):
        sel = probs_ref[s] < p_bkg
        if sel.sum() < 0.001 * sel.size:
            sel = probs_ref[s] < np.percentile(probs_ref[s], 50)
        bkg[s] = np.median(raw_map[s][sel])
    return np.clip(raw_map - bkg[:, None, None], 0, None).astype(np.float32), bkg

raw_clean, bkg = subtract_background(raw, probs_by_mode["HC"])

B_clean    = extract_blobs(probs_by_mode["HC"], raw_clean, mpu, LBOX, t=T_BASE,
                           min_area=MIN_AREA_2D_PX)
comp_clean = link_blobs(B_clean, LBOX, n_neighbors=LINK_NEIGHBORS)
pred_clean = build_catalog(B_clean, comp_clean, LBOX, DZ, min_area_3d=AREA_BASE)
pairs_clean, pm_clean, fm_clean = match_catalogs(pred_clean, fof, LBOX, DZ)
ms_clean = mass_stats(pairs_clean, pred_clean, fof)

same_det = (len(B_clean["x"]) == len(get_blobs("HC", T_BASE)["x"]))
print(f"Detection HC: {len(B_clean['x']):,} blob | {len(pred_clean['mass']):,} oggetti "
      f"-> {'IDENTICA' if same_det else 'DIVERSA'} alla §3.1")

print(f"\n{'':<15} {'PRIMA (raw)':>12} {'DOPO (clean)':>13}")
print("-" * 44)
print(f"{'pendenza a':<15} {ms['HC']['a']:>12.3f} {ms_clean['a']:>13.3f}")
print(f"{'sigma_MAD':<15} {ms['HC']['sigma_mad']:>12.3f} {ms_clean['sigma_mad']:>13.3f}")

delta_a = ms_clean["a"] - ms["HC"]["a"]
verdict = ("FONDO ASSOLTO: non è la causa di bias"
           if abs(delta_a) < 0.01 else "effetto non trascurabile: indagare")
print(f"\nDelta a = {delta_a:+.4f} -> {verdict}")

if same_det:
    # Quota di massa rimossa per oggetto (stessi indici: detection invariata)
    m0 = res["HC"]["pred"]["mass"][res["HC"]["pairs"][:, 0]]
    m1 = pred_clean["mass"][res["HC"]["pairs"][:, 0]]
    q = 1.0 - m1 / m0
    print(f"quota di massa rimossa — mediana {100*np.median(q):.1f}%")
else:
    print("detection diversa: il confronto per oggetto non è applicabile")

```

    Detection HC: 7,055 blob | 5,747 oggetti -> IDENTICA alla §3.1
    
                     PRIMA (raw)  DOPO (clean)
    --------------------------------------------
    pendenza a             0.866         0.865
    sigma_MAD              0.129         0.129
    
    Delta a = -0.0012 -> FONDO ASSOLTO: non è la causa di bias
    quota di massa rimossa — mediana 0.3%
    

## 8. Riepilogo Finale

La tabella confronta le due detection **alla configurazione operativa HC** scelta in §5.4: stessa fotometria, stessa verità, stessa tolleranza di matching — ogni differenza è della detection. Le righe di completezza (attuale, ceiling, margine) citano la decomposizione §3.3 alla configurazione di riferimento $(0.95, 4)$; le righe di contaminazione delle zone vuote vengono dalla §7.1. *(Nel NB3 originale questa tabella conteneva due righe duplicate — "Delta a dopo sottr. background" e "Massa rimossa" ripetute con riferimenti §8.1/§8.2 inesistenti: qui sono state eliminate e i riferimenti sono corretti, correzione M4.)*

I tre pilastri da portare nel capitolo Risultati restano quelli del NB3 originale, ora con il confronto: (1) **purity fisica** — la rete segmenta materia reale in entrambe le modalità; (2) **completezza** — il confronto baseline vs HC dice se l'alto contrasto alza il catalogo o il suo ceiling; (3) **bias di massa** — il fit per cella della griglia mostra se la detection cambia la fotometria apparente o solo il numero di oggetti.

### 8.1 Tabella riassuntiva BASE vs HC alla configurazione operativa



```python
# ============================================================
# 8.1 — TABELLA RIASSUNTIVA: BASE vs HC ALLA CONFIGURAZIONE OPERATIVA HC
# ============================================================
def cell(m, col):
    sel_row = df_grid[(df_grid["mode"] == m) & (df_grid["t"] == T_FINAL_HC)
                      & (df_grid["area"] == AREA_FINAL_HC)]
    return sel_row[col].iloc[0]

def empty_metric(mode, column, digits):
    value = df_empty.loc[mode, column]
    return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"

summary = pd.DataFrame([
    ("Configurazione operativa (t, area)",
     f"({T_FINAL_HC}, {AREA_FINAL_HC})", f"({T_FINAL_HC}, {AREA_FINAL_HC})", "§5.4"),
    ("FoF (Group) sopra MASS_CUT", f"{len(fof['mass']):,}", f"{len(fof['mass']):,}", "§1"),
    ("Subhalo totali",             f"{len(sub_full['mass']):,}", f"{len(sub_full['mass']):,}", "§1"),
    ("Oggetti 3D predetti",        f"{cell('BASE','n_pred'):,}", f"{cell('HC','n_pred'):,}", "§5.1"),
    ("Coppie pred <-> FoF",        f"{cell('BASE','n_match'):,}", f"{cell('HC','n_match'):,}", "§5.1"),
    ("Completeness C",             f"{cell('BASE','C'):.3f}", f"{cell('HC','C'):.3f}", "§5.1"),
    ("Purity grezza",              f"{cell('BASE','P_raw'):.4f}", f"{cell('HC','P_raw'):.4f}", "§5.1"),
    ("Purity fisica",              f"{cell('BASE','P_phys'):.4f}", f"{cell('HC','P_phys'):.4f}", "§5.1"),
    ("  sotto MASS_CUT (reali)",   f"{int(cell('BASE','n_below'))}", f"{int(cell('HC','n_below'))}", "§4.1"),
    ("  subhalo (reali)",          f"{int(cell('BASE','n_sub'))}", f"{int(cell('HC','n_sub'))}", "§4.1"),
    ("  doppie detection",         f"{int(cell('BASE','n_double'))}", f"{int(cell('HC','n_double'))}", "§4.1"),
    ("  rumore vero",              f"{int(cell('BASE','n_noise'))}", f"{int(cell('HC','n_noise'))}", "§4.1"),
    ("Fit massa pendenza a",       f"{cell('BASE','a'):.3f}", f"{cell('HC','a'):.3f}", "§5.1"),
    ("Fit massa sigma_MAD [dex]",  f"{cell('BASE','sigma_MAD'):.3f}", f"{cell('HC','sigma_MAD'):.3f}", "§5.1"),
    ("Completeness attuale (0.95, 4)", f"{100*DEC_095['BASE']['comp_now']:.1f}%",
                                       f"{100*DEC_095['HC']['comp_now']:.1f}%", "§3.3"),
    ("Completeness ceiling (0.95, 4)", f"{100*DEC_095['BASE']['comp_ceiling']:.1f}%",
                                       f"{100*DEC_095['HC']['comp_ceiling']:.1f}%", "§3.3"),
    ("Margine algoritmico [pp] (0.95, 4)",
     f"{100*(DEC_095['BASE']['comp_ceiling']-DEC_095['BASE']['comp_now']):.1f}",
     f"{100*(DEC_095['HC']['comp_ceiling']-DEC_095['HC']['comp_now']):.1f}", "§3.3"),
    ("P99 probabilità su pixel vuoti", empty_metric('BASE', 'P99_empty', 4),
                                       empty_metric('HC', 'P99_empty', 4), "§7.1"),
    ("Pixel detezione in zone vuote @0.95 [%]",
     empty_metric('BASE', 'empty%_0.95', 3), empty_metric('HC', 'empty%_0.95', 3), "§7.1"),
], columns=["Quantità", "BASE", "HC", "Sezione"])

print(summary.to_string(index=False))

```

                                   Quantità      BASE        HC Sezione
         Configurazione operativa (t, area) (0.95, 4) (0.95, 4)    §5.4
                 FoF (Group) sopra MASS_CUT     4,701     4,701      §1
                             Subhalo totali    16,223    16,223      §1
                        Oggetti 3D predetti     5,962     5,747    §5.1
                        Coppie pred <-> FoF     3,846     3,999    §5.1
                             Completeness C     0.818     0.851    §5.1
                              Purity grezza    0.6451    0.6958    §5.1
                              Purity fisica    0.9968    0.9972    §5.1
                     sotto MASS_CUT (reali)      2084      1701    §4.1
                            subhalo (reali)        13        31    §4.1
                           doppie detection         7        12    §4.1
                                rumore vero        12         4    §4.1
                       Fit massa pendenza a     0.884     0.866    §5.1
                  Fit massa sigma_MAD [dex]     0.133     0.129    §5.1
             Completeness attuale (0.95, 4)     81.8%     85.1%    §3.3
             Completeness ceiling (0.95, 4)     99.4%     99.5%    §3.3
         Margine algoritmico [pp] (0.95, 4)      17.6      14.4    §3.3
             P99 probabilità su pixel vuoti       N/A       N/A    §7.1
    Pixel detezione in zone vuote @0.95 [%]       N/A       N/A    §7.1
    

### 8.2 Scrittura di `chosen_threshold.json` in `unet_HC/`

L'ultima cella chiude il contratto NB2 → NB3: `chosen_threshold.json` — fino ad ora mai scritto (NB2-HC verifica la sua assenza e lo segnala) — viene creato **solo qui**, nella directory di run HC, con la configurazione operativa, le metriche della cella scelta e la provenance completa (sorgente delle probabilità BASE e HC, file di metadati, impianto fotometrico, regola di scelta). Il file della baseline in `unet/` resta intatto: le due modalità restano confrontabili a posteriori. Da qui in poi il catalogo HC di LH_105 è riproducibile da un solo file di configurazione.



```python
# ============================================================
# 8.2 — CHOSEN_THRESHOLD.JSON (SOLO qui, solo in unet_HC/)
# ============================================================
chosen_path = os.path.join(RUN_DIR, "chosen_threshold.json")
with open(chosen_path, "w") as fh:
    json.dump({
        "t":      float(T_FINAL_HC),
        "area3d": int(AREA_FINAL_HC),
        "mode":   "highcontrast",
        "note":   "scelta dopo griglia 4x3 head-to-head su LH_105 (val) — conn_HC v1.0.0",
        "provenance": {
            "pipeline":       "conn_HC (NB3-HC)",
            "prob_source":    os.path.join(RUN_DIR, "val_prob_slices.npy"),
            "prob_baseline":  os.path.join(RUN_DIR_BASE, "val_prob_slices.npy"),
            "info_file":      INFO_FILENAME,
            "candidate_file": "candidate_thresholds.json",
            "photometry":     "densità I2 (X_patches.npy baseline, MED_D/SIG_D) + mpu_per_slice.npy; detection HC",
            "rules":          "P_phys >= 0.99, n_noise/n_pred <= 0.005, C massima",
        },
        "metrics": {
            "C":         float(row_final["C"]),
            "P_phys":    float(row_final["P_phys"]),
            "n_pred":    int(row_final["n_pred"]),
            "n_noise":   int(row_final["n_noise"]),
            "a_fit":     float(row_final["a"]),
            "sigma_MAD": float(row_final["sigma_MAD"]),
        },
    }, fh, indent=2)

print(f"Scritto {chosen_path}")
print(f"  t={T_FINAL_HC}, area3d={AREA_FINAL_HC}, C={row_final['C']:.3f}, "
      f"P_phys={row_final['P_phys']:.4f}, rumore={int(row_final['n_noise'])}")
print(f"\nIl file della baseline ({os.path.join(RUN_DIR_BASE, 'chosen_threshold.json')}) "
      f"resta invariato: le due modalità restano confrontabili a posteriori.")

```

    Scritto C:\Users\kekko\Downloads\Python\TESI\CAMELS_data\unet_HC\chosen_threshold.json
      t=0.95, area3d=4, C=0.851, P_phys=0.9972, rumore=4
    
    Il file della baseline (C:\Users\kekko\Downloads\Python\TESI\CAMELS_data\unet\chosen_threshold.json) resta invariato: le due modalità restano confrontabili a posteriori.
    
