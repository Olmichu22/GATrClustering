# GATrClustering — dossier de estado

**Fecha:** 2026-08-06 · **Referencia actual:** s14a (K=3, e/pi/mu) · **Rama:** `main`

Documento para discutir cómo mejorar. Recoge qué hace el sistema, qué se ha
probado, qué ha fallado y por qué, y dónde están los cuellos de botella reales.

---

## 1. El problema

Clasificar eventos del SDHCAL (test beam 2012) en electrón / pión / muón **sin
verdad de referencia fiable**. `particle_type` existe en los ficheros pero es
otro clasificador ruidoso, no una etiqueta: a 70 GeV su flag de electrón es
bimodal (nHits p10/50/90 = 17/40/876), es decir, misID. Por eso el enfoque no es
clasificación supervisada sino **clustering guiado por anchors**: unos pocos
cientos de eventos revisados a mano fijan la semántica de cada cluster y el
resto de la muestra se organiza sola.

Consecuencia metodológica que atraviesa todo el proyecto: **no hay matriz de
confusión que valga**. Las métricas son (a) accuracy sobre anchors *held-out*
(los que cayeron en el split de validación y nunca entraron ni en la CE ni en la
inicialización de prototipos) y (b) diagnósticos no supervisados — ocupación de
clusters, distribución de nHits por cluster, proyecciones del latente.

---

## 2. Arquitectura

```
hits (x,y,z,thr,k)
  └─ GATrEncoder            geometric algebra transformer, equivariante E(3)
       └─ agregación         hits → 1 vector por evento
            └─ PrototypeHead  proyección + L2-norm → z en la hiperesfera
                 └─ logits_k = cos(z, c_k) / tau
```

| Pieza | Fichero | Nota |
|---|---|---|
| Encoder | `src/models/gatr_module.py` | 4 bloques, 16 canales MV / 64 escalares, dropout 0.2. Requiere GPU (no corre en CPU). |
| Agregación | `src/models/aggregation.py` | `mean`, `attention` (PMA), `token`, `attention_card`, **`attention_density`** (el que se usa). |
| Cabeza | `src/models/prototype_head.py` | Prototipos **aprendibles**, inicializados con la media de los anchors de cada clase. `temperature: 0.25`. |
| Modelo | `src/models/clustering_model.py` | Se construye entero desde el YAML. |

### El pooling es una decisión de física, no de ingeniería

Mean/attention pooling **promedian** los hits: el embedding resultante es ciego a
la cardinalidad, y electrón y muón colapsan. `attention_card` añade `log1p(nHits)`,
pero nHits crece con la *profundidad* que atraviesa el shower. `attention_density`
usa **hits por capa activa** (compacidad transversal), que describe la forma del
shower con independencia de cuántas capas recorre. Es el que usan todos los runs
recientes.

### Loss (`src/train_clustering.py`, ensamblado en `training_step`)

```
L = w_swap · L_swap  +  λ_ce · L_ce  +  λ_vic · L_vicreg  +  λ_prior · KL(prior ‖ p̄)
```

- **`L_swap`** (`losses/swap_loss.py`) — SwAV sin Sinkhorn: dos vistas del mismo
  evento vía `hit_dropout: 0.25`; la asignación de una predice la otra. Es el
  término no supervisado que hace el trabajo.
- **`L_ce`** — cross-entropy sobre los anchors del split de train. Es lo único
  que ancla la *semántica* (qué cluster es "electrón").
- **`L_vicreg`** (`losses/vicreg.py`) — varianza + covarianza sobre `z_raw`
  (antes de normalizar; en `z` normalizado el hinge pelearía con la norma).
  Evita el colapso del embedding a un punto. Sin término de invariancia: ese
  papel lo cubre el swap.
- **`KL(prior ‖ p̄)`** (`losses/prior_loss.py`) — anticolapso de *asignación*.
  El swap tiene un mínimo trivial (todo a un cluster) y VICReg no lo ve porque
  solo mira `z`. El KL diverge cuando la masa de un cluster tiende a 0. Prior por
  clase medido sobre la muestra, con warmup y EMA.
- `losses/domain_align.py` existe (alineamiento sim↔datos) pero está desactivado
  en todos los runs vigentes.

### Datos (`src/data/dataset.py`)

Lector HDF5 plano (`offsets` + arrays de hits). Lo que importa para configurar:

- `anchor_datasets` — ficheros externos cuyos eventos entran **todos** con una
  etiqueta forzada (así entran los electrones del run dedicado).
- `ignore_anchor_labels` / `remap_anchor_labels` — el orden es **ignore primero,
  remap después** (se corrigió el 2026-08-05; al revés, un remap podía colisionar
  con un índice ignorado y desetiquetar en silencio una clase entera).
- `hit_affine` — transformación afín por coordenada, aplicada a **todos** los
  ficheros del reader. Ver §3: es la trampa más cara del proyecto.
- `filters` / `min_hits` — corte de calidad opcional.

### Evaluación

| Script | Qué hace |
|---|---|
| `src/evaluate_clustering.py` | Re-hace el split y evalúa **solo validación**: acc de anchors held-out, ocupación, nHits por cluster, proyecciones. |
| `src/infer_eval.py` | Corre un checkpoint sobre **todos** los eventos de un h5 (sin split ni filtro) y vuelca `assignments_full.npz` con `cluster/particle_type/anchor/nhits/energy/filter_status` + cross-tab por consola. Es la evaluación que vale. |
| `src/projections.py` | `pca`, `proto`, `tsne`, `umap`. **t-SNE descartado**: z vive en la hiperesfera, un cluster tiene el 97% de la muestra (t-SNE lo trocea en islas falsas) y no hay transform out-of-sample para los prototipos. |
| `src/latent_explorer_demo.py` | Dash sobre `latent_explorer.npz`; permite ver el evento 3D de cada punto. |

**Proyección `proto`**: similitudes coseno `z·p_k` centradas por fila y mapeadas
sobre un K-gono regular. Lineal, determinista, exacta para los prototipos y sin
pérdida para K=3 — muestra exactamente la geometría que usa la cabeza. Es la que
hay que leer para diagnosticar deriva de prototipos.

> Nota: el muestreo del explorer (`explorer_selection`) es **estratificado** —
> los clusters raros entran enteros. La densidad de puntos **no** es la ocupación
> real; esa se lee del cross-tab.

---

## 3. Los datos y sus marcos de referencia

Tres marcos distintos conviven, y confundirlos ha costado ya varios experimentos:

| Fichero | Unidades | Capa `k` | `z` |
|---|---|---|---|
| `data/E70GeV_2012.h5` (marco del pipeline) | mm | 0-based (0..47) | 226.5 .. 1542.5 |
| `data/filtered/E70GeV_2012_filtered.h5` | cm | **1-based** (1..50) | 2.8 .. 140.0 |
| `data/filtered/E30GeV_2012_filtered.h5` | cm | **0-based** (0..48) | 0.0 .. 134.4 |

Por tanto el `hit_affine` correcto **no es el mismo para 70 y 30 GeV**:

```yaml
# 70 GeV filtrado -> marco del pipeline
hit_affine: {x: [10.0, -5.204], y: [10.0, -5.204], z: [10.0, 198.5], k: [1.0, -1.0]}
# 30 GeV filtrado -> marco del pipeline   (¡una capa de diferencia!)
hit_affine: {x: [10.0, -5.204], y: [10.0, -5.204], z: [10.0, 226.5], k: [1.0,  0.0]}
```

**Deuda abierta:** todas las inferencias hechas hasta hoy sobre el fichero de
30 GeV usaron el afín de 70 GeV, es decir, con las capas desplazadas una posición
y `k = -1` fuera del rango del minmax. Los resultados de 30 GeV de §5 hay que
releerlos con esa reserva y, si el 30 GeV pasa a ser línea principal, repetirlos.

### Escala de nHits con la energía

Es el otro hecho estructural. Medido:

| | muón p50 | pión p50 | electrón p50 |
|---|---|---|---|
| 70 GeV | 74 | 995 | (flag no fiable) |
| 30 GeV | 76 | 469 | 296 |

El muón no se mueve (es un MIP) pero el pión se reduce a la mitad. Un modelo
entrenado a 70 GeV y aplicado a 30 GeV manda el 68% de los piones al cluster de
muones. **La energía no es una feature del modelo** (a 70 GeV era una columna
constante). Ésta es la motivación del "plan b" (energía como escalar global +
mezcla 30/70 GeV), construido pero nunca lanzado.

---

## 4. Etiquetado manual — `anchor_labeler/`

App Flask+Plotly, fuera del contenedor GATr, que propone eventos por clase, los
dibuja en 3D y registra la decisión. Nació al concluir que **los anchors
automáticos no son la partícula que dicen ser** (decisión de 2026-08-03): todos
los intentos previos de fabricar anchors a partir de flags o de cortes
geométricos acabaron enseñándole al modelo la etiqueta equivocada.

- `run_labeler.sh --config <yml> --port <n>`, `install.sh` crea `.venv_viewer`.
- Sesiones en `sessions/*.json` (autosave 5 s), exports en `sessions/exports/`.
- `teardown_labelers.sh` exporta **y verifica antes de matar nada**: escribe el
  primario con `anchor_label` relleno y los subsets por clase, comprueba que los
  conteos del h5 cuadran con la sesión, y solo entonces apaga los servidores.
- Convención de etiquetas (≠ índice de cluster):
  `0 electron · 1 pion · 2 muon · 3 multip · 4 penetrante · 5 ruido`.

**Dos fuentes por energía**, porque en el primario no hay electrones creíbles a
70 GeV: la sesión del primario (pi/mu/penetrante/ruido) y una sesión sobre el run
dedicado de electrones, donde el trabajo es *quitar contaminación*, no encontrar
la clase.

---

## 5. Qué se ha aprendido (comprimido)

| Etapa | Resultado |
|---|---|
| Pooling | Promediar hits ⇒ ciego a cardinalidad ⇒ e/mu colapsan. Resuelto con `attention_density`. |
| Anchors de sim | Fuera de dominio ⇒ sobreajuste. |
| Anchors del e-beam dedicado | Isla latente aislada a la que los electrones reales del TB nunca llegan. |
| Anchors in-domain (s6) | Cierra el hueco: val/loss 2.94→0.38, acc 0.50→0.97. |
| Repesos (s7/s8) | Fallan los dos por razones geométricas. |
| s9/s10 @30 GeV | e↔mu resuelto (mu 99% limpio); **e↔pi es el cuello**: ~30% de piones se cuelan. |
| **s14 (anchors 100% manuales, 833k)** | pi/mu perfectos; el electrón sigue sin salir a 70 GeV. Ver abajo. |

**Trampa metodológica recurrente**: contar anchors dentro de las métricas infla
todo (el modelo los ha visto etiquetados). Excluir siempre `anchor >= 0`, o
reportar sobre *todos* los eventos dejando claro que los anchors van dentro.

### Barrido s14 (veredicto 2026-08-06)

Los tres comparten primario, unidades, features, modelo y optimizador; **solo
cambia qué clases se convierten en cluster**.

| | s14a (K=3) | s14b (K=4 +ruido) | s14c (K=5 +penetrante) |
|---|---|---|---|
| acc anchors held-out | **0.812** | 0.720 | 0.726 |
| e-anchors correctos | 0/37 | 0/37 | 7/37 |
| cluster-e @30 GeV | **4845 ev, 72% puro** | muerto (N=1) | 3150 ev, 50% puro |
| cluster-e @70 GeV | 30 ev (basura) | vacío | 3 ev |
| clusters nuevos | — | ruido: p50 701 hits, 71% piones | ruido 7 ev, **penetrante vacío** |

1. **Un cluster de ruido aprendido no aprende ruido.** Los 281 806 eventos casi
   vacíos (p50 13 hits) van al cluster de **muones** en los tres runs, incluido
   el que tenía cluster de ruido con sus propios anchors. El corte de calidad a
   mano se queda.
2. **26 anchors no sostienen un cluster.** `penetrante` sale vacío a las dos
   energías; la banda 150–350 hits sigue yendo 83% a muones.
3. **Piones y muones están resueltos** (mu recall 1.00, pi purity 0.98 @70 GeV) y
   son idénticos en los tres runs.

---

## 6. Los problemas abiertos, por impacto

### (A) Deriva de prototipos — el cuello de botella real

Con la métrica de validación ya arreglada (antes se calculaba sobre ~14 anchors
por un tope de recolección, así que la selección de checkpoint era ruido), la
accuracy held-out **oscila entre 0.45 y 0.72 y luego se encalla en ~0.52** hasta
el final del entrenamiento. Los "mejores" checkpoints son transitorios.

`eval/proj_proto.png` dice exactamente qué pasa: **los anchors de electrón
forman un blob compacto y separado — la representación distingue electrones —
pero el prototipo de electrón está aparcado en una región vacía del latente.**
No es colapso de representación, es que la cabeza pierde el prototipo.

Palancas plausibles, de menos a más invasiva:
1. **EMA / congelado de prototipos**: actualizarlos como media móvil de los
   anchors de su clase en vez de por gradiente, o congelarlos tras la
   inicialización durante N épocas.
2. **Re-inicializar desde anchors** cada época (barato, ya existe
   `init_prototypes_from_anchors`).
3. **Penalizar la distancia prototipo↔centroide de sus anchors** como término
   explícito.
4. Revisar la interacción `λ_prior` × clase minoritaria: el prior empuja masa
   hacia un cluster cuyo prototipo está donde no hay datos; en s14b `p̄_ruido`
   se quedó en 0.055 contra un objetivo de 0.33 y `loss/prior` no bajó de 0.37
   en todo el run. El prior puede estar *causando* la deriva.

### (B) El electrón solo existe a 30 GeV

El cluster de electrones de s14a es basura a 70 GeV (30 eventos) y funciona a
30 GeV (4845 eventos, 72% puros). No es que el prototipo esté muerto: está donde
viven los electrones de 30 GeV. Simétricamente, los piones de 30 GeV se
encogen y caen en el cluster de muones. **Evaluar a una sola energía engaña en
las dos direcciones.**

Palanca: energía como escalar global + entrenamiento con mezcla 30/70 GeV
(construido en `configs/planb_s11.yml`, nunca lanzado). Requiere resolver antes
lo del `hit_affine` de §3.

### (C) La banda penetrante (150–350 hits)

Sin resolver desde s12. Ni `multip` ni 26 anchors propios la capturan. A 30 GeV
la banda tiene 25 179 eventos (1.58% frente al 0.3% de 70 GeV) y ahora se solapa
con la cola baja de los piones reales — puede que a 30 GeV se pueda etiquetar en
volumen y deje de ser un problema de escasez.

### (D) Infraestructura

- Condor **spolea el ejecutable**: derivar el repo de `$BASH_SOURCE` apunta al
  spool y muere sin logs. Pasar `REPO=` en `environment` y hacer `tee` del log
  propio; `output`/`error` del `.sub` no vuelven.
- `condor_q <id> -af` devuelve vacío de forma intermitente; grepear
  `condor_q -nobatch`.
- **Control de versiones (paso 0, no punto final).** `main` local tiene historia
  real, pero `origin/main` sigue clavado en el `Initial commit`: nada se ha
  empujado nunca. De ahí que los worktrees salgan vacíos — `EnterWorktree` parte
  de `origin/main`. Además `condor/` estaba en `.gitignore`, así que los `.sub`
  (la definición reproducible de cada job) no se versionaban, y la clave de W&B
  estaba hardcodeada en seis scripts. Corregido el 2026-08-06: `condor/` se
  versiona, la clave vive en `.wandb_key` (gitignored, 0600) y el árbol entero
  está commiteado. **Pendiente: `git push origin main`** — hasta entonces la
  historia solo existe en esta máquina y los worktrees siguen rotos.

---

## 7. Estado ahora mismo

- **Referencia**: `results/manual_s14a/20260805_135126_s14a_epimu`, checkpoint
  `best-epoch009`. Con una reserva: s14a **no se relanzó** tras los bugfixes del
  05-08, así que su checkpoint se eligió con la métrica rota. Conviene repetirlo
  (`condor_submit condor/manual_s14a.sub`) antes de tomarlo como base.
- **Etiquetado 30 GeV en marcha**:
  - `:8050` → primario 1.6M (`configs/labeler_e30_filtered.yml`), propone
    30 piones + 30 muones + 30 de la banda penetrante por ronda. Sesión:
    `sessions/filtered30_2012.json`.
  - `:8051` → electrones dedicados 30 GeV (`configs/labeler_elec30_filtered.yml`);
    `watch_elec30_labeler.sh` lo arranca en cuanto el h5 exista y esté completo.
- A 30 GeV el flag de electrón del **primario** sí parece electrón (18 822 ev,
  p10/50/90 = 82/296/410), al contrario que a 70 GeV. Están como *correction
  target*; descomentar su `source` daría anchors de electrón **in-domain**, que
  es justo lo que arregló la isla latente en su día.

## 8. Orden propuesto

1. Arreglar el `hit_affine` de 30 GeV antes de entrenar nada con esa muestra.
2. Etiquetar 30 GeV (en curso) — piones/muones en el primario, electrones aparte.
3. Atacar la deriva de prototipos sobre la base K=3, **no** añadiendo clases.
4. Con (1)+(2)+(3): entrenamiento mixto 30/70 GeV con la energía como escalar.
