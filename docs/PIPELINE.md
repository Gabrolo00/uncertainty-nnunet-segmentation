# Pipeline: dal dataset ai risultati

Questo documento descrive il giro completo per stimare e valutare l'incertezza di segmentazione con ensemble di nnU-Net v2: costruzione dei dataset, training, inferenza, calibrazione, mappe di incertezza, soglie, valutazione, analisi, autoencoder e viewer. Gli esperimenti fatti e i risultati sono in `EXPERIMENTS.md`.

## 1. Dove sono le cose

**Codice:** questa repository (sul server in `/home/grolo/thesis_pipeline/`), una cartella per fase. Ogni script ha in testa due righe su cosa fa e come si lancia.

| Cartella | Contenuto |
|---|---|
| `1_data/` | costruzione dei dataset da TotalSegmentator e preparazione del test esterno Humanitas |
| `2_models/training/` | training (deep ensemble, checkpoint ensemble, fine-tuning) e i loro launcher |
| `2_models/inference/` | inferenza con salvataggio dei logit e i loro launcher |
| `3_uncertainty/` | temperature scaling, mappe, DEU/EBCU, mappe combinate, soglie, valutazione; i launcher `pipeline_*.sh` eseguono tutto per un modello |
| `4_analysis/` | statistica, confronti tra modelli e domini, valutazione del test esterno, script delle figure (`figures/`) |
| `5_autoencoder/` | autoencoder sulle mappe di incertezza; i launcher sono numerati nell'ordine di esecuzione |
| `6_viewer/` | i due viewer Streamlit |

**Dati:** tutto in `/mnt/sda2/grolo/`.

| Cartella | Contenuto |
|---|---|
| `nnUNet/raw/` | dataset nnU-Net (`Dataset904_KidneyTSMechanical`, `Dataset903_TracheaTS`), con `splits_final.json` dei 5 fold |
| `nnUNet/results/` | pesi dei modelli (una cartella per configurazione di training) |
| `kidney/predictions/{deep,checkpoint}/` | segmentazioni e logit del test interno del rene |
| `kidney/{deep,checkpoint}/` | risultati del rene per modello: T\*, mappe, DEU/EBCU, soglie, valutazione, statistica |
| `kidney/finetune/` | T\* del modello fine-tuned |
| `kidney/external/` | test esterno Humanitas: ritaglio, predizioni, mappe e valutazione (from scratch e fine-tuned) |
| `kidney/autoencoder/` | dataset, modelli e valutazione dell'autoencoder |
| `trachea/...` | come il rene (`predictions/`, `deep/`, `checkpoint/`) |
| `comparisons/` | confronti tra modelli, strutture e domini |
| `humanitas_source/` | CT e contorni Humanitas convertiti in NIfTI, coorte, controllo qualità, `inventory/` |
| `labels_source/` | label binarie estratte da TotalSegmentator |
| `logs/` | log di training e pipeline |

**Dati di partenza (sola lettura):** TotalSegmentator in `/mnt/sda2/DATASETS/TOTALSEGMENTATOR`, Humanitas (DICOM e RTSTRUCT) in `/mnt/sda2/DATASETS/AUTOMI`.

## 2. Ambienti

| Ambiente | Uso |
|---|---|
| `source /home/grolo/venv_explainability/bin/activate` | analisi: T\*, mappe, soglie, valutazione, statistica, autoencoder, viewer |
| `/home/aiart/anaconda3/envs/nnunet_env/bin/python` | nnU-Net: preprocessing, training, inferenza (ha SimpleITK) |
| `/home/aiart/anaconda3/envs/nnunet_medcam/bin/python` | conversione e ritaglio delle CT Humanitas |

Variabili usate dai launcher di nnU-Net:
```bash
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims      # evita un errore di import di nnU-Net in questo ambiente
export NUMPY_MADVISE_HUGEPAGE=0                  # evita il forte rallentamento sulle CT grandi
```
I preprocessati non sono più su disco: per riaddestrare si rigenerano con `2_models/training/preprocess_kidney904.sh`. Inferenza e analisi non ne hanno bisogno.

## 3. Pipeline completa

### 3.1 Dataset (`1_data/`)
- **Label binarie** da TotalSegmentator: `build_kidney_labels.py` (rene sinistro OR destro), `build_trachea_labels.py`. Solo criteri di esclusione meccanici: CT presente e leggibile da SimpleITK, struttura presente. Per la trachea si tengono anche 45 CT senza trachea ma con un campo di vista che potrebbe includerla, come casi negativi.
- **Split e dataset nnU-Net:** `build_kidney_mechanical_dataset.py` (rene), `select_trachea_split.py` (trachea, stratificato per presenza della struttura e scanner). Il test non entra mai nei fold di training/validazione.
- **Test esterno Humanitas:** `inventory_humanitas.py` (inventario dei DICOM), `prepare_humanitas_kidney.py --stage cohort|convert|qa` (un contorno per paziente, conversione in NIfTI, controllo qualità). Il ritaglio in z attorno ai reni (`crop_humanitas_kidney.py`) si fa dopo un'inferenza veloce, senza usare la reference.

### 3.2 Training (`2_models/training/`)
- **Deep ensemble:** 5 modelli nnU-Net 3d_fullres, uno per fold, 1000 epoche (`train_deep_ensemble.py`).
- **Checkpoint ensemble:** un solo training sul fold 0 con learning rate ciclico, 3 cicli da 400 epoche; i membri sono 6 snapshot (quartultima e ultima epoca di ogni ciclo) (`train_checkpoint_ensemble.py`).
- **Fine-tuning:** come il deep ensemble, partendo dai pesi pretrained di TotalSegmentator (teste di segmentazione escluse) (`train_deep_ensemble_finetune.py`).
- Launcher: `preprocess_kidney904.sh`, `train_kidney904_deep.sh`, `train_kidney904_finetune.sh`, `train_kidney904_checkpoint_and_trachea_deep.sh`, `train_trachea_checkpoint.sh`.

### 3.3 Inferenza (`2_models/inference/`)
`predict_from_raw_data.py` è l'inferenza di nnU-Net modificata: media dei logit dei membri, TTA a specchio di nnU-Net, sliding window, e con `--save_logits` salva:
- `<caso>.nii.gz`: segmentazione (argmax della media dei logit);
- `<caso>_logits.pt`: logit medi dell'ensemble;
- `fold_logits/<caso>_fold_<k>_logits.pt`: logit di ogni membro (servono per le misure di disaccordo).

Si lancia con `run_inference.py` (deep) o `run_checkpoint_ensemble_inference.py` (checkpoint). L'inferenza è l'unico passo costoso: tutto il resto lavora sui file salvati.

Per il temperature scaling serve anche l'inferenza **out-of-fold** sui casi di training (ogni fold predice i propri casi di validazione), poi `collect_oof_logits.py`. Per il checkpoint ensemble si usano i casi di validazione del fold 0. Launcher: `inference_kidney904_deep_oof.sh`, `inference_trachea_deep.sh`, `inference_kidney904_finetune.sh`, `inference_humanitas_kidney904.sh` (passaggio veloce sulla CT intera, ritaglio, ensemble completo sul ritaglio).

### 3.4 Temperature scaling (`3_uncertainty/temperature/`)
`loo_temperature_scaling.py`: per ogni caso out-of-fold, T\* che minimizza la NLL sugli altri casi, cercata su una griglia logaritmica a due livelli; T\* del modello = mediana. Le metriche (NLL, ECE, Brier) sono calcolate nella **regione di interesse**: dilatazione di 15 mm dell'unione di reference e predizione. La temperatura divide i logit e non cambia la segmentazione.

### 3.5 Mappe di incertezza (`3_uncertainty/maps/`)
`total_uncertainty_measures.py --T <T*>`: una mappa per misura e caso, `<caso>_<misura>.nii.gz`.

| Famiglia | Misure |
|---|---|
| mean prediction | entropy, confidence_map, sample_mean_uncertainty (SMU) |
| disagreement | variance, std, mutual_information, epkl, max_fold_deviation, boundary_disagreement |
| anti-confidence | anti_confidence_fn, anti_confidence_fp, anti_confidence_total |
| distance-aware | DEU_entropy, ebcu_entropy (passo 3.6) |
| scomposizione | expected_entropy (aleatoria), total_entropy |

Le mappe sono salvate a 8 bit (256 livelli): effetto piccolo e leggermente pessimistico sulle metriche.

### 3.6 Misure distance-aware (`3_uncertainty/distance_aware/`)
1. `build_baseline_EU_given_d.py`: tabella dell'entropia attesa in funzione della distanza dal bordo predetto (per mm, dentro e fuori), stimata sulle mappe out-of-fold (checkpoint: validazione del fold 0).
2. `compute_DEU.py`: DEU = max(U − E[U | d], 0).
3. `make_ebcu_maps.py`: EBCU = quanto U supera il 90° percentile atteso alla stessa distanza, scalato tra 90° e 99°, in [0, 1].

### 3.7 Soglie (`3_uncertainty/thresholds/`)
`Calibrate_threshold_loo_v2.py`: ogni mappa è normalizzata in [0, 1] per caso (min-max); per ogni caso la soglia τ\* è quella che, sugli altri casi, massimizza il Dice tra mappa binarizzata ed errori di segmentazione nella regione di interesse (leave-one-out, ricerca grossolana poi fine; `--cache_case_scores` per andare più veloci). Sul test esterno non si ricalibra: si usa la soglia di riepilogo del test interno.

### 3.8 Valutazione (`3_uncertainty/evaluation/`)
`evaluate_threshold.py`: per caso e mappa, alla τ\* del caso: recall dei falsi positivi, dei falsi negativi e totale, precision, F1 (= `dice_unc_vs_errors`, la metrica principale), tasso di falsi allarmi. Due regioni: **ROI 15 mm** e **no-border**, cioè la ROI senza la banda di ±2 mm attorno al bordo predetto (`--border_mm` per altre larghezze). Output `metrics_per_case_4.csv`.

### 3.9 Statistica e confronti (`4_analysis/`)
- `uncertainty_ranking_stats.py`: Friedman sulle misure, poi Wilcoxon tra la migliore e ognuna delle altre, correzione di Benjamini-Hochberg; gruppo di testa = misure non significativamente peggiori della migliore. Output `ranking_stats/step1_stats_all.csv`.
- `compare_deep_vs_checkpoint.py`, `table_maps_deep_vs_checkpoint.py`: confronti appaiati tra ensemble (Wilcoxon).
- `ensemble_dice_diversity_calibration.py`: Dice, diversità dei membri e calibrazione per caso.
- `compare_rankings_organs_domain.py`, `compare_rankings_checkpoint_pairs.py`: concordanza delle classifiche (Spearman) tra strutture, ensemble e domini; classifica per famiglia.
- `domain_shift_stats.py`: test interno contro esterno (Mann-Whitney).
- `froc_eval.py`: confronto a parità di volume di falsi allarmi sulle zone di errore lontane dal bordo.
- `humanitas_eval.py`, `humanitas_member_diversity.py`: test esterno (soglie trasferite, segmentazione, calibrazione, AUROC/AUPRC senza soglia).

### 3.10 Mappe combinate (`3_uncertainty/combined/`)
`make_threshold_scaled_combined_maps.py`: ogni misura è divisa per la propria τ\*, così 1 vuol dire "sopra soglia" per tutte. Regole: switch entropia/std (entropia entro 2 mm dal bordo, std oltre), massimo entropia/std, switch DEU/MI. Soglia 1 sulla mappa unita, nessuna nuova calibrazione. Launcher `combined_maps_all_models.sh`.

### 3.11 Launcher che fanno tutto per un modello (`3_uncertainty/`)
`pipeline_kidney904_deep.sh`, `pipeline_kidney904_checkpoint.sh`, `pipeline_trachea_deep.sh`, `pipeline_trachea_checkpoint.sh`: T\* → mappe → DEU/EBCU → soglie (14 misure) → valutazione → statistica. `pipeline_humanitas_kidney904.sh`: lo stesso sul test esterno con T\*, tabella E[U | d] e soglie del test interno. `pipeline_kidney904_finetune.sh`: T\* e mappe del fine-tuned e confronto sul test esterno. `decomposition_kidney904.sh`: scomposizione aleatoria/epistemica.

### 3.12 Autoencoder (`5_autoencoder/`)
Un autoencoder 3D impara la mappa di incertezza "attesa"; il residuo (eccesso U − Û e mancanza Û − U) stima l'incertezza inattesa. Si esegue con `AE_DATASET=904` e i launcher in ordine:
1. `autoencoder_1_build_dataset_*.sh`: dataset sulla griglia della rete (1.5 mm), con entropia, logit, CT, predizione, reference, distanza dal bordo; casi di test interni ed esterni; 5 gruppi di cross-fitting.
2. `autoencoder_2_select_configuration.sh`: scelta di profondità e collo di bottiglia su un gruppo di selezione.
3. `autoencoder_3_train_crossfit.sh`, `autoencoder_4_infer_residuals.sh`: un modello per gruppo e variante di ingresso (a: entropia; b: + logit; c: + CT), ricostruzione e residui sui casi non visti.
4. `autoencoder_5_evaluate.sh`: AUPRC normalizzata nelle viste standard, no-border e target; Wilcoxon con correzione di Holm contro entropia ed EBCU.
5. `autoencoder_6`–`8`: training sui soli casi ben segmentati e su un controllo casuale, valutazione, combinazioni C1–C4 (`autoencoder_combine.py`), analisi per zona (`autoencoder_zones.py`).

### 3.13 Viewer (`6_viewer/`)
```bash
source /home/grolo/venv_explainability/bin/activate
cd /home/grolo/thesis_pipeline/6_viewer
streamlit run uncertainty_viewer.py --server.port 8501   # segmentazioni e mappe, tutti i modelli
streamlit run autoencoder_viewer.py --server.port 8502   # mappe dell'autoencoder
```

## 4. Convenzioni da ricordare
- **Regione di interesse:** dilatazione di 15 mm dell'unione di reference e predizione, per calibrazione, soglie e valutazione.
- **No-border:** si escludono i voxel entro ±2 mm dal bordo predetto. **Vista target** (autoencoder e benchmark): positivi = errori oltre la banda, negativi = tutti i voxel corretti della ROI, banda compresa.
- **AUPRC normalizzata:** (AUPRC − p) / (1 − p), con p la frazione di positivi: 0 = mappa casuale, 1 = perfetta.
- **T\*** si stima per ogni modello e non si riusa mai tra modelli.
- **Test esterno:** nessun parametro ricalibrato (T\*, tabella E[U | d] e soglie dal test interno); analisi principale sui 107 casi con contorni clinici; i falsi negativi dentro l'inviluppo convesso del rene predetto (seno renale, convenzione diversa) sono esclusi dai positivi nelle viste dell'autoencoder.
- **Trachea:** i 9 casi di test senza trachea hanno ROI vuota e sono esclusi da calibrazione e metriche delle mappe.
