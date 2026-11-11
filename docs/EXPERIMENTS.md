# Esperimenti

Riassunto degli esperimenti della tesi e dei risultati principali. Come si eseguono è descritto in `PIPELINE.md`. I percorsi dei risultati sono relativi a `/mnt/sda2/grolo/`.

## 1. Dati

| | Rene | Trachea | Rene, test esterno |
|---|---|---|---|
| Fonte | TotalSegmentator | TotalSegmentator | Humanitas (CT di pianificazione radioterapica, contorni clinici) |
| Dataset | `Dataset904_KidneyTSMechanical` | `Dataset903_TracheaTS` | `humanitas_source/` |
| Training | 234 (5 fold da 187/47) | 236 (5 fold da 188-189/47-48) | — |
| Test | 97 | 60 (51 con trachea + 9 senza) | 110 (107 contorni clinici + 3 automatici) |

- **Criteri di esclusione solo meccanici:** CT presente e leggibile, struttura presente. Per la trachea sono tenute 45 CT senza trachea ma con un campo di vista che potrebbe includerla (casi negativi, divisi in proporzione tra training e test).
- **Humanitas:** un contorno per paziente (clinico prima di quello automatico), CT whole body ritagliate in z attorno ai reni senza usare la reference. Fette da 5 mm nella maggior parte dei casi (pochi casi a 2.5-3 mm, 2 a 7.5 mm), contro 1.5 mm di TotalSegmentator. Il contorno clinico include il seno renale, che TotalSegmentator esclude: è la causa principale dei falsi negativi sul test esterno.
- Il test interno di TotalSegmentator era nei dati di pretraining dei pesi pubblici: per questo il modello fine-tuned è valutato solo sul test esterno.

## 2. Modelli

| Modello | Struttura | Training | Membri | T\* |
|---|---|---|---|---|
| Deep ensemble | rene | 5 fold × 1000 epoche, pesi casuali | 5 | 2.075 |
| Checkpoint ensemble | rene | fold 0, 3 cicli × 400 epoche, LR ciclico | 6 snapshot | 2.126 |
| Deep ensemble fine-tuned | rene | come il deep, dai pesi pretrained di TotalSegmentator | 5 | 2.178 |
| Deep ensemble | trachea | 5 fold × 1000 epoche, pesi casuali | 5 | 2.704 |
| Checkpoint ensemble | trachea | fold 0, 3 cicli × 400 epoche | 6 snapshot | 2.976 |

Validation Dice del deep ensemble del rene: 0.946 / 0.973 / 0.927 / 0.970 / 0.954 (fine-tuned 0.944 / 0.970 / 0.927 / 0.976 / 0.960). Pesi in `nnUNet/results/`.

## 3. Deep ensemble contro checkpoint ensemble

Test interno; Wilcoxon appaiato. Risultati in `comparisons/deep_vs_checkpoint_{kidney,trachea}/` e `comparisons/ensemble_dice_diversity/`.

| | Rene deep | Rene checkpoint | Trachea deep | Trachea checkpoint |
|---|---|---|---|---|
| Dice | 0.937 ± 0.157 | 0.924 ± 0.170 | 0.969 ± 0.035 | 0.939 ± 0.046 |
| NLL T = 1 → T\* | 0.150 → 0.081 | 0.156 → 0.085 | 0.045 → 0.024 | 0.095 → 0.048 |
| ECE T = 1 → T\* | 0.016 → 0.013 | 0.017 → 0.014 | 0.0063 → 0.0048 | 0.0117 → 0.0071 |
| Dice medio tra membri | 0.966 | 0.946 | 0.984 | 0.956 |
| MI media nella ROI (×10⁻³ bit) | 3.11 | 5.80 | 2.81 | 7.27 |

- **Il deep ensemble è più accurato e meglio calibrato** su entrambe le strutture (rene: migliore in 89/97 casi; trachea: in 50/51). Il distacco è maggiore sulla trachea.
- **Tutti i modelli sono sovraconfidenti** (T\* > 2). Il temperature scaling riduce la NLL di circa il 46-49% e non cambia la segmentazione.
- **I membri del checkpoint sono più diversi** (circa 2-3 volte la varianza e la MI) ma singolarmente meno accurati: le loro mappe di disaccordo coprono più errori con precisione più bassa, senza migliorare l'F1.
- **Mappe:** sul rene, nella ROI, le F1 dei due ensemble sono equivalenti (0.33-0.40, EBCU esclusa). Fuori dal bordo, con il deep vincono le misure di disaccordo (variance 0.172, MI 0.166); con il checkpoint vince la DEU (0.172). Sulla trachea il checkpoint ha F1 più alte fuori dal bordo, ma solo perché sbaglia di più e ha più errori da trovare.
- **Coerenza tra strutture:** con il deep ensemble la classifica delle 14 misure è molto simile tra rene e trachea (Spearman 0.93 nella ROI, 0.86 nel no-border); con il checkpoint no (0.67 e −0.11).

## 4. Fine-tuning dai pesi di TotalSegmentator

Rene, test esterno (107 casi clinici). Risultati in `kidney/external/eval_scratch_vs_finetune/`.

- **Segmentazione:** Dice 0.851 (from scratch) contro 0.850 (fine-tuned), p = 0.10.
- **Calibrazione:** a T = 1 il fine-tuned è un po' più sovraconfidente (NLL 0.648 contro 0.626); dopo il proprio T\* sono equivalenti (0.318 contro 0.321).
- **Diversità:** il fine-tuned ha membri più simili (varianza e MI circa −40%), ma il suo disaccordo è più concentrato sugli errori: AUROC della MI 0.723 → 0.789 nella ROI e 0.625 → 0.711 nel no-border.
- **Conclusione:** il pretraining non migliora né segmentazione né calibrazione su dati esterni; con qualche centinaio di casi per una struttura, il training from scratch basta.

## 5. Cambio di dominio (deep ensemble del rene, test interno contro esterno)

Mann-Whitney. Risultati in `comparisons/domain_shift/` e `kidney/external/eval/`.

- **Dice** da 0.937 a 0.851. Sul test esterno il modello sottostima il volume (269 contro 330 mL) e gli errori sono soprattutto falsi negativi (72 mL per caso, contro 12 di falsi positivi). Circa un quarto dei falsi negativi lontani dal bordo cade nel seno renale (sul test interno l'1%). Sui 3 contorni automatici il Dice è 0.922.
- **Calibrazione:** la NLL a T = 1 è circa 4 volte quella interna (0.626 contro 0.150), ma il T\* stimato sul test interno funziona ancora (−49% di NLL, contro −46% sul test interno).
- **Incertezza:** il livello medio delle mappe sale (entropia ×1.8, varianza ×3.2). L'aumento è soprattutto aleatorio e cresce con lo spessore delle fette (99° percentile dell'entropia sui voxel corretti 0.07 interno, 0.12 a 2.5 mm, 0.17 a 5 mm).
- **Soglie:** le τ\* trasferite funzionano male. L'F1 nella ROI scende di 0.07-0.11 per tutte le mappe tranne l'EBCU, mentre l'AUPRC scende meno: va ricalibrata la soglia, non la temperatura.
- **Livello del caso:** il Dice medio tra membri resta correlato al Dice del caso (Spearman 0.81 interno, 0.50 esterno).

## 6. Confronto delle misure di incertezza

Risultati in `<struttura>/<modello>/ranking_stats/` e `comparisons/organs_and_domain/`.

- **ROI 15 mm:** differenze piccole. Le migliori F1 sono 0.39-0.41; nel gruppo di testa ci sono sempre l'entropia e, con il deep ensemble, SMU, anti-confidence total e DEU. L'EBCU è ultima (0.14-0.19).
- **No-border:** con il deep ensemble vincono le misure di disaccordo, mentre entropia e confidence sono tra le ultime. L'EBCU ha il recall più alto (0.49-0.55) ma la precisione più bassa (0.08-0.14).
- **Scomposizione** (rene, deep): vicino al bordo è più utile la parte aleatoria (F1 0.383 contro 0.326 della MI), lontano dal bordo la parte epistemica (0.166 contro 0.125), anche se la MI vale solo lo 0.7% dell'entropia totale.
- **Vista target** (separazione tra incertezza attesa e inattesa, AUPRC normalizzata): nessuna misura separa bene le due componenti. Interno: entropia 0.008, MI 0.005, DEU 0.007, EBCU 0.017. Esterno: 0.028 / 0.013 / 0.022 / 0.040. L'EBCU è significativamente meglio dell'entropia solo sul test esterno (79/107 casi).
- **A parità di falsi allarmi** (zone di errore oltre 4.5 mm, rene deep, `kidney/deep/froc/`): con un budget piccolo vincono le misure basate sull'entropia (DEU, entropia, expected entropy trovano circa il 40% delle zone con al massimo 1 mL di falsi allarmi per paziente, circa il 75% con 5 mL). L'EBCU funziona solo con budget larghi.
- **DEU** migliora l'entropia nel no-border in 3 ensemble su 4 (+20% rene deep, +77% rene checkpoint, +38% trachea deep).

## 7. Mappe combinate

Ogni misura divisa per la propria soglia, poi unite (switch entropia/std, massimo entropia/std, switch DEU/MI); risultati in `<struttura>/<modello>/ranking_stats_with_combined/`.

- Con il deep ensemble una sola mappa raggiunge la migliore misura di ogni regione senza superarla. Sul rene lo switch DEU/MI è l'unica mappa, tra 17, nel gruppo di testa sia nella ROI (0.386) sia nel no-border (0.166).
- La regola del massimo ha il recall più alto (rene deep 0.655 contro 0.611) e sul test esterno l'F1 migliore nella ROI (0.333 contro 0.323).
- Con il checkpoint ensemble non portano vantaggi.

## 8. Autoencoder sulle mappe di incertezza

Rene, deep ensemble; 96 casi interni (65 con errori oltre 2 mm, 29 oltre 4.5 mm) e 107 esterni; cross-fitting a 5 gruppi; configurazione scelta: 4 livelli, collo di bottiglia da 16 canali. Risultati in `kidney/autoencoder/eval/`.

| AUPRC normalizzata (mediana) | Int. standard | Int. target 2 mm | Est. standard | Est. target 2 mm |
|---|---|---|---|---|
| Entropia | 0.263 | 0.008 | 0.146 | 0.028 |
| DEU | 0.361 | 0.007 | 0.142 | 0.022 |
| EBCU | 0.000 | 0.017 | 0.017 | 0.040 |
| Eccesso, varianti a/b/c | 0.195-0.204 | 0.002 | 0.064-0.067 | 0.006-0.012 |
| Combinazione C4 (variante c) | 0.204 | 0.006 | 0.161 | 0.032 |

- **Il residuo non migliora le misure esistenti**, con nessuna variante di ingresso: l'autoencoder ricostruisce anche le zone insolite, quindi il residuo è piccolo proprio dove l'incertezza è inattesa. La mancanza è circa zero sul test interno.
- **Combinazioni con la mappa originale (C1-C4):** riproducono l'entropia. Unico guadagno significativo: C4 sul test esterno (+0.015 nella vista standard, 92/107 casi).
- **Training sui soli casi ben segmentati:** migliora l'eccesso rispetto a un controllo casuale di pari numero (interno 57/65, esterno 76/107), ma resta sotto entropia ed EBCU.
- **Analisi per zona:** tutte le mappe derivate dall'entropia, autoencoder compreso, mettono il 92-97% degli allarmi entro 2 mm dal bordo; l'EBCU solo il 16% (interno) e il 7% (esterno). Con un budget dell'1% della ROI, l'EBCU copre il 3.0% / 6.7% del volume degli errori oltre 4.5 mm, la MI l'1.2% / 3.6%, le altre mappe al massimo lo 0.9%.

## 9. Conclusioni

- Il deep ensemble è la scelta migliore: più accurato, meglio calibrato e con un comportamento delle mappe coerente tra strutture.
- Il temperature scaling va stimato per ogni modello e si trasferisce a dati esterni; le soglie delle mappe no.
- Vicino al bordo sono più utili entropia e DEU, lontano dal bordo il disaccordo tra membri; l'EBCU solo quando si accetta un budget di revisione ampio.
- Nessun metodo post-hoc, autoencoder compreso, trova gli errori su cui l'ensemble è sicuro: il limite è l'informazione contenuta nell'output dell'ensemble.

## 10. Limiti

- Test esterno su un solo ospedale e solo per il rene, con una convenzione di contornatura diversa (seno renale).
- Il checkpoint ensemble usa l'80% dei casi di training (solo il fold 0).
- Il T\* del fine-tuned è stimato su casi visti nel pretraining.
- Mappe salvate a 8 bit: numeri leggermente pessimistici, soprattutto per MI ed EBCU.
- Autoencoder: valutato solo sul rene; in un caso interno (s0536) 34 mL di reference sono fuori dal ritaglio.
- Pochi casi con errori lontani dal bordo sul test interno (29 oltre 4.5 mm).
