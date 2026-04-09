# Carnet d'expériences — Hackathon GPU Mode Paris 2026

**Équipe** : ricy (Edouard + Boris + 2 coéquipiers)
**Track** : 1 — LLM Pre-training sur B300
**Objectif** : meilleure validation loss en 10 min sur 32 GPUs B300
**Métrique** : validation loss (perplexity, lower is better). Tiebreaker: HellaSwag accuracy.

---

## Exp 0 — Baseline local (fake data, Mac MPS)
- **Date** : 2026-04-09 ~11h00
- **Hypothèse** : vérifier que le starter kit fonctionne
- **Config** : GPT-2 original, 4 layers, 256 dim, 4 heads, 11.8M params
- **Hardware** : Mac Apple Silicon (MPS), 1 device
- **Données** : 1M tokens fake (Zipf distribution)
- **Résultat** : loss 9.99 → 9.02 en 50 steps, ~340ms/step
- **Conclusion** : baseline fonctionnel, contrat checkpoint OK

## Exp 1 — SwiGLU (local, fake data)
- **Date** : 2026-04-09 ~11h30
- **Hypothèse** : SwiGLU (LLaMA-style) donne meilleure loss/FLOP que GELU
- **Changement** : GELU MLP → SwiGLU MLP dans model.py
- **Config** : même que Exp 0, 11.9M params
- **Résultat** : loss 10.08 → 9.04 en 50 steps
- **Conclusion** : similaire sur 50 steps fake data, mais standard moderne, on garde

## Exp 2 — + RMSNorm (local, fake data)
- **Date** : 2026-04-09 ~11h35
- **Hypothèse** : RMSNorm est plus rapide que LayerNorm
- **Changement** : LayerNorm → RMSNorm
- **Résultat** : loss 10.10 → 9.09, ~390ms/step
- **Conclusion** : légèrement plus lent sur MPS (cast float32), sera plus rapide sur CUDA

## Exp 3 — + RoPE (local, fake data)
- **Date** : 2026-04-09 ~11h40
- **Hypothèse** : RoPE encode mieux les positions que les embeddings appris
- **Changement** : suppression wpe, ajout RoPE dans attention
- **Config** : 11.6M params (-0.3M, wpe supprimé)
- **Résultat** : loss 10.08 → 9.07, ~420ms/step
- **Conclusion** : moins de params, standard moderne

## Exp 4 — + torch.compile (local, fake data)
- **Date** : 2026-04-09 ~11h45
- **Hypothèse** : torch.compile fusionne les ops pour ~20-40% speedup
- **Changement** : ajout torch.compile dans train.py (CUDA only)
- **Résultat** : non actif sur MPS, sera actif sur CUDA
- **Conclusion** : à valider sur le cluster

## Exp 5 — + Muon optimizer (local, fake data)
- **Date** : 2026-04-09 ~11h50
- **Hypothèse** : Muon (Newton-Schulz) converge plus vite sur les poids matriciels
- **Changement** : AdamW seul → Muon (2D weights) + AdamW (embeddings/norms)
- **Config** : Muon lr=0.02, momentum=0.95
- **Résultat** : loss 9.85 → 9.03 en 50 steps. Step 10: 9.85 vs 9.99 baseline
- **Conclusion** : convergence plus rapide au début ! Exactement ce qu'on veut pour 10 min

## Exp 6 — Toutes améliorations sur cluster B300 (vraies données)
- **Date** : 2026-04-09 12:35
- **Hypothèse** : nos améliorations + vraies données + CUDA = bonne loss
- **Config** : 12 layers, 12 heads, 768 dim, 110.1M params, batch 16, grad_accum 4
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Données** : 49 shards, 49B tokens, /home/data/
- **Flags** : torch.compile activé, Muon + AdamW, DDP sur 2 GPUs
- **Résultats partiels** :
  - step 10: loss 9.59 (~320ms/step)
  - step 20: loss 9.02
  - step 30: loss 8.45
  - step 40: loss 8.13
  - step 50: loss 7.86
- **Résultat final** :
  - 1832 steps en 10.0 min (~300ms/step moyen)
  - loss finale: **~4.15** (oscillant entre 4.1 et 4.3)
  - checkpoint sauvé: checkpoint_ed.pt
- **Conclusion** : excellent premier run ! Loss 9.59 → 4.15, torch.compile actif, Muon+AdamW convergent bien

## Exp 7 — v2 Quick Wins Karpathy (cluster B300)
- **Date** : 2026-04-09 ~13h30
- **Hypothèse** : 5 quick wins ajoutent de la stabilité et du speed sans changer l'archi
- **Changements** :
  1. **QK Norm** : `F.rms_norm` sur Q et K avant attention → stabilise les gradients
  2. **Logit soft-capping** : `30 * tanh(logits/30)` → empêche l'explosion des logits (à la Gemma)
  3. **Norm après embedding** : `F.rms_norm` sur les embeddings → stabilise le début du réseau
  4. **GC disable** : `gc.disable()` pendant le training → élimine les pauses ~100ms du ramasse-miettes
  5. **WSD schedule** : Warmup → Stable (80%) → Linear Decay (20%) → garde le LR max plus longtemps
- **Config** : 12 layers, 12 heads, 768 dim, 110.1M params, batch 16, grad_accum 4
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultat final** :
  - 1872 steps en 10.0 min (~250ms/step — 17% plus rapide que v1 !)
  - loss finale: **~4.07** (oscillant entre 4.03 et 4.15)
  - checkpoint sauvé: checkpoint_v2.pt
  - LR stable à 6e-4 tout le run (WSD decay kick-in à step 8000, on n'y arrive pas)
- **Conclusion** : amélioration vs v1 (4.15 → 4.07), plus rapide grâce au GC disable. QK norm stabilise bien.

## Exp 8 — v3 Architecture avancée (cluster B300)
- **Date** : 2026-04-09 ~14h00
- **Hypothèse** : un modèle plus gros + innovations architecturales = meilleure loss
- **Changements par rapport à v2** :
  1. **Modèle plus gros** : 768d/12L/12H (110M) → 1024d/16L/16H (~350M params)
  2. **Value Embeddings (ResFormer)** : embeddings apprises ajoutées aux V dans l'attention — enrichit les représentations
  3. **x0 Residual Connection** : chaque block reçoit un skip depuis l'embedding initial x0 (learnable lambda, init=0)
  4. **Per-layer Residual Lambdas** : scalaires apprenables par couche pour attn et mlp (init=1)
  5. **LR réduit** : 4e-4 (au lieu de 6e-4) pour stabiliser le training du modèle plus gros
  6. **Dropout supprimé** (plus besoin, embedding drop aussi supprimé)
- **Config** : 16 layers, 16 heads, 1024 dim, ~350M params, batch 16, grad_accum 4
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultats partiels** :
  - step 10: loss 10.44 (~487ms/step), warmup en cours
  - step 50: loss 7.54
  - step 90: loss 6.86 (fin du warmup)
  - Estimation: ~1250 steps en 10 min, decay WSD ~step 1200
- **Résultat final** :
  - 1177 steps en 10.0 min (~450ms/step)
  - loss finale: **~4.28** (oscillant entre 4.14 et 4.43)
  - LR resté à 4e-4 tout le run (decay à step 1200, atteint seulement 1177 — raté de peu !)
  - checkpoint sauvé: checkpoint_v3.pt
- **Conclusion** : **RÉGRESSION** vs v2 (4.07) et même v1 (4.15). Le modèle plus gros (253M) n'a pas assez de steps pour converger sur 2 GPUs. Moins de steps = moins de mise à jour des poids = sous-entraîné. La taille de modèle est un trade-off : plus de capacité mais moins de steps en 10 min.
- **Leçon clé** : sur 2 GPUs, ~110M params (v2) est le sweet spot. Un modèle plus gros ne sera rentable que sur 32 GPUs où chaque step voit 16x plus de données.

## Exp 9 — v4s Fused CE + Val Eval + 110M optimisé (cluster B300)
- **Date** : 2026-04-09 ~15h00
- **Hypothèse** : rester à 110M (sweet spot 2 GPUs) + fused CE pour accélérer + val loss pour mesurer proprement
- **Changements par rapport à v2** :
  1. **Fused Cross-Entropy** : chunked linear+CE, ne matérialise jamais le tenseur (B*T, 32768)
  2. **Validation eval** : dernier shard réservé pour val, évalué toutes les 200 steps (10 batches)
  3. **avg50** : moyenne mobile des 50 derniers steps pour comparaison stable
  4. **max_steps=2000** : WSD decay à step 1600 (vs 10000 avant — le decay va enfin s'activer !)
  5. **Eval finale** : affiche val_loss à la fin du run
- **Config** : 12 layers, 12 heads, 768 dim, 110M params, batch 16, grad_accum 4
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultat final** :
  - 1669 steps en 10.0 min (~315ms/step)
  - Best val_loss: **4.1689** ★ (step 1600)
  - Final val_loss: 4.1991
  - WSD decay activé à step 1600 (80% de 2000) → LR 6e-4 → 5.19e-4 en fin de run
  - avg50 finale: ~4.18
- **Conclusion** : **MEILLEUR RÉSULTAT MESURÉ !** val_loss 4.1689 — première mesure fiable (les exp précédentes ne rapportaient que la train loss, toujours plus optimiste). Le WSD decay s'active enfin et aide. Train avg50 ~4.18, comparable à v1 (train ~4.15) mais la val loss est le vrai indicateur. Le modèle 110M reste le sweet spot sur 2 GPUs.

## Exp 10 — v5 Architecture enrichie + throughput max (cluster B300)
- **Date** : 2026-04-09 ~15h30
- **Hypothèse** : combiner le meilleur de v2 (taille) et v3 (archi) + optimiser le throughput
- **Changements par rapport à v4s** :
  1. **Value Embeddings** : +9.4M params, embeddings apprises ajoutées aux V dans l'attention
  2. **x0 Residual** : skip connection depuis l'embedding initial (lambda init=0)
  3. **Per-layer Lambdas** : attn/mlp scaling apprenables par couche
  4. **Batch 32, grad_accum 2** : même tokens/step (65K), mais 2x moins de micro-steps → step plus rapide
  5. **Muon 3 Newton-Schulz iters** (au lieu de 5) → optimizer ~20% plus rapide
  6. **WSD pour Muon** : Muon LR 0.02 → 0.002 en fin de run (au lieu de fixe)
  7. **Data prefetch** : charge le prochain batch sur CPU thread pendant que le GPU calcule
- **Config** : 12 layers, 12 heads, 768 dim, **119.6M params** (+8% vs v2), batch 32, grad_accum 2
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultat final** :
  - 1964 steps en 10.0 min (~237ms/step — 25% plus rapide que v4s !)
  - Best val_loss (eval): **4.1410** (step 1800)
  - Final val_loss: **4.0939** ★ (**MEILLEUR SCORE GLOBAL**)
  - Train avg50 finale: ~4.04
  - WSD decay actif à step 1600: Adam LR 6e-4 → 1.15e-4, Muon LR 0.02 → 0.004
- **Conclusion** : **EXCELLENT !** La combinaison throughput (batch 32/accum 2 + Muon 3 iters + prefetch) donne 300 steps de plus que v4s. Le WSD sur Muon provoque un drop spectaculaire pendant le decay (val 4.14 → 4.09). Les features architecturales (val_embed, x0, lambdas) + le throughput = le sweet spot.
- **Leçon clé** : optimiser le throughput (ms/step) est aussi important que l'architecture. Plus de steps = plus de tokens vus = meilleure loss.

## Exp 11 — v6 ReLU² + U-Net Skips + Zero-Init (cluster B300)
- **Date** : 2026-04-09 ~15h30
- **Hypothèse** : ReLU² est plus rapide que SwiGLU (2 matmuls vs 3), U-Net skips raccourcissent le chemin des gradients, zero-init stabilise la convergence
- **Changements par rapport à v5** :
  1. **ReLU² MLP** : remplace SwiGLU (3 projections → 2 projections, hidden=4*768=3072, même # params)
     → 1 matmul en moins par couche par step → steps plus rapides
  2. **U-Net Skip Connections** : connections symétriques layer 0→11, 1→10, ..., 5→6
     → Scalaires apprenables (init=0), s'activent seulement si utile
     → Raccourcit le chemin des gradients, aide l'apprentissage profond
  3. **Zero-init c_proj & w_down** : output projections initialisées à 0 (muP-like)
     → Le modèle initial est quasi-identity via résiduel → convergence plus stable au début
  4. Garde toutes les améliorations v5 : Value Embeddings, x0 Residual, Per-layer Lambdas, Muon WSD, prefetch, val eval
- **Config** : 12 layers, 12 heads, 768 dim, ~119M params, batch 32, grad_accum 2
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultat final** :
  - 2000 steps en 9.1 min (~220ms/step — encore 7% plus rapide que v5 !)
  - Best val_loss (eval): **4.0168** (step 2000)
  - Final val_loss: **4.0127** ★ (**NOUVEAU RECORD !**)
  - Train avg50 finale: **4.0000**
  - max_steps=2000 atteint à 9.1 min → **0.9 min de GPU inutilisé !**
  - WSD decay actif step 1600→2000: val 4.18 → 4.01 (drop massif)
- **Conclusion** : **EXCELLENT !** Les 3 techniques marchent toutes. ReLU² accélère les steps (220ms vs 237ms). Zero-init stabilise la convergence. U-Net skips + decay = drop spectaculaire en fin de run. Le run finit AVANT les 10 min → on peut augmenter max_steps.
- **Leçon clé** : on laisse 0.9 min sur la table. Augmenter max_steps à 2500 pour utiliser tout le temps.

---

## Exp 12 — v7 max_steps=2500 (cluster B300)
- **Date** : 2026-04-09 ~16h00
- **Hypothèse** : v6 finit à 9.1 min (max_steps=2000) → 0.9 min gaspillé. max_steps=2500 utilise tout le temps.
- **Changement** : uniquement `max_steps=2500` (vs 2000). WSD decay à step 2000 (80% de 2500).
- **Config** : identique v6 sauf max_steps
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultat final** :
  - 2500 steps en 9.6 min (~230ms/step)
  - Best val_loss (eval): **3.9361** (step 2400)
  - Final val_loss: **3.9199** ★ (**NOUVEAU RECORD — sous la barre des 4.0 !**)
  - Train avg50 finale: **3.9339**
  - Encore 0.4 min de marge → on pourrait monter à max_steps=2700
  - WSD decay step 2000→2500: val 4.03 → 3.92
- **Conclusion** : **Gain gratuit !** Juste +500 steps → -0.09 val_loss (4.01 → 3.92). Plus de steps au LR max + decay plus tard = meilleure convergence. Reste 0.4 min de marge.

---

## Exp 13 — v8 max_steps=2700 (cluster B300)
- **Date** : 2026-04-09 ~16h15
- **Hypothèse** : v7 finit à 9.6 min → 0.4 min de marge → max_steps=2700
- **Changement** : uniquement `max_steps=2700`. WSD decay à step 2160.
- **Config** : identique v7 sauf max_steps
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultat final** :
  - 1823 steps en 10.0 min (~210ms/step — plus lent que v7, probablement état GPU résiduel du Ctrl+C)
  - Best val_loss: **4.1446** (step 1800)
  - Final val_loss: **4.0989**
  - **Le WSD decay n'a JAMAIS démarré** : decay à step 2160 (80% de 2700), mais le run s'est arrêté à step 1823
  - LR constant à 6e-04 / μlr 0.020 pendant tout le run
- **Conclusion** : **RÉGRESSION vs v7 (3.92 → 4.10).** max_steps=2700 est trop haut — le decay ne s'active pas dans le temps imparti. Le modèle ne bénéficie pas de la phase de convergence. **max_steps=2500 est le sweet spot.**
- **Leçon clé** : ne jamais mettre max_steps au-delà de ce qu'on peut atteindre. Le WSD decay est critique et doit s'activer.

## Exp 14 — v9 FP8 training via torchao (cluster B300)
- **Date** : 2026-04-09
- **Hypothèse** : FP8 natif Blackwell → ~1.3-2x throughput sur matmuls → plus de steps en 10 min
- **Changement** : `torchao.float8.convert_to_float8_training(model)` avant torch.compile
  - Convertit tous les nn.Linear en FP8 automatiquement
  - Fallback gracieux vers BF16 si torchao absent
  - Nécessite `torchao` dans requirements.txt
- **Config** : identique v7 (max_steps=2500) + FP8
- **Hardware** : 2x B300 GPUs (devices 6,7), node verda-hackathon-2
- **Résultat partiel** (tué par SIGTERM — autre job a pris le node) :
  - 170 steps en ~1.7 min avant kill
  - **350-380ms/step** — **60% plus lent que v7 (230ms) !**
  - FP8 overhead (scaling, conversion) > gain matmul pour un modèle 120M
- **Conclusion** : **FP8 = RÉGRESSION de throughput sur petit modèle.** L'overhead FP8 (calcul des scaling factors, conversions) est trop coûteux vs le gain sur des matmuls de taille 768×3072. FP8 ne vaut le coup que pour des modèles 1B+ où les matmuls dominent le temps de compute.
- **Leçon clé** : ne pas utiliser FP8 sur un modèle <500M params. Rester en BF16.

## Exp 15 — v10 Scaling 8 GPUs (node verda dédié)
- **Date** : 2026-04-09
- **Hypothèse** : 8 GPUs → grad_accum=1 (plus besoin d'accumuler) → ~1.5-2x steps/s → plus de steps + effective batch 256 (vs 128)
- **Changements vs v7** :
  - `grad_accum_steps` : 2 → **1** (8 GPUs compensent)
  - `max_steps` : 2500 → **3500** (steps plus rapides → on peut en faire plus)
  - `max_lr` : 6e-4 → **8.5e-4** (linear scaling rule: batch 2x → LR √2 ≈ 1.4x)
  - `muon_max_lr` : 0.02 → **0.028** (idem)
  - `warmup_steps` : 100 → **150** (plus de steps total → warmup proportionnel)
  - WSD decay à step 2800 (80% de 3500)
- **Config** : 8x B300, batch_size=32/GPU, effective batch=256 seqs (262k tokens)
- **Hardware** : 8x B300 GPUs (node verda dédié)
- **Résultat** : [PAS ENCORE LANCÉ]
- **Conclusion** : [à compléter]

## Exp 16 — v11 modded-nanogpt improvements (2/8/32 GPUs)
- **Date** : 2026-04-09
- **Hypothèse** : modded-nanogpt (record GPT-2 124M sur 8x H100) utilise 50% cooldown + batch scheduling. Notre cooldown de 20% rate de la convergence.
- **Changements vs v7** (inspirés de https://github.com/KellerJordan/modded-nanogpt) :
  1. **Cooldown 50%** (vs 20%) : commence le decay à 50% des steps. Plus de temps en decay = meilleure convergence.
     - v7: 80% stable + 20% decay → cooldown_start = step 2000/2500
     - v11: 50% stable + 50% decay → cooldown_start = step 1250/2500
     - modded-nanogpt utilise 60% cooldown ! On prend 50% comme compromis.
  2. **min_lr = 15% du peak** (vs 10%) — modded-nanogpt utilise 15%.
  3. **Batch scheduling** : batch_size ramp de batch//2 → batch sur les premiers 40% des steps.
     - Petit batch au début = gradients bruités = meilleure exploration
     - Gros batch ensuite = gradients stables = convergence
     - LR adapté au batch : scale ×√(cur_bs/max_bs)
  4. **Auto grad_accum** : `8 // world_size` (2 GPUs → accum=4, 8 GPUs → accum=1, 32 GPUs → accum=1)
     - Keeps effective batch constant ~256 seqs quelle que soit le nombre de GPUs
  5. **Script universel** : mêmes train.py + model.py pour 2, 8, 32 GPUs (tout en CLI args)
- **Config 2 GPUs** : `--grad_accum_steps -1 --max_lr 6e-4 --muon_max_lr 0.02 --cooldown_frac 0.50 --max_steps 2500`
- **Config 32 GPUs (submit.sh)** : `--grad_accum_steps -1 --max_lr 6e-4 --muon_max_lr 0.02 --cooldown_frac 0.50 --max_steps 4000`
- **Hardware** : any (2/8/32 GPUs)
- **Résultat** : [PAS ENCORE LANCÉ]
- **Conclusion** : [à compléter]

---

## Idées à tester (backlog — v12+)
- [ ] Untie embeddings à 2/3 du training
- [ ] Multi-token prediction (prédit 2-3 tokens à la fois)
- [ ] Token-Routed MoE (de Boris) — capacité massive à FLOPs constants
- [ ] Mu-Guidance (de Boris) — scaling rules pour les hyperparams
- [ ] Sliding window attention (contexte plus long sans O(T^2))
- [ ] Modèle 1B+ pour le run final 32 GPUs
- [ ] GQA (Grouped Query Attention) — moins de params KV, plus rapide
- [ ] Fused Triton kernels (ReLU² + linear fusionnés, comme modded-nanogpt)
- [ ] Sequence length scheduling (commencer court, allonger)
