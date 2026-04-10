# Lessons Learned — GPU Mode Paris Hackathon 2026

**Resultat** : 3e place (podium), val_loss 4.08 sur 32 GPUs B300 en 10 min
**Meilleur run test** : val_loss 3.62 sur 32 GPUs (v7, sans LR decay)
**Meilleur run 2 GPUs** : val_loss 3.92 (v7)

---

## 1. La règle d'or : Throughput > Model Size

Sur un budget de temps court (10 min), le nombre de steps compte plus que la taille du modèle.

| Modèle | ms/step | Steps en 10 min | val_loss |
|--------|---------|-----------------|----------|
| 350M params | 450ms | 1177 | 4.28 (pire) |
| 110M params | 300ms | 1832 | 4.15 |
| 119M optimisé | 230ms | 2500 | 3.92 (meilleur) |

Un modèle 3x plus petit mais 2x plus rapide fait **2x plus de steps** et obtient une **bien meilleure loss**.

**Pourquoi** : chaque step met à jour les poids avec un batch de tokens. Plus de steps = plus de mises à jour = meilleur apprentissage. Un gros modèle a plus de capacité par step, mais si il fait 2x moins de steps, il est sous-entraîné.

**Quand un gros modèle devient rentable** : quand on a assez de GPUs pour que le throughput reste élevé (32+ GPUs). Sur 2 GPUs, 110-120M est le sweet spot.

---

## 2. Optimisations de throughput (chaque ms compte)

### gc.disable()
- Désactive le garbage collector Python pendant le training
- Élimine des pauses imprévisibles de ~100ms
- **Gain** : ~5-10% throughput, gratuit, aucun risque

### Batch size élevé + grad_accum bas
- Batch 32 avec grad_accum 2 >> Batch 16 avec grad_accum 4
- Même nombre de tokens/step, mais 2x moins de micro-steps
- Chaque micro-step a un overhead fixe (sync, kernel launch)
- **Gain** : ~15-20% throughput

### torch.compile
- Fusionne les opérations CUDA en kernels optimisés
- Première compilation lente (~30-60s sur B300), ensuite c'est gratuit
- **Gain** : ~20-40% throughput sur CUDA
- Ne fonctionne pas sur MPS (Mac)

### Data prefetch
- Charge le prochain batch sur un thread CPU pendant que le GPU calcule
- `threading.Thread` qui remplit un buffer numpy → torch tensor
- **Gain** : ~5-10%, élimine le temps d'attente I/O

### ReLU² au lieu de SwiGLU
- SwiGLU : 3 projections linéaires (up, gate, down)
- ReLU² : 2 projections linéaires (up, down), relu(x)²
- Même nombre de paramètres, 1 matmul en moins par couche
- **Gain** : ~7% throughput (220ms vs 237ms/step)

### Muon : 3 Newton-Schulz iters (pas 5)
- L'orthogonalisation converge déjà bien à 3 itérations
- 5 iters = 40% de compute en plus pour l'optimizer, gain marginal
- **Gain** : ~5% throughput

---

## 3. Architecture (ce qui marche sur 119M params)

### RMSNorm (au lieu de LayerNorm)
- `norm = rsqrt(mean(x²) + eps)` — pas de soustraction de la moyenne
- Plus rapide, aussi stable. Standard LLaMA/Mistral/Gemma.

### RoPE (Rotary Position Embeddings)
- Encode la position dans Q et K via rotation complexe
- Remplace les embeddings positionnels appris (économise des params)
- Meilleure généralisation à des longueurs différentes

### QK Norm
- `F.rms_norm` sur Q et K avant le dot product attention
- Empêche les gradients d'exploser quand les produits scalaires deviennent grands
- Particulièrement utile avec Muon qui peut faire de gros updates

### Logit soft-capping (Gemma-style)
- `30 * tanh(logits / 30)` avant le softmax
- Empêche les logits de dépasser ±30
- Stabilise la cross-entropy loss, surtout en début de training

### Norm après embedding
- `F.rms_norm(x, (x.size(-1),))` juste après le token embedding
- Normalise l'entrée de la première couche
- Aide la stabilité, surtout avec des embeddings partagées (weight tying)

### Value Embeddings (ResFormer)
- `v = v + val_embed[:T]` — biais positionnel appris sur les V
- Enrichit les représentations sans coût de compute significatif
- +9.4M params mais gain en qualité de représentation

### x0 Residual Connection
- Chaque block reçoit un skip depuis l'embedding initial : `x = x + lambda_x0 * x0`
- `lambda_x0` initialisé à 0, le modèle apprend à l'utiliser si c'est utile
- Aide le gradient à remonter jusqu'aux embeddings

### Per-layer Learnable Lambdas
- `x = x + lambda_attn * attn(x) + lambda_mlp * mlp(x)`
- Chaque couche apprend son propre scaling pour attn et mlp
- Plus flexible que des résidus fixes (×1)

### U-Net Skip Connections
- Connections symétriques : couche 0 → couche 11, couche 1 → couche 10, etc.
- Scalaires apprenables initialisés à 0
- Raccourcit le chemin des gradients dans les réseaux profonds
- Inspiré de l'architecture U-Net (segmentation d'images)

### Zero-init Output Projections
- `nn.init.zeros_(block.attn.c_proj.weight)` et `nn.init.zeros_(block.mlp.w_down.weight)`
- Le modèle à l'initialisation est quasi-identity (via le résiduel)
- Convergence plus stable au début, similaire à muP (maximal update parameterization)

### Weight Tying
- `wte.weight = lm_head.weight` — embedding et output projection partagent les mêmes poids
- Économise ~25M params (vocab_size × n_embd = 32768 × 768)
- Standard dans les modèles modernes

---

## 4. Le WSD Schedule (Warmup-Stable-Decay) — CRITIQUE

Le schedule de learning rate est le facteur le plus important après le throughput.

### Comment ça marche
1. **Warmup** (100 steps) : LR monte de 0 à max_lr
2. **Stable** : LR reste au max pendant la majorité du training
3. **Decay** : LR descend linéairement vers min_lr

### Pourquoi c'est mieux que cosine
- Cosine commence à décroître immédiatement après le warmup
- WSD garde le LR max plus longtemps → exploration plus agressive
- Le decay final est la phase de "convergence" où le modèle affine ses poids

### Le piège fatal : max_steps trop haut
Si `max_steps` est plus grand que le nombre de steps atteignable en 10 min, le decay ne s'active jamais.

| max_steps | Decay commence | Steps atteints | Decay ? | val_loss |
|-----------|---------------|----------------|---------|----------|
| 2000 | step 1600 | 2000 | OUI, 400 steps | 4.01 |
| 2500 | step 2000 | 2500 | OUI, 500 steps | 3.92 |
| 2700 | step 2160 | 1823 | NON (trop tôt) | 4.10 |
| 4000 | step 3200 | 2960 | NON | 3.62* |

*3.62 grâce aux 32 GPUs (16x plus de tokens/step) mais le decay aurait amélioré encore.

### Leçon apprise : schedule basé sur le TEMPS, pas les steps
On ne sait pas à l'avance combien de steps on fera (ça dépend du throughput, du nombre de GPUs, de torch.compile...). La solution robuste :

```python
time_frac = elapsed / time_limit_seconds
if time_frac < (1.0 - cooldown_frac):
    return max_lr  # Phase stable
else:
    progress = (time_frac - (1 - cooldown_frac)) / cooldown_frac
    return min_lr + (1 - progress) * (max_lr - min_lr)  # Decay
```

Avec `cooldown_frac=0.50` et `time_limit=10 min` :
- 0-5 min : LR stable au max
- 5-10 min : LR décroît linéairement vers min_lr (15% du max)

**Ceci est le fix qu'on aurait dû faire dès le départ.** On l'a implémenté trop tard (v11) et n'a pas pu le tester.

---

## 5. Muon Optimizer

### Concept
- Pour les poids 2D (matrices) : orthogonalise le gradient avec Newton-Schulz avant la mise à jour
- Pour le reste (embeddings, norms, biases) : AdamW classique
- La direction du gradient est "nettoyée" pour être plus efficace

### Paramètres
- `lr=0.02` (10x plus élevé que AdamW — normal car l'update est normalisé)
- `momentum=0.95`
- 3 itérations de Newton-Schulz (suffisant, pas besoin de 5)

### Avantage
- Converge plus vite au début (crucial pour 10 min)
- Le WSD decay sur Muon donne un drop massif en fin de training

### Routing des paramètres
```python
for name, p in model.named_parameters():
    if p.dim() < 2:         # 1D → AdamW (norms, biases)
    elif "wte" or "lm_head": # Embeddings → AdamW (pas orthogonalisable)
    else:                    # 2D matrices → Muon
```

---

## 6. Scaling Multi-GPU

### DDP (Distributed Data Parallel)
- Chaque GPU a une copie du modèle
- Chaque GPU traite un batch différent
- Les gradients sont moyennés via AllReduce (NCCL)
- Effective batch = batch_per_gpu × grad_accum × world_size

### grad_accum_steps
- Sur 2 GPUs : accum=2 → effective batch = 32×2×2 = 128
- Sur 32 GPUs : accum=1 → effective batch = 32×1×32 = 1024
- Règle auto : `max(1, 8 // world_size)`

### LR scaling avec le batch
- Quand le batch est Nx plus grand, le gradient est Nx plus "lisse" (moins bruité)
- On peut augmenter le LR par √N (square root scaling rule)
- Ou utiliser le batch scheduling (ramp progressif) pour compenser

### SLURM (cluster management)
- `sbatch submit.sh` : soumet un job à la queue
- `squeue -u ricy` : voir ses jobs
- `scancel <job_id>` : annuler un job
- `sinfo` : voir l'état des nodes
- `srun --nodelist=<node> nvidia-smi` : vérifier un node

### Pièges multi-node
- `--time` dans `#SBATCH` peut être rejeté par QOS → ne pas le mettre si la politique du cluster l'interdit
- Pas besoin de `.venv` si PyTorch est installé system-wide (vérifier avec `which python3 && python3 -c "import torch"`)
- Port 29500 peut être occupé → utiliser `--master_port=29501`
- Les logs vont dans le répertoire courant si `logs/` n'existe pas

---

## 7. Ce qui n'a PAS marché

### Modèle trop gros (350M sur 2 GPUs)
- 450ms/step vs 230ms → moitié moins de steps
- Le modèle est "plus intelligent" par step mais trop lent
- val_loss 4.28 vs 3.92 pour le modèle 3x plus petit
- **Leçon** : calibrer la taille du modèle au throughput disponible

### FP8 sur petit modèle (120M)
- 60% plus lent ! 350ms vs 230ms
- L'overhead (scaling factors, conversions FP32↔FP8) dépasse le gain matmul
- Les matmuls 768×3072 sont trop petits pour amortir l'overhead FP8
- **Leçon** : FP8 uniquement pour modèles >500M-1B+

### max_steps=2700 (WSD decay raté)
- Le decay commence à step 2160 (80% de 2700)
- Mais seulement 1823 steps en 10 min → decay jamais activé
- Régression 3.92 → 4.10
- **Leçon** : toujours s'assurer que le decay s'active. Utiliser un schedule basé sur le TEMPS.

### Soumettre du code non testé
- Le v11 avec time-based schedule n'a pas été testé sur 32 GPUs avant la soumission finale
- Le score final (4.08) est pire que le test v7 (3.62)
- **Leçon** : ne soumettre que du code testé et prouvé. Le v7 brut aurait peut-être fait 3.62 si on l'avait soumis tel quel.

---

## 8. Workflow idéal pour un hackathon GPU

### Phase 1 : Itérations rapides sur 2 GPUs (premières heures)
1. Partir du baseline, vérifier que ça tourne
2. Empiler les optimisations d'architecture une par une
3. Mesurer la val_loss (pas juste la train loss)
4. Optimiser le throughput (ms/step) en parallèle
5. Trouver le max_steps optimal (WSD decay doit s'activer)

### Phase 2 : Test sur 32 GPUs (dernière heure)
1. Adapter grad_accum et max_steps au nombre de GPUs
2. Faire un run complet de 10 min
3. Vérifier que la val_loss est meilleure qu'en 2 GPUs
4. Si possible, tester 2-3 configurations

### Phase 3 : Soumission finale (30 min avant la deadline)
1. **Ne soumettre QUE du code prouvé** — pas d'expérimentation de dernière minute
2. Vérifier les défauts de train.py (data_dir, checkpoint_path, max_steps)
3. Tester les défauts : `python train.py` sans arguments → est-ce que ça marche ?
4. S'assurer que le schedule (LR decay) est robuste aux arguments des organisateurs

### Règle de soumission
- Les organisateurs lancent `train.py` avec LEURS arguments
- On ne contrôle pas `--max_steps`, `--batch_size`, etc.
- Les **défauts** dans train.py doivent être raisonnables
- Le schedule devrait être basé sur le **temps** (time_limit), pas les steps

---

## 9. Pense-bête technique

### Checkpoint contract
```python
# train.py doit sauvegarder :
torch.save({
    "step": step,
    "model": raw_model.state_dict(),
    "config": asdict(cfg),
}, "checkpoint.pt")

# model.py doit exposer :
def get_model(config: dict) -> nn.Module
# forward(idx, targets=None) -> (logits, loss)
```

### BFloat16
- `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`
- Suffisant pour les modèles <1B params
- Plus rapide que FP32, pas de scaling issues comme FP16

### Commandes cluster utiles
```bash
# Lancer un job
sbatch submit.sh

# Voir la queue
squeue -u <user>

# Annuler
scancel <job_id>

# Voir les logs en temps réel
tail -f slurm-<job_id>.out

# Copier des fichiers
scp -i key local_file user@host:remote_path

# Vérifier PyTorch
python3 -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

---

## 10. Pour la prochaine fois

1. **Implémenter le schedule time-based dès le départ** — c'est le fix le plus important
2. **Tester sur 32 GPUs AVANT la soumission** — minimum 2 runs complets
3. **Ne pas toucher au code dans les 30 dernières minutes** — soumettre le meilleur code testé
4. **Préparer un script de soumission automatique** — qui copie les bons fichiers au bon endroit
5. **Logger les défauts de train.py** — imprimer tous les hyperparams au démarrage pour vérifier
6. **Essayer un modèle plus gros sur 32 GPUs** — 250-400M pourrait être le sweet spot
7. **Tester les Triton kernels** — ReLU² + linear fusionnés, comme modded-nanogpt
8. **Batch scheduling + sequence length scheduling** — commencer court (512 tokens) puis allonger (1024)
9. **GQA (Grouped Query Attention)** — moins de params K/V, plus rapide, surtout avec de longs contextes
10. **Comprendre les arguments des organisateurs** — demander avant la deadline quel submit.sh ils utilisent
