# Domain-Informed Matched-Filter CNN v2

This directory is a clean experimental implementation of the dissertation's
matched-filter hypothesis. It does not modify the legacy scripts.

## Implemented

- deterministic stratified train/validation splits;
- one- and two-layer matched-filter CNNs with raw logits;
- a conventional pooled Standard CNN with a linear classification head;
- Random, Kaiming, Gabor, PCA, class-conditional K-means, diagonal DI-WMF
  and low-rank DI-WMF initialisation;
- greedy layer-wise initialisation for the two-layer model;
- portable class-conditional multi-prototype dictionaries for layer-wise
  Standard CNN initialisation;
- a shared sparse matched-concept bottleneck with an exact per-concept logit
  decomposition and an L1-regularised classification head;
- optional proximal concept-head shrinkage that creates exact zero
  class--concept connections;
- a validation-only Adaptive DI-WMF pilot over K-means, diagonal and low-rank
  candidates, with a complete selection audit stored in each run config;
- a compact residual architecture and canonical torchvision ResNet18 for
  progressively stronger cross-architecture tests;
- a tied-weight Correncoder whose classification encoder is jointly trained
  with a centred reconstruction-correlation objective;
- the published task-matched six-layer 1D PPG-to-respiration Correncoder,
  CapnoBase pretraining, and 53-fold BIDMC LOSO evaluation;
- fold-training-only layerwise 1D matched-filter initialisation, leakage-free
  affine Epoch-0 calibration, and fixed-overlap lag-tolerant Pearson loss;
- a C4 rotation-invariant CNN that evaluates shared matched templates at four
  quarter-turn orientations without adding trainable parameters;
- unified initialisation-plus-training time and NVML energy-to-fixed-accuracy
  analysis with right-censored threshold failures;
- half-width and quarter-width Standard CNNs for real parameter-count and
  checkpoint-size efficiency tests;
- training-only global logit-scale calibration applied equally to every method;
- fixed or linearly decaying cosine anchor regularisation;
- correct per-epoch losses and a validation-only model-selection loop;
- optional cosine learning-rate decay for 50--100 epoch convergence studies;
- epoch-0 accuracy, explicitly separated AULC 0:T and post-update AULC 1:T,
  time-to-95%, kernel drift, class-group evidence gap and per-channel
  best-class selectivity, and
  template-assignment stability;
- deterministic Gaussian, correlated Gaussian, salt-and-pepper, occlusion,
  rotation, translation, brightness and contrast corruptions;
- self-contained run configuration, checkpoints and CSV/JSON results.
- post-training global magnitude-pruning evaluation, quantitative template
  semantics, activation-grid visualisation, and CAM/Grad-CAM/Integrated-
  Gradients deletion comparisons.
- optional training-only recovery of physically channel-pruned models.

DI-WMF estimates pooled within-class variance from the training split and
constructs shrinkage-stabilised discriminants. The diagonal variant models
independent coordinates; the low-rank variant adds the leading correlated
covariance directions and applies the inverse with the Woodbury identity. No
validation or test image is used to initialise the model.

## Quick run

From this directory, using the existing `CL2` Conda environment:

```powershell
conda run -n CL2 python run_experiment.py --dataset mnist --model 2layer `
  --init di_wmf --epochs 5 --train-fraction 0.1
```

Regularised run:

```powershell
conda run -n CL2 python run_experiment.py --dataset mnist --model 2layer `
  --init di_wmf --epochs 20 --reg-schedule linear --reg-lambda 0.1
```

Experiment matrix:

```powershell
conda run -n CL2 python run_matrix.py --datasets mnist fashion `
  --methods kaiming classmean di_wmf --seeds 0 1 2 3 4 --epochs 20 `
  --with-regularised-wmf
```

Standard CNN cross-architecture matrix on CUDA:

```powershell
conda run -n CL2 python run_matrix.py --datasets fashion cifar10 `
  --model standard --init-layers 3 --methods random kaiming kmeans di_wmf `
  --seeds 0 1 2 3 4 --epochs 20 --train-fraction 0.1 --device cuda
```

For `--model standard`, layer-wise DI-WMF uses four local prototypes per class
in the stem, eight activation-space prototypes per class in the second
convolution, and a diagonal matched discriminant in the pooled feature space.
Only the training split is used. Initialisation time is reported separately
from gradient-training time.

Gaussian-noise robustness evaluation of clean-trained checkpoints:

```powershell
conda run -n CL2 python evaluate_corruptions.py --datasets fashion sign `
  --models standard --methods random di_wmf --seeds 0 1 2 3 4 `
  --severities 0.05 0.1 0.2 0.3 --noise-seeds 0 1 2 --device cuda
```

Noise is injected in the common `[0, 1]` pixel domain. CIFAR-10 is
de-normalised before corruption and re-normalised afterwards. Noise tensors
are reset from the same seed for each model so paired methods receive
identical corruptions.

## Frontier studies

Experiment A is a controlled stem-only comparison. Every method uses a
Kaiming-initialised second convolution and classifier; only the first
convolutional bank changes.

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion --model standard --init-layers 1 `
  --methods random_stem kaiming gabor pca kmeans di_wmf `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --device cuda --output-root outputs_frontier_a
```

The completed cross-task extension adds rank-16 low-rank DI-WMF and applies
the identical stem-only protocol to all three image tasks. Sign is extended to
ten paired seeds, producing 140 runs in total:

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion cifar10 sign --model standard --init-layers 1 `
  --methods random_stem kaiming gabor pca kmeans di_wmf `
  --seeds 0 1 2 3 4 --epochs 10 `
  --train-fraction 0.1 --batch-size 512 --device cuda `
  --output-root outputs_first_layer_cross_task --resume

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion cifar10 sign --model standard --init-layers 1 `
  --methods lowrank_wmf --covariance-rank 16 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --device cuda `
  --output-root outputs_first_layer_cross_task --resume

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets sign --model standard --init-layers 1 `
  --methods random_stem kaiming gabor pca kmeans di_wmf `
  --seeds 5 6 7 8 9 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --device cuda `
  --output-root outputs_first_layer_cross_task --resume

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets sign --model standard --init-layers 1 `
  --methods lowrank_wmf --covariance-rank 16 `
  --seeds 5 6 7 8 9 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --device cuda `
  --output-root outputs_first_layer_cross_task --resume

conda run --no-capture-output -n CL2 python analyze_first_layer_cross_task.py
conda run --no-capture-output -n CL2 python analyze_multiple_comparisons.py
```

The controlled result is intentionally different from the full layer-wise
study: no stem has a significant Epoch-0 advantage because the downstream
network is random. Structured stems improve short-budget AULC and final
accuracy on Fashion-MNIST and CIFAR-10. With ten Sign seeds, only the K-means
AULC gain survives six-method Holm correction; no Sign final-accuracy gain does.

Experiment B compares complete layer-wise initialisation. `kmeans` supplies
unwhitened class-conditional prototypes, `di_wmf` adds diagonal whitening, and
`lowrank_wmf` adds correlated covariance directions.

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion --model standard --init-layers 3 `
  --methods kmeans di_wmf lowrank_wmf --covariance-ranks 4 8 16 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --device cuda --output-root outputs_frontier_b
```

Shared concepts and residual cross-architecture runs:

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion --model concept --init-layers 3 `
  --methods kaiming di_wmf --sparsity-lambda 0.05 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --device cuda --output-root outputs_frontier_arch

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion --model smallresnet --init-layers 3 `
  --methods kaiming di_wmf --seeds 0 1 2 3 4 --epochs 10 `
  --train-fraction 0.1 --batch-size 512 --device cuda `
  --output-root outputs_frontier_arch
```

Multi-corruption evaluation accepts one or more models and all eight
corruptions listed above. `analyze_frontier_results.py` reports seed-paired
differences and 95% confidence intervals, rather than treating the five seeds
as unrelated runs.

```powershell
conda run --no-capture-output -n CL2 python analyze_frontier_results.py `
  --corruption-roots corruption_results/frontier_fashion_arch_5seed
```

First-layer selectivity for Experiment A is evaluated directly from spatially
averaged ReLU responses, rather than inferred from the final classifier:

```powershell
conda run --no-capture-output -n CL2 python evaluate_filter_selectivity.py `
  --output-root outputs_frontier_a `
  --results-root selectivity_results/frontier_fashion_stem_5seed `
  --datasets fashion --models standard `
  --methods random_stem kaiming gabor pca kmeans di_wmf `
  --seeds 0 1 2 3 4 --epochs 10 --init-layers 1 --device cuda
```

Exact evidence-faithfulness and explanation-stability evaluation:

```powershell
conda run -n CL2 python evaluate_faithfulness.py `
  --datasets fashion cifar10 sign --methods random di_wmf `
  --seeds 0 1 2 3 4 --fractions 0.1 0.25 0.5 `
  --severities 0.1 0.2 0.3 --noise-seeds 0 1 2 --device cuda
```

The same exact intervention analysis works for the shared concept bottleneck:

```powershell
conda run --no-capture-output -n CL2 python evaluate_faithfulness.py `
  --output-root outputs_frontier_arch `
  --results-root faithfulness_results/frontier_concept_5seed `
  --datasets fashion --models concept --methods kaiming di_wmf `
  --reference-method kaiming --comparison-method di_wmf `
  --seeds 0 1 2 3 4 --epochs 10 --init-layers 3 --device cuda
```

For StandardCNN, SharedConceptCNN and SmallResNet, each class logit is
decomposed exactly into signed
per-channel template contributions. The evaluator measures sufficiency by
retaining only the top-k contributing channels, comprehensiveness by deleting
them, and causal advantage against an equal-size random deletion. Under
Gaussian noise it also measures top-k explanation Jaccard, signed contribution
cosine similarity, rank correlation, and prediction consistency. All
interventions occur at the pooled convolutional bottleneck and require no
retraining.

## Cross-task and long-horizon extension

The layer-wise covariance ablation and SmallResNet study have also been run on
CIFAR-10 and Sign Language MNIST with five paired seeds:

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets cifar10 sign --model standard --init-layers 3 `
  --methods kmeans di_wmf lowrank_wmf --covariance-ranks 8 16 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --num-workers 2 --device cuda `
  --output-root outputs_cross_task_b

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets cifar10 sign --model smallresnet --init-layers 3 `
  --methods kaiming di_wmf --seeds 0 1 2 3 4 --epochs 10 `
  --train-fraction 0.1 --batch-size 512 --num-workers 2 --device cuda `
  --output-root outputs_cross_task_arch
```

The best methods were then compared with 100% training data for 20 epochs and
five seeds. CIFAR-10 and Fashion-MNIST use batch size 512; the much smaller
Sign split uses batch size 64 so that every epoch contains multiple optimiser
updates.

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets cifar10 fashion --model standard --init-layers 3 `
  --methods kaiming di_wmf lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 20 --train-fraction 1.0 `
  --batch-size 512 --num-workers 2 --device cuda

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets sign --model standard --init-layers 3 `
  --methods kaiming di_wmf lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 20 --train-fraction 1.0 `
  --batch-size 64 --num-workers 2 --device cuda
```

The canonical ResNet18 extension uses the same five-seed, 10% data protocol:

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion cifar10 --model resnet18 --init-layers 3 `
  --methods kaiming di_wmf lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --batch-size 512 --num-workers 2 --device cuda `
  --output-root outputs_resnet18_cross_task

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets sign --model resnet18 --init-layers 3 `
  --methods kaiming di_wmf lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --batch-size 64 --num-workers 2 --device cuda `
  --output-root outputs_resnet18_cross_task
```

## Correncoder, convergence, efficiency and semantic explanations

The Correncoder uses the same convolutional weights in its classifier encoder
and transposed-convolution decoder. The auxiliary term is `1 - correlation`
between centred inputs and reconstructions, so it is insensitive to a trivial
global intensity offset.

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion cifar10 --model correncoder --init-layers 3 `
  --methods kaiming lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --auxiliary-lambda 0.1 --batch-size 512 --num-workers 2 --device cuda `
  --output-root outputs_correncoder_cross_task

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets sign --model correncoder --init-layers 3 `
  --methods kaiming lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 `
  --auxiliary-lambda 0.1 --batch-size 64 --num-workers 2 --device cuda `
  --output-root outputs_correncoder_cross_task
```

The long-horizon comparison uses full data, 50 epochs, five paired seeds and a
cosine schedule. Compact models use the same data and optimiser protocol as
their full-width references.

```powershell
conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets fashion cifar10 --model standard --init-layers 3 `
  --methods kaiming lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 50 --train-fraction 1.0 `
  --lr-scheduler cosine --min-learning-rate 1e-5 `
  --batch-size 512 --num-workers 2 --device cuda `
  --output-root outputs_convergence50

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets sign --model standard --init-layers 3 `
  --methods kaiming lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 50 --train-fraction 1.0 `
  --lr-scheduler cosine --min-learning-rate 1e-5 `
  --batch-size 64 --num-workers 2 --device cuda `
  --output-root outputs_convergence50

conda run --no-capture-output -n CL2 python run_matrix.py `
  --datasets sign --model standard --init-layers 3 `
  --methods kaiming lowrank_wmf --covariance-ranks 16 `
  --seeds 0 1 2 3 4 --epochs 100 --train-fraction 1.0 `
  --lr-scheduler cosine --min-learning-rate 1e-5 `
  --batch-size 64 --num-workers 2 --device cuda `
  --output-root outputs_convergence100_sign
```

For the compact study, run the following two dataset groups for both widths:

```powershell
foreach ($model in 'standard_half','standard_quarter') {
  conda run --no-capture-output -n CL2 python run_matrix.py `
    --datasets fashion cifar10 --model $model --init-layers 3 `
    --methods kaiming lowrank_wmf --covariance-ranks 16 `
    --seeds 0 1 2 3 4 --epochs 20 --train-fraction 1.0 `
    --batch-size 512 --num-workers 2 --device cuda `
    --output-root outputs_efficiency_width

  conda run --no-capture-output -n CL2 python run_matrix.py `
    --datasets sign --model $model --init-layers 3 `
    --methods kaiming lowrank_wmf --covariance-ranks 16 `
    --seeds 0 1 2 3 4 --epochs 20 --train-fraction 1.0 `
    --batch-size 64 --num-workers 2 --device cuda `
    --output-root outputs_efficiency_width
}
```

The pruning script reports accuracy retention against actual non-zero weights;
it deliberately does not claim dense-PyTorch latency gains from unstructured
zeros. The semantic evaluator measures top-activation label purity and entropy,
then compares exact template CAM, Grad-CAM, Integrated Gradients and matched
random masks using pixel deletion.

```powershell
conda run --no-capture-output -n CL2 python evaluate_pruning.py `
  --output-roots outputs_full20_fashion outputs_full20_cifar10 `
    outputs_full20_sign_bs64 `
  --results-root pruning_results/full20_standard_5seed --device cuda

conda run --no-capture-output -n CL2 python evaluate_template_semantics.py `
  --output-roots outputs_full20_fashion outputs_full20_cifar10 `
    outputs_full20_sign_bs64 --max-semantic-samples 500 `
  --explanation-samples 100 `
  --results-root template_semantics_results/full20_standard_5seed --device cuda
```

See `CROSS_TASK_LONG_HORIZON_RESULTS.md` for the complete five-seed protocol,
eight-corruption results, convergence diagnostics, ResNet18 means, paired
confidence intervals, GPU audit and bounded conclusions. See
`ADVANCED_EXTENSION_RESULTS.md` for the Correncoder, 50--100 epoch convergence,
compact-model, pruning, template-semantics and explanation results.

## Full Correncoder, C4 invariance, and total cost

Run the public-task Correncoder pretraining and complete BIDMC LOSO protocol:

~~~powershell
conda run --no-capture-output -n CL2 python pretrain_correncoder_capnobase.py --epochs 80 --batch-size 30 --device cuda --output-dir outputs_correncoder_pretrain

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --num-workers 0 --device cuda --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --output-root outputs_correncoder_regression_full

conda run --no-capture-output -n CL2 python plot_correncoder_results.py --results-root outputs_correncoder_regression_full --output-dir correncoder_regression_results
~~~

The regression extensions isolate initialisation and objective design. The
initialisation comparison keeps architecture, MSE objective, optimiser and
80-epoch BIDMC LOSO protocol fixed, and changes only Random, CapnoBase
pretraining, or the fold-training-only 1-D matched-filter stem. The objective
ablation fixes the CapnoBase checkpoint and compares MSE with correlation,
respiratory-band spectral, and combined auxiliary terms (weight 0.1).

~~~powershell
conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation random --resume --output-root outputs_correncoder_initialisation/random_mse

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation matched_1d --matched-max-patches 50000 --matched-shrinkage 0.1 --resume --output-root outputs_correncoder_initialisation/matched_1d_mse

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation pretrained --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --correlation-lambda 0.1 --resume --output-root outputs_correncoder_loss_ablation/pretrained_corr

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation pretrained --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --spectral-lambda 0.1 --resume --output-root outputs_correncoder_loss_ablation/pretrained_spectral

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation pretrained --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --correlation-lambda 0.1 --spectral-lambda 0.1 --resume --output-root outputs_correncoder_loss_ablation/pretrained_corr_spectral

conda run --no-capture-output -n CL2 python evaluate_correncoder_epoch0.py --device cuda --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --output-root correncoder_epoch0_results
~~~

The deeper extension and phase-tolerant objective use the same 53-fold,
80-epoch protocol:

~~~powershell
conda run --no-capture-output -n CL2 python evaluate_correncoder_epoch0.py --device cuda --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --matched-deep-max-patches 10000 --output-root correncoder_epoch0_extended_results

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation matched_layerwise --matched-max-patches 50000 --matched-deep-max-patches 10000 --matched-shrinkage 0.1 --affine-calibration --resume --output-root outputs_correncoder_extensions/matched_layerwise_calibrated_mse

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation pretrained --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --correlation-lambda 0.1 --correlation-mode max_lag --max-lag-samples 30 --lag-step 3 --resume --output-root outputs_correncoder_extensions/pretrained_lag_corr

conda run --no-capture-output -n CL2 python analyze_correncoder_extensions.py --experiment pretrained_mse=outputs_correncoder_regression_full --experiment matched_1d=outputs_correncoder_initialisation/matched_1d_mse --experiment matched_layerwise_calibrated=outputs_correncoder_extensions/matched_layerwise_calibrated_mse --experiment pretrained_corr=outputs_correncoder_loss_ablation/pretrained_corr --experiment pretrained_lag_corr=outputs_correncoder_extensions/pretrained_lag_corr --comparison matched_layerwise_calibrated=matched_1d --comparison matched_layerwise_calibrated=pretrained_mse --comparison pretrained_lag_corr=pretrained_corr --comparison pretrained_lag_corr=pretrained_mse --output-root correncoder_extension_results
~~~

The completed safe-calibration depth/covariance and differentiable soft-lag
extensions are reproduced with:

~~~powershell
conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation matched_layerwise --matched-depth 1 --matched-deep-covariance diagonal --safe-calibration-head --calibration-min-gain 0.05 --calibration-warmup-epochs 3 --matched-max-patches 50000 --matched-deep-max-patches 10000 --resume --output-root outputs_correncoder_depth_ste/depth1

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation matched_layerwise --matched-depth 3 --matched-deep-covariance lowrank --matched-covariance-rank 16 --safe-calibration-head --calibration-min-gain 0.05 --calibration-warmup-epochs 3 --matched-max-patches 50000 --matched-deep-max-patches 10000 --resume --output-root outputs_correncoder_depth_ste/depth3_lowrank

conda run --no-capture-output -n CL2 python run_correncoder_regression.py --dataset bidmc --epochs 80 --batch-size 30 --device cuda --initialisation pretrained --pretrained-capnobase outputs_correncoder_pretrain/checkpoint_capnobase_all.pt --correlation-lambda 0.1 --correlation-mode soft_lag --max-lag-samples 30 --lag-step 3 --lag-temperature 0.05 --resume --output-root outputs_correncoder_extensions/pretrained_soft_lag_corr
~~~

The full 53-fold comparison finds no significant adjacent-depth or rank-16
low-rank-versus-diagonal effect. Soft-lag significantly raises max-lag
correlation over ordinary Pearson, but is statistically indistinguishable from
hard max-lag. See `correncoder_completed_extension_results` for the paired
tables and plot.

Completed results show no significant 80-epoch difference among the three
initialisations. MSE + 0.1(1-Pearson correlation) significantly improves
held-out MSE, MAE and waveform correlation versus MSE-only; the spectral term
does not establish an RR-error benefit and adds no significant gain over the
correlation objective. Full numerical results and claim boundaries are in
`CORRENCODER_ROTATION_COST_RESULTS.md`.

The layerwise matched variant significantly improves converged MSE, MAE, and
zero-lag correlation over the matched stem, but not over CapnoBase pretraining.
The max-lag objective significantly reduces RR MAE relative to ordinary
Pearson loss; its other paired effects against ordinary Pearson are
inconclusive. The stopped calibrated-stem run is a documented failure
ablation: a near-zero absorbed gain caused constant predictions.

Train and evaluate the exact quarter-turn invariant architecture:

~~~powershell
conda run --no-capture-output -n CL2 python run_matrix.py --datasets fashion cifar10 sign --model rotation_invariant --methods kaiming lowrank_wmf --seeds 0 1 2 3 4 --epochs 10 --train-fraction 0.1 --batch-size 256 --num-workers 1 --init-layers 3 --covariance-rank 16 --device cuda --output-root outputs_rotation

conda run --no-capture-output -n CL2 python evaluate_rotation_invariance.py --roots outputs_rotation --models standard rotation_invariant --methods kaiming lowrank_wmf --datasets fashion cifar10 sign --angles -180 -150 -120 -90 -60 -30 0 30 60 90 120 150 180 --batch-size 512 --num-workers 1 --device cuda --output-dir rotation_invariance_results
~~~

Generate time and energy to fixed accuracy from energy-instrumented runs with
analyze_total_cost.py. The completed protocols, numerical results, and claim
boundaries are documented in CORRENCODER_ROTATION_COST_RESULTS.md.

## Final thesis-strengthening experiments

`correncoder_full` adds an independently trainable decoder initialised from
the matched encoder weights. It is a controlled fuller encoder--decoder
ablation; it is not labelled as an exact reproduction of the published
Correncoder regression task. The tied model is swept over auxiliary weights
0, 0.03, 0.1 and 0.3, and the untied model is compared at the pre-specified
weight 0.1.

Modern CIFAR-10 augmentation is enabled with `--augmentation cifar_standard`.
Random crop and horizontal flip are applied only to optimiser training views;
the matched-filter fitting loader, validation loader and test loader remain
clean and deterministic.

The following evaluators reuse the full-data 20-epoch checkpoints:

```powershell
conda run --no-capture-output -n CL2 python evaluate_structured_efficiency.py `
  --output-roots outputs_full20_fashion outputs_full20_cifar10 `
    outputs_full20_sign_bs64 `
  --results-root structured_efficiency_results/full20_standard `
  --datasets fashion cifar10 sign --methods kaiming lowrank_wmf `
  --seeds 0 1 2 3 4 --sparsities 0 0.25 0.5 0.75 `
  --batch-sizes 1 64 --device cuda

conda run --no-capture-output -n CL2 python evaluate_reliability.py `
  --output-roots outputs_full20_fashion outputs_full20_cifar10 `
    outputs_full20_sign_bs64 `
  --results-root reliability_results/full20_standard `
  --datasets fashion cifar10 sign --methods kaiming lowrank_wmf `
  --seeds 0 1 2 3 4 --max-eval-samples 2000 `
  --max-adversarial-samples 1000 --pgd-steps 10 --device cuda
```

Structured pruning physically removes class-balanced channels and reports
parameter/FLOP reduction, batch-1 and batch-64 CUDA latency, incremental CUDA
allocation and sampled board power. The `nvidia-smi` energy value is explicitly
treated as a noisy secondary estimate. Reliability evaluation reports raw and
temperature-scaled ECE/NLL/Brier score, selective AURC, cross-dataset OOD
AUROC/FPR95, and pixel-domain FGSM/PGD accuracy.

After all extension experiments complete, generate the Correncoder and
augmentation statistics, then the consolidated dissertation tables and plots:

```powershell
conda run --no-capture-output -n CL2 python analyze_correncoder_ablation.py `
  --experiment random_mse=outputs_correncoder_initialisation/random_mse `
  --experiment pretrained_mse=outputs_correncoder_regression_full `
  --experiment matched_1d_mse=outputs_correncoder_initialisation/matched_1d_mse `
  --experiment pretrained_corr=outputs_correncoder_loss_ablation/pretrained_corr `
  --experiment pretrained_spectral=outputs_correncoder_loss_ablation/pretrained_spectral `
  --experiment pretrained_corr_spectral=outputs_correncoder_loss_ablation/pretrained_corr_spectral `
  --output-root correncoder_regression_ablation_results

conda run --no-capture-output -n CL2 python analyze_augmentation.py `
  --output-roots outputs_full20_cifar10 outputs_cifar10_augmentation `
  --results-root augmentation_results

conda run --no-capture-output -n CL2 python make_thesis_artifacts.py
```

The dissertation-ready organisation and derivation are recorded in
`THESIS_EXPERIMENT_MAP.md` and `THESIS_METHODS_AND_DERIVATION.md`.

The completed, bounded interpretation of every main and extension experiment
is in `FINAL_THESIS_RESULTS_DISCUSSION.md`. Regenerate the submission-facing
CSV tables and seven core figures with `python make_thesis_artifacts.py`; the
outputs are written to `thesis_artifacts/`. This includes the 10-seed Sign
confirmation and the expanded explanation/faithfulness tables, so the reported
values are derived from stored checkpoints rather than copied manually.
