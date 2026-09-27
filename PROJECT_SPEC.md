# PneumoVision

## Chest X-Ray Pneumonia Classification with PyTorch, Transfer Learning and Fine-Tuning

You are the primary coding agent responsible for implementing this project.

Your role is not merely to generate code. You must build a working, reproducible deep-learning project that can be inspected, executed, experimented with, and explained by its owner in an AI/ML interview.

The project should demonstrate genuine competence in:

* PyTorch
* Computer Vision
* CNN architecture
* transfer learning
* fine-tuning
* training pipeline design
* class-imbalance handling
* experiment design
* model evaluation
* error analysis
* model interpretability

The project is intended for an AI/ML engineering portfolio and should emphasize **model development and experimentation**, not application-layer complexity.

---

# 1. HIGH-LEVEL OBJECTIVE

Build a binary chest X-ray classification system:

```text
Input:
Chest X-ray image

Output:
NORMAL
or
PNEUMONIA
```

The central question of the project is:

> How do model architecture, transfer learning, fine-tuning, and class-imbalance handling affect chest X-ray classification performance?

The project should follow:

```text
Data
  ↓
Data Audit
  ↓
Reproducible Split
  ↓
Preprocessing / Augmentation
  ↓
Custom CNN Baseline
  ↓
ResNet18 Transfer Learning
  ↓
ResNet18 Partial Fine-Tuning
  ↓
EfficientNet-B0 Fine-Tuning
  ↓
Controlled Experiments
  ↓
Evaluation
  ↓
Error Analysis
  ↓
Grad-CAM / Threshold Analysis
  ↓
Final Inference
```

Do not optimize for the appearance of a sophisticated repository.

Optimize for:

> A project that the owner can genuinely understand and explain during an AI interview.

---

# 2. IMPORTANT SCOPE DECISION

Do NOT turn this project into a giant ML platform.

Do NOT add unnecessary:

* MobileNet
* DenseNet
* ResNet50
* YOLO
* object detection
* MLflow
* Kubernetes
* Docker
* large hyperparameter searches
* complex cloud deployment
* unnecessary APIs
* unnecessary microservices

The core project is:

```text
Custom CNN
      ↓
ResNet18
      ↓
EfficientNet-B0
      ↓
Transfer Learning
      ↓
Fine-Tuning
      ↓
Controlled Experiments
      ↓
Evaluation
```

Two pretrained architectures are enough.

---

# 3. HARDWARE CONSTRAINT

Development environment is CPU-first.

The machine does NOT have an NVIDIA GPU.

Therefore:

* CUDA may be unavailable.
* The project MUST run completely on CPU.
* Detect CUDA automatically if available.
* Never assume CUDA exists.
* Keep batch size configurable.
* Keep image size configurable.
* Keep num_workers configurable.
* Avoid unnecessarily large models.
* Avoid huge hyperparameter searches.

The default configuration should be realistic for CPU execution.

---

# 4. DATASET

Dataset location:

```text
data/chest_xray/
```

Expected structure:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Current approximate counts:

```text
TRAIN
NORMAL:      1341
PNEUMONIA:   3875

VAL
NORMAL:      8
PNEUMONIA:   8

TEST
NORMAL:      234
PNEUMONIA:   390
```

These numbers must NOT simply be trusted.

The first task is to inspect the actual filesystem and generate the real counts.

---

# 5. DATASET LIMITATION — IMPORTANT

The original validation set contains only 16 images.

That is too small to be a reliable validation set for model selection.

Therefore:

* The original TEST split must remain completely untouched.
* Do NOT tune hyperparameters using the test set.
* Do NOT select the final model using test performance.
* Create a proper validation split from the original TRAIN data.
* Use the new validation split for development/model selection.
* Use TEST only for final evaluation after the model/configuration has been selected.

Use a reproducible seed.

Prefer a stratified split preserving class proportions.

Document this clearly.

---

# 6. PHASE 0 — REPOSITORY INSPECTION

Before writing substantial code:

1. Inspect the existing repository.
2. Inspect current files.
3. Inspect dataset structure.
4. Inspect existing scripts.
5. Inspect existing dependencies.
6. Determine what is already implemented.
7. Preserve useful existing work.
8. Avoid blindly overwriting working code.

Then report:

```text
Current state
Existing components
Missing components
Proposed implementation order
```

Do not implement the entire project in one step.

---

# 7. IMPLEMENTATION WORKFLOW

Implement the project incrementally.

Use this order:

```text
Phase 1  Data Audit
Phase 2  Reproducible Dataset Split
Phase 3  PyTorch Dataset/DataLoader
Phase 4  Custom CNN
Phase 5  Training Engine
Phase 6  Evaluation
Phase 7  ResNet18 Transfer Learning
Phase 8  ResNet18 Fine-Tuning
Phase 9  EfficientNet-B0
Phase 10 Controlled Experiments
Phase 11 Class-Weighted Loss
Phase 12 Error Analysis
Phase 13 Threshold Analysis
Phase 14 Grad-CAM
Phase 15 Inference CLI
Phase 16 Documentation
```

After each major phase:

1. Run relevant checks.
2. Fix errors.
3. Verify behavior.
4. Report what was actually completed.
5. Do not claim success without executing the code.

---

# 8. PHASE 1 — DATA AUDIT

Create a dataset audit script.

Suggested:

```text
scripts/audit_dataset.py
```

Inspect:

* image counts
* class distribution
* image dimensions
* image modes
* corrupted images
* unreadable files
* missing files
* suspicious duplicates if computationally reasonable
* train/validation/test distribution

Generate a machine-readable report:

```text
experiments/data_audit.json
```

Generate useful plots if appropriate.

Do NOT silently delete or modify dataset files.

The audit should be reproducible.

---

# 9. PHASE 2 — REPRODUCIBLE SPLIT

Create a proper validation set from the original training data.

Requirements:

* fixed random seed
* stratified class distribution
* test set untouched
* no leakage
* reproducible sample assignment

Save split metadata.

For example:

```text
experiments/data_split.json
```

The same train/validation samples must be reused across experiments.

Do not regenerate a different validation set every time training starts.

---

# 10. PHASE 3 — PYTORCH DATA PIPELINE

Implement a clean PyTorch input pipeline.

Requirements:

* Dataset
* DataLoader
* training transforms
* validation transforms
* test transforms

Use a configurable image size, defaulting to:

```text
224 x 224
```

Training augmentation may include:

* Resize
* RandomHorizontalFlip
* small RandomRotation
* mild RandomAffine
* mild brightness/contrast variation where reasonable

Avoid aggressive transformations.

Validation/test:

```text
Resize
Normalize
```

No stochastic augmentation.

Make transforms configurable.

The implementation must correctly handle X-ray images and pretrained ImageNet models.

---

# 11. PHASE 4 — CUSTOM CNN BASELINE

Implement a small CNN from scratch using PyTorch.

The architecture should include reasonable combinations of:

```text
Conv2d
BatchNorm
ReLU
MaxPool
Global Average Pooling
Linear
```

The model does NOT need to be state-of-the-art.

Purpose:

> Establish a genuine from-scratch PyTorch baseline.

The CNN should be simple enough that the owner can explain every major component.

Do not make it unnecessarily deep.

---

# 12. PHASE 5 — TRAINING ENGINE

Create a reusable training engine.

Required capabilities:

* training loop
* validation loop
* optimizer
* learning-rate scheduler
* early stopping
* checkpointing
* best-model saving
* training history
* seed handling
* device selection

Use AdamW by default unless there is a strong reason otherwise.

Configurable parameters:

```text
batch_size
learning_rate
epochs
weight_decay
scheduler
early_stopping_patience
image_size
model
augmentation
class_weighting
```

Do not scatter hard-coded hyperparameters across the codebase.

---

# 13. PHASE 6 — EVALUATION ENGINE

Implement reusable evaluation.

Required metrics:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Confusion Matrix

Report:

* NORMAL metrics
* PNEUMONIA metrics
* macro averages
* weighted averages where appropriate

Do NOT treat accuracy as the primary metric.

The project must explicitly inspect:

```text
PNEUMONIA recall
```

and:

```text
False Negatives:
PNEUMONIA → NORMAL
```

Do not make clinical claims.

This is model-performance analysis on a dataset.

---

# 14. PHASE 7 — RESNET18 TRANSFER LEARNING

Implement pretrained ResNet18.

Use torchvision pretrained weights.

Replace the classification head appropriately.

First experiment:

```text
Image
 ↓
Pretrained ResNet18
 ↓
Frozen backbone
 ↓
Trainable classification head
```

Question being tested:

> How useful are pretrained ImageNet representations when transferred to this chest X-ray classification task?

Keep the experiment reproducible.

---

# 15. PHASE 8 — RESNET18 PARTIAL FINE-TUNING

This is a CORE experiment.

Do NOT treat fine-tuning as an optional implementation detail.

Compare:

```text
Experiment B:
Frozen ResNet18 backbone

vs

Experiment C:
Partial fine-tuning
```

For partial fine-tuning:

* keep early layers frozen
* unfreeze later layers, such as layer4
* keep classifier trainable

Use differential learning rates when appropriate.

For example:

```text
classifier LR > backbone LR
```

Do not blindly use the same learning rate for everything.

The exact values should be configurable.

The purpose is to investigate:

> Does adapting the pretrained representation to the chest X-ray domain improve validation performance?

---

# 16. PHASE 9 — EFFICIENTNET-B0

Implement EfficientNet-B0 with pretrained weights.

Use it as the second pretrained architecture.

The main experiment should be comparable to the best ResNet18 configuration.

Do not perform an enormous hyperparameter search.

Question:

> Does a different pretrained architecture provide better performance or a different accuracy/complexity trade-off?

Do not assume EfficientNet will win.

The actual experiment results determine the conclusion.

---

# 17. PHASE 10 — CONTROLLED EXPERIMENT MATRIX

Use a small experiment matrix with a clear purpose.

Minimum experiments:

### Experiment A — Custom CNN

Purpose:

```text
from-scratch baseline
```

### Experiment B — ResNet18 Frozen

Purpose:

```text
transfer learning
```

### Experiment C — ResNet18 Partial Fine-Tuning

Purpose:

```text
effect of domain adaptation through fine-tuning
```

### Experiment D — EfficientNet-B0 Fine-Tuning

Purpose:

```text
architecture comparison
```

### Experiment E — Best Architecture + Weighted Loss

Purpose:

```text
effect of class imbalance handling
```

### Experiment F — Best Configuration + Threshold Analysis

Purpose:

```text
decision threshold trade-off
```

Do not run arbitrary experiments just to increase the number of rows in a table.

Every experiment must answer a question.

---

# 18. CLASS IMBALANCE

The training data is imbalanced.

PNEUMONIA is substantially more common than NORMAL.

Implement class-weighted loss.

Calculate class weights from the training split only.

Never use test data to calculate them.

Compare:

```text
standard loss
vs
weighted loss
```

The weighted-loss experiment should be applied to the best architecture/configuration identified using validation data.

Do not assume weighted loss will improve every metric.

---

# 19. EXPERIMENT LOGGING

Use a lightweight experiment registry.

Do NOT introduce MLflow unless there is a strong reason.

CSV/JSON is sufficient.

Each experiment should record:

```text
experiment_id
model
pretrained
fine_tuning_strategy
augmentation
class_weighting
optimizer
learning_rate
batch_size
epochs
best_epoch
validation_loss
validation_accuracy
validation_precision
validation_recall
validation_f1
validation_roc_auc
validation_pr_auc
```

Store results under:

```text
experiments/
```

Never overwrite previous experiment results.

---

# 20. MODEL SELECTION

This is important:

DO NOT decide beforehand that:

```text
EfficientNet > ResNet
```

or:

```text
ResNet > CNN
```

The experiment results must determine the conclusion.

Model selection should be based on validation performance and the project's stated evaluation priorities.

Do NOT use the test set for model selection.

Once the final model/configuration is selected:

```text
LOCK CONFIGURATION
        ↓
EVALUATE ON TEST SET ONCE
```

Clearly distinguish:

```text
validation results
```

from:

```text
final test results
```

---

# 21. THRESHOLD ANALYSIS

After model selection, implement threshold analysis.

Use validation data.

Evaluate thresholds such as:

```text
0.30
0.40
0.50
0.60
0.70
```

For each threshold calculate:

* Precision
* Recall
* F1

Plot the trade-off.

Do NOT select a threshold using test data.

Then evaluate the selected threshold on the held-out test set.

Document this procedure.

---

# 22. ERROR ANALYSIS

Implement error analysis.

Identify:

```text
False Positive:
NORMAL → PNEUMONIA

False Negative:
PNEUMONIA → NORMAL
```

For each sample record:

* image path
* true label
* predicted label
* probability
* error type

Generate visual grids for representative examples.

Do not invent medical explanations for individual images.

The analysis should describe model behavior, not diagnose the image.

---

# 23. GRAD-CAM

Implement Grad-CAM for the final selected CNN-based architecture.

Requirements:

* original image
* predicted class
* probability
* Grad-CAM visualization

The purpose is:

> Inspect whether model attention is concentrated in plausible image regions.

Do NOT claim:

> Grad-CAM proves that the model detects pneumonia lesions.

Grad-CAM is an interpretability visualization, not clinical proof.

If Grad-CAM becomes disproportionately difficult or unstable, prioritize all core ML components first.

---

# 24. INFERENCE CLI

Create:

```text
scripts/inference.py
```

Example:

```bash
python scripts/inference.py \
    --image path/to/image.jpg \
    --model checkpoints/best_model.pt
```

Expected output:

```text
Prediction: PNEUMONIA
Probability: 0.91
Model: EfficientNet-B0
```

The inference code must reuse the same preprocessing implementation used during evaluation.

Do not duplicate preprocessing logic.

---

# 25. FASTAPI

FastAPI is NOT part of the core project.

Only implement a minimal API if:

* all core experiments are complete
* evaluation is complete
* error analysis is complete
* documentation is complete

Never sacrifice ML work for deployment work.

---

# 26. TESTING

At minimum verify:

### Dataset

* loading works
* labels are correct
* transforms work

### Model

* forward pass works
* output shape is correct

### Training

* one mini-batch can train successfully
* gradients flow
* loss decreases during a basic smoke test where expected

### Checkpoint

* model can save
* model can reload

### Evaluation

* metrics work
* confusion matrix works

### Inference

* one real image can be passed through the trained model

Create a lightweight smoke-test command.

Example:

```bash
python scripts/train.py --config configs/smoke_test.yaml
```

The smoke test must not require a full training run.

---

# 27. CPU-FIRST TRAINING STRATEGY

The project must be practical on CPU.

Use:

* moderate image resolution
* moderate batch size
* limited epochs
* early stopping
* lightweight architectures
* configurable workers

Do not run large-scale hyperparameter sweeps.

The goal is:

```text
meaningful experiments
```

not:

```text
maximum compute
```

---

# 28. PROJECT STRUCTURE

Use a clean structure similar to:

```text
pneumonia-detection/

├── data/
│   └── chest_xray/
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── audit.py
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   ├── resnet.py
│   │   └── efficientnet.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── threshold.py
│   │   └── error_analysis.py
│   │
│   └── visualization/
│       └── gradcam.py
│
├── scripts/
│   ├── audit_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── configs/
│
├── experiments/
│   ├── data_audit.json
│   ├── data_split.json
│   ├── results.csv
│   └── figures/
│
├── checkpoints/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

You may change the exact structure if you have a clear engineering reason.

Do not create unnecessary abstractions.

---

# 29. CONFIGURATION

Prefer YAML or JSON configuration files.

For example:

```text
configs/
├── smoke_test.yaml
├── cnn.yaml
├── resnet18_frozen.yaml
├── resnet18_finetune.yaml
└── efficientnet_b0.yaml
```

Avoid hard-coded experiment parameters.

---

# 30. CODE QUALITY

Use:

* clear naming
* functions with single responsibilities
* type hints where useful
* pathlib
* logging
* configuration-driven experiments
* minimal duplication

Avoid:

* giant scripts
* giant notebooks
* magic numbers
* duplicated preprocessing
* excessive comments explaining obvious Python syntax
* fake sophistication
* unused abstractions

The code should look like something a junior ML engineer could maintain.

---

# 31. README

The README should explain:

## Problem

What is being predicted?

## Dataset

What dataset is being used?

## Dataset limitations

Especially:

* original validation set is too small
* class imbalance
* dataset size
* domain limitations
* no clinical validation

## Data strategy

Explain:

```text
Original train
    ↓
reproducible stratified split
    ↓
train + validation

Original test
    ↓
held out
```

## Models

Explain:

* Custom CNN
* ResNet18
* EfficientNet-B0

## Transfer Learning

Explain:

* frozen backbone
* partial fine-tuning

## Experiments

Explain what question each experiment answers.

## Evaluation

Explain:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

## Error Analysis

Explain:

* false positives
* false negatives

## Interpretability

Explain Grad-CAM.

## Final Results

Only include ACTUAL experiment results.

Never invent numbers.

---

# 32. MEDICAL CLAIMS

This is an ML project using a public medical imaging dataset.

Do NOT claim:

* clinical validation
* diagnostic accuracy in hospitals
* clinical deployment readiness
* medical diagnosis
* superiority over radiologists

Use language such as:

```text
model performance on the dataset
```

rather than:

```text
clinical diagnostic performance
```

---

# 33. DEFINITION OF DONE

## MUST HAVE

* [ ] dataset audit
* [ ] reproducible split
* [ ] PyTorch Dataset
* [ ] DataLoader
* [ ] preprocessing
* [ ] augmentation
* [ ] custom CNN
* [ ] ResNet18
* [ ] EfficientNet-B0
* [ ] frozen transfer learning
* [ ] partial fine-tuning
* [ ] training engine
* [ ] checkpointing
* [ ] class-weighted loss
* [ ] evaluation metrics
* [ ] experiment registry
* [ ] model comparison
* [ ] error analysis
* [ ] README

## SHOULD HAVE

* [ ] threshold analysis
* [ ] Grad-CAM
* [ ] inference CLI

## OPTIONAL

* [ ] FastAPI

Do not work on OPTIONAL items until MUST HAVE and SHOULD HAVE are complete.

---

# 34. FINAL INTERVIEW-ORIENTED REQUIREMENT

The project must allow the owner to answer questions such as:

### Data

* Why did you create a new validation split?
* Why can't you use the original validation set?
* How did you prevent test leakage?
* How did you handle class imbalance?

### Architecture

* Why start with a CNN baseline?
* Why ResNet18?
* Why EfficientNet-B0?
* What does transfer learning mean here?

### Training

* Why freeze the backbone initially?
* Why fine-tune only later layers?
* Why use a smaller learning rate for the backbone?
* Why AdamW?
* What does the scheduler do?

### Evaluation

* Why is accuracy insufficient?
* What does recall mean here?
* Why inspect PR-AUC?
* What does the confusion matrix tell you?

### Experiments

* What did you learn from the CNN baseline?
* Did transfer learning help?
* Did fine-tuning help?
* Did class weighting help?
* Which architecture performed better and why might that be?
* What happened to false negatives?

### Interpretability

* What is Grad-CAM?
* What does it actually tell you?
* What can it NOT prove?

The implementation should make these questions answerable from the actual code and experiment results.

---

# 35. AGENT BEHAVIOR

You are allowed to make reasonable engineering decisions.

However:

* Do not silently change the project objective.
* Do not add unnecessary technologies.
* Do not fabricate results.
* Do not skip experiments simply because the code runs.
* Do not claim an experiment was completed unless it was actually executed.
* Do not overwrite useful existing work without inspection.
* Do not use test data for tuning.
* Do not optimize for README appearance.
* Do not optimize for the number of files.
* Do not optimize for the number of models.

Prioritize:

```text
Correctness
>
Reproducibility
>
Experiment quality
>
Explainability
>
Engineering cleanliness
>
Extra features
```

---

# 36. FIRST TASK

Do NOT immediately implement everything.

Start by:

1. Inspecting the current repository.
2. Inspecting the dataset.
3. Running the existing checks/scripts.
4. Identifying what has already been implemented.
5. Auditing the current project structure.
6. Proposing the exact implementation plan.
7. Identifying any dataset issues that could affect the plan.

Then stop and report your findings before proceeding to implementation.

Once implementation begins, work phase-by-phase and verify each phase before moving to the next one.
