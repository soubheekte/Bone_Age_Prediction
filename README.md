# Bone Age Prediction — Project Documentation

## Project Overview (Problem Statement)
Automate pediatric bone age assessment from left-hand X‑ray images using deep learning. The goal is to predict chronological bone age (in months) from radiographs and associated simple demographic features (gender). Automating this task reduces inter-reader variability, speeds diagnosis, and aids large-scale epidemiological studies.

Goals:
- Build a robust pipeline for preprocessing X‑ray images, training a regression model, and validating performance.
- Keep the system reproducible, logged, and friendly for incremental preprocessing (batch-wise).
- Target clinical-grade performance (see Ideal Metrics section).

## Dataset
Primary dataset: RSNA-style hand X‑ray collection with labels in CSV form:
- Labels CSV: [`Data/boneage-dataset-labels.csv`](Data/boneage-dataset-labels.csv)
- Raw images are expected under the `Data/` tree. The repository uses paths defined in [`config.DataConfig`](config/config.py).

Research & references: see curated literature in `research/`:
- [`research/research.csv`](research/research.csv)
- [`research/Best-WhyItsaTopResource-Year-Link.csv`](research/Best-WhyItsaTopResource-Year-Link.csv)
- [`research/claude.csv`](research/claude.csv)

## Repository Layout (key files)
- [main.py](main.py) — CLI entrypoint that orchestrates preprocessing, training, and validation, using [`config.Config`](config/config.py) settings.
- [preprocessing_images.py](preprocessing_images.py) — image loading, batching and saving logic; main class: [`preprocessing_images.BoneAgeDataPreprocessing`](preprocessing_images.py).
- [model.py](model.py) — model architecture and training; key symbols: [`model.BoneAgeModelTrainer`](model.py) and [`model.BoneAgeDataGenerator`](model.py).
- [validation.py](validation.py) — validation runner that loads the latest saved model and computes metrics via [`validation.ValidationModel`](validation.py).
- [utils/logger.py](utils/logger.py) — structured logging helpers and hourly rotating handler: [`utils.logger.get_logger`](utils/logger.py).
- [utils/common_utils.py](utils/common_utils.py) — metric computation and model rotation saving helpers: [`utils.common_utils.compute_regression_metrics`](utils/common_utils.py), [`utils.common_utils.save_model_with_rotation`](utils/common_utils.py).
- [config/config.py](config/config.py) — central configuration dataclasses: [`config.Config`](config/config.py) and [`config.DataConfig`](config/config.py).
- [requirements.txt](requirements.txt) — pinned Python dependencies.

## Data preprocessing pipeline
Implemented in [`preprocessing_images.BoneAgeDataPreprocessing`](preprocessing_images.py).

Key behavior:
- Reads labels CSV via `DataConfig.labels_csv_path` and composes image paths using `DataConfig.train_image_path` or `DataConfig.valid_image_path`.
- Filters out missing image files.
- Loads images with TensorFlow utilities and converts them to grayscale arrays resized to 224×224:
  - `load_and_preprocess_image` uses `tf.keras.utils.load_img(..., target_size=(224,224), color_mode='grayscale')`.
  - Normalization: divide image array by its maximum pixel value (if > 1).
  - Images are saved in batches as NumPy `.npy` files under `Data/...` directories defined in `DataConfig`.
- Preprocessing is incremental and idempotent: processed IDs are tracked in text files (e.g., `Data/train_processed_image_ids.txt`) to resume processing.

Files produced:
- Batch image files: `Data/*_preprocessed_images/*.npy`
- Batch label files: `Data/*_preprocessed_labels/*.npy`
- Batch gender files: `Data/*_preprocessed_genders/*.npy`

## Model architecture
Defined in [`model.BoneAgeModelTrainer`](model.py).

Summary:
- Two-input architecture:
  - Image branch: small AlexNet‑like convolutional stack operating on (224,224,1), followed by Dense(8) projection.
  - Demographic branch: Dense layers for the binary `male` input.
  - Concatenation of image+demographic features, followed by two large Dense layers (4096 → 4096) with Dropout and final Dense(1) regression output.
- Data feeding: [`model.BoneAgeDataGenerator`](model.py) is a Keras Sequence that reads saved `.npy` batch files and yields ((images, gender), labels).

Notes:
- The architecture is intentionally simple / educational. Replace or extend with pretrained backbones (ResNet, EfficientNet, ViT) for production-quality performance.

## Training
Entry points:
- CLI runner: [main.py](main.py)
  - Example commands:
    - Preprocess training set: python main.py --preprocess train
    - Preprocess validation set: python main.py --preprocess validate
    - Run full pipeline: python main.py --run_pipeline True
    - Run training only: python main.py --train True

Hyperparameters and paths are centrally in [`config.Config`](config/config.py) and [`config.DataConfig`](config/config.py):
- `Config.EPOCHS`, `Config.BATCH_SIZE`, `Config.TEST_SIZE`, `Config.LEARNING_RATE`.
- Default BASEDIR is `.` so model files go under `./model_files`.

Training specifics:
- Loss: Mean Absolute Error (MAE).
- Optimizer: Adam.
- Early stopping: `tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` is enabled in [`model.BoneAgeModelTrainer.train_model`](model.py).
- Model saving uses [`utils.common_utils.save_model_with_rotation`](utils/common_utils.py) to keep only a rotating set of model files.

## Validation & Evaluation
Validation runner: [`validation.ValidationModel`](validation.py).
- Loads the latest saved model from `./model_files` (prefers `.keras`, falls back to `.h5`).
- Predicts on validation batches using the same `BoneAgeDataGenerator`.
- Aggregates true labels and predictions and computes metrics via [`utils.common_utils.compute_regression_metrics`](utils/common_utils.py).
- Logged metrics use [`utils.logger.get_logger`](utils/logger.py).

Implemented metrics (returned by `compute_regression_metrics`):
- Mean Squared Error (MSE): $MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$
- Root Mean Squared Error (RMSE): $RMSE = \sqrt{MSE}$
- Mean Absolute Error (MAE): $MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
- R² (coefficient of determination)
- Mean Absolute Percentage Error (MAPE) in percent, with small epsilon guard to avoid division-by-zero

Formulas are computed in [`utils.common_utils.compute_regression_metrics`](utils/common_utils.py).

## Ideal / Target Metrics (guidance)
Targets depend on dataset and clinical requirements. Typical literature numbers (RSNA-style & state-of-art) show:

- Baseline acceptable: MAE ≲ 6.0 months
- Research-grade / competitive: MAE ≲ 3.5–4.5 months
- State-of-the-art (best published): MAE ≲ 2.0–3.0 months

Other useful targets:
- RMSE should be close to MAE magnitude for low-variance errors.
- R² > 0.85 indicates strong correlation; R² > 0.9 is excellent.

Note: Clinical deployment requires careful external validation, calibration, and fairness checks across sex, age ranges, and ethnicities.

## Logging & Model Management
- Logging: [`utils.logger.get_logger`](utils/logger.py) configures both a console stream handler and a custom HourlyFileHandler that rotates active file per hour. It also supports JSON formatting.
- Model rotation: [`utils.common_utils.save_model_with_rotation`](utils/common_utils.py) saves models with versioned filenames and keeps a limited number of recent models in `model_files/`.

## Reproducibility & Configuration
- Global parameters in [`config.Config`](config/config.py).
- Data paths and processed file locations in [`config.DataConfig`](config/config.py).
- Set `CUDA_VISIBLE_DEVICES` in [main.py](main.py) if you want to disable GPU for debugging (already set to `-1` in current main).

## How to set up the environment
1. Create a virtual environment and install dependencies:
   - python 3.10+ recommended (match the TensorFlow version in [requirements.txt](requirements.txt)).
   - pip install -r requirements.txt

2. Prepare data:
   - Place raw images under directories defined in [`config.DataConfig.train_image_path`](config/config.py) and [`config.DataConfig.valid_image_path`](config/config.py).
   - Ensure labels CSV is at [`Data/boneage-dataset-labels.csv`](Data/boneage-dataset-labels.csv) or update `DataConfig.labels_csv_path`.

3. Run preprocessing (batch-safe):
   - python main.py --preprocess train
   - python main.py --preprocess validate

4. Train:
   - python main.py --train True

5. Validate:
   - python main.py --validate True

6. Full pipeline:
   - python main.py --run_pipeline True

## Extensibility & Next Steps
- Replace the backbone with a pretrained encoder (e.g., EfficientNet, ResNet, Vision Transformer). Use transfer learning and fine‑tuning to improve sample efficiency.
- Add data augmentation (rotation, small shifts, contrast) while keeping medically plausible transforms.
- Implement stratified validation by age bins and gender.
- Add explainability (Grad-CAM / Score-CAM) to visualize model focus regions.
- Add unit tests for preprocessing, generators, and metrics.

## Known Caveats & Assumptions
- Current preprocessing normalizes each image by its max pixel value (adaptive per image). For strict comparability, prefer fixed-range normalization (e.g., divide by 255).
- The dataset loader assumes `.png` images with IDs matching the labels CSV `id` column.
- The training Dense sizes (4096) are large; for small datasets reduce capacity to avoid overfitting.

## Contact / References
- See the curated reading list in `research/` for recent state-of-the-art and datasets:
  - [`research/research.csv`](research/research.csv)
  - [`research/Best-WhyItsaTopResource-Year-Link.csv`](research/Best-WhyItsaTopResource-Year-Link.csv)
  - [`research/claude.csv`](research/claude.csv)

## Quick links to code symbols
- [`preprocessing_images.BoneAgeDataPreprocessing`](preprocessing_images.py)
- [`model.BoneAgeModelTrainer`](model.py)
- [`model.BoneAgeDataGenerator`](model.py)
- [`validation.ValidationModel`](validation.py)
- [`utils.logger.get_logger`](utils/logger.py)
- [`utils.common_utils.compute_regression_metrics`](utils/common_utils.py)
- [`utils.common_utils.save_model_with_rotation`](utils/common_utils.py)
- [`config.Config`](config/config.py) and [`config.DataConfig`](config/config.py)
- [main.py](main.py)
- [requirements.txt](requirements.txt)

## License & Ethics
- Ensure patient data is de-identified and you have rights to use the dataset.
- Evaluate fairness across demographic groups before any clinical deployment.
