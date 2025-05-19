# DDoS Detection Model

A Python-based machine learning pipeline for detecting Distributed Denial-of-Service (DDoS) attacks using the CICDDoS2019 dataset.
The project encompasses data loading, preprocessing, exploratory data analysis, feature selection, handling class imbalance with ADASYN, model training with Random Forest, LightGBM, and MLP classifiers, evaluation, visualization, and a simulation of real-time inference.

## Features

* Load and preprocess CICDDoS2019 parquet files for training and testing.
* Map and clean labels, drop irrelevant classes.
* Exploratory Data Analysis (EDA) with descriptive statistics, histogram, and heatmap visualizations.
* Feature selection by removing low-variance and highly correlated features.
* Train/validation/test split with target encoding.
* Handle class imbalance using ADASYN oversampling.
* Scale features with MinMaxScaler.
* Train multiple classifiers: RandomForest, LightGBM (with early stopping), MLPClassifier.
* Evaluate models with accuracy, precision, recall, F1-score, ROC AUC, cross-validation.
* Visualize ROC curves and metric bar charts.
* Simulate real-time inference for LightGBM model with configurable interval.

## Project Structure

```
ddos-detection-model/
├── main.py             # Main script implementing the pipeline
├── cicddos2019.zip     # Compressed dataset files (parquet format)
├── Final Paper.docx    # Final paper documentation
└── README.md           # Project overview and instructions
```

## Requirements

* Python 3.7+

Install dependencies:

```bash
pip install numpy pandas tqdm imbalanced-learn lightgbm scikit-learn matplotlib seaborn
```

## Usage

1. Extract `cicddos2019.zip` to a local directory.
2. Update the `cicddos2019_path` variable in `main.py` to point to the extracted folder.
3. Run the script:

```bash
python main.py
```

4. Observe outputs: EDA visualizations, model evaluation metrics, ROC curves, bar charts, and real-time inference simulation.

## Dataset

The CICDDoS2019 dataset contains multiple parquet files for various DDoS attack types and benign traffic. Ensure both `*-training.parquet` and `*-testing.parquet` files are available in your `cicddos2019` directory.

## Contact

For questions or contributions, contact Ron Tetroashvili ([ront3t@gmail.com](mailto:ront3t@gmail.com)).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
