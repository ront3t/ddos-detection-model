import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# =========== Imports for ADASYN and LightGBM ==========
from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier, early_stopping
# ======================================================

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1) DATA LOADING
# ----------------------------------------------------------------------------
cicddos2019_path = r'C:\Users\97250\Desktop\RonT\MBA\סמינריון\cicddos2019'

dfps_train = []
dfps_test = []

for dirname, _, filenames in os.walk(cicddos2019_path):
    for filename in filenames:
        if filename.endswith('-training.parquet'):
            dfp = os.path.join(dirname, filename)
            dfps_train.append(dfp)
            print("Training File:", dfp)
        elif filename.endswith('-testing.parquet'):
            dfp = os.path.join(dirname, filename)
            dfps_test.append(dfp)
            print("Testing File:", dfp)

train_prefixes = [os.path.basename(dfp).split('-')[0] for dfp in dfps_train]
test_prefixes = [os.path.basename(dfp).split('-')[0] for dfp in dfps_test]
common_prefixes = list(set(train_prefixes).intersection(test_prefixes))

dfps_train = [dfp for dfp in dfps_train if os.path.basename(dfp).split('-')[0] in common_prefixes]
dfps_test = [dfp for dfp in dfps_test if os.path.basename(dfp).split('-')[0] in common_prefixes]

train_df = pd.concat([pd.read_parquet(dfp) for dfp in dfps_train], ignore_index=True)
test_df = pd.concat([pd.read_parquet(dfp) for dfp in dfps_test], ignore_index=True)

print("Shapes of Train and Test Data:", train_df.shape, test_df.shape)
print(train_df.head())
train_df.info()

# ----------------------------------------------------------------------------
# 2) BASIC DATA CHECKS AND LABEL MAPPING
# ----------------------------------------------------------------------------
print("\nTraining Data Label Distribution:")
print(train_df["Label"].value_counts())

print("\nTesting Data Label Distribution:")
print(test_df["Label"].value_counts())

# Drop the WebDDoS class from the testing data because it isn't in training data
test_df = test_df[test_df["Label"] != "WebDDoS"]

# Map labels to a consistent format
label_mapping = {
    'DrDoS_UDP': 'UDP',
    'UDP-lag': 'UDPLag',
    'DrDoS_MSSQL': 'MSSQL',
    'DrDoS_LDAP': 'LDAP',
    'DrDoS_NetBIOS': 'NetBIOS',
    'Syn': 'Syn',
    'Benign': 'Benign'
}
test_df["Label"] = test_df["Label"].map(label_mapping)

print("\nUpdated Testing Data Label Distribution:")
print(test_df["Label"].value_counts())


# ----------------------------------------------------------------------------
# 3) EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------------
def perform_eda(df):
    print("\n=== Exploratory Data Analysis ===")
    # Display basic descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe(include='all').transpose())

    # Distribution of target variable
    plt.figure(figsize=(8, 4))
    sns.countplot(x="Label", data=df, palette='viridis')
    plt.title("Class Distribution in Dataset")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Histograms for numerical features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols].hist(bins=20, figsize=(14, 10))
    plt.suptitle("Histograms of Numerical Features")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap for numerical features
    plt.figure(figsize=(12, 10))
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.show()


# Run EDA on training data
perform_eda(train_df)


# ----------------------------------------------------------------------------
# 4) COLUMN ANALYSIS
# ----------------------------------------------------------------------------
def grab_col_names(data, cat_th=10, car_th=20):
    cat_cols = [col for col in data.columns if data[col].dtypes == "O"]
    num_but_cat = [col for col in data.columns if data[col].nunique() < cat_th and data[col].dtypes != "O"]
    high_card_cat_cols = [col for col in data.columns if data[col].nunique() > car_th and data[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in high_card_cat_cols]

    num_cols = [col for col in data.columns if data[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {data.shape[0]}")
    print(f"Variables: {data.shape[1]}")
    print(f"Categorical Columns: {len(cat_cols)}")
    print(f"Numerical Columns: {len(num_cols)}")
    print(f"High Cardinality Categorical Columns: {len(high_card_cat_cols)}")
    print(f"Number but Categorical Columns: {len(num_but_cat)}\n")

    return cat_cols, num_cols, high_card_cat_cols


cat_cols, num_cols, high_card_cat_cols = grab_col_names(train_df)
print(f"Categorical Columns: {cat_cols}")
print(f"Numerical Columns: {num_cols}")
print(f"High Cardinality Categorical Columns: {high_card_cat_cols}")

# Check columns with only one unique value
single_unique = [c for c in train_df.columns if train_df[c].nunique() == 1]
print(f"\nColumns with only one unique value: {single_unique}")

# Missing values
print(f"\nTotal number of missing values in train_df: {train_df.isnull().sum().sum()}")

# Duplicate Rows
print(f"Number of Duplicate Rows in train_df: {train_df.duplicated().sum()}")

train_df = train_df.drop_duplicates()

# ----------------------------------------------------------------------------
# 5) FEATURE SELECTION
# ----------------------------------------------------------------------------
# Remove non-informative single-value columns
train_df.drop(single_unique, axis=1, inplace=True)
test_df.drop(single_unique, axis=1, inplace=True)

print("Shape after removing single-value columns:", train_df.shape, test_df.shape)

# Remove highly correlated features
numerical_df = train_df.select_dtypes(include=[np.number])
corr_matrix = numerical_df.corr().abs()
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper_triangle = corr_matrix.where(mask)
high_corr_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.8)]

print(f"Total number of highly correlated columns: {len(high_corr_cols)}")
print("Highly correlated columns are:", high_corr_cols)

train_df.drop(high_corr_cols, axis=1, inplace=True)
test_df.drop(high_corr_cols, axis=1, inplace=True)
print("Shape after removing highly correlated columns:", train_df.shape, test_df.shape)

# ----------------------------------------------------------------------------
# 6) TRAIN-VAL-TEST SPLIT
# ----------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop("Label", axis=1),
    train_df["Label"],
    test_size=0.2,
    random_state=42
)
X_test, y_test = test_df.drop("Label", axis=1), test_df["Label"]

# ----------------------------------------------------------------------------
# 7) ENCODE TARGET
# ----------------------------------------------------------------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

label_map = {index: label for index, label in enumerate(le.classes_)}
print("Label Map:", label_map)

# ----------------------------------------------------------------------------
# 8) OUTPUT CLASS DISTRIBUTION BEFORE ADASYN
# ----------------------------------------------------------------------------
unique_vals, counts_vals = np.unique(y_train, return_counts=True)
plt.figure(figsize=(8, 4))
class_labels = [label_map[u] for u in unique_vals]
plt.bar(class_labels, counts_vals, color='skyblue')
plt.title("Label Distribution BEFORE ADASYN")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nBefore ADASYN Sampling - Class Distribution in y_train:")
for u, c in zip(unique_vals, counts_vals):
    print(f"Class {u} ({label_map[u]}): {c} samples")

# ----------------------------------------------------------------------------
# 9) ADASYN FOR CLASS IMBALANCE
# ----------------------------------------------------------------------------
adasyn = ADASYN()
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

# Output class distribution AFTER ADASYN
unique_after, counts_after = np.unique(y_train_res, return_counts=True)
plt.figure(figsize=(8, 4))
class_labels_after = [label_map[u] for u in unique_after]
plt.bar(class_labels_after, counts_after, color='lightgreen')
plt.title("Label Distribution AFTER ADASYN")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAfter ADASYN Sampling - Class Distribution in y_train_res:")
for u, c in zip(unique_after, counts_after):
    print(f"Class {u} ({label_map[u]}): {c} samples")

# ----------------------------------------------------------------------------
# 10) FEATURE SCALING
# ----------------------------------------------------------------------------
scaler = MinMaxScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ----------------------------------------------------------------------------
# TRAIN & EVALUATE MODELS
# ----------------------------------------------------------------------------
def train_model(X_train, X_val, y_train, y_val):
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            random_state=42
        ),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(30,), max_iter=100),
    }

    scores_list = []
    plt.figure(figsize=(10, 8))

    for name, model in tqdm(classifiers.items(), desc="Training Models"):
        print(f"\nTraining {name}...")

        start_time = time.time()
        if name == "LightGBM":
            print("early stopping is here")
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[early_stopping(stopping_rounds=10, verbose=True)]
            )
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_val, y_proba, multi_class="ovr")

        cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=3))

        for i in range(len(np.unique(y_val))):
            fpr, tpr, _ = roc_curve(y_val, y_proba[:, i], pos_label=i)
            plt.plot(fpr, tpr, label=f'{name} - Class {i} (AUC={roc_auc:.4f})')

        scores_list.append({
            "Model": name,
            "Training Time (s)": train_time,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "CV Score": cv_score
        })

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models (Multiclass)')
    plt.legend(loc='lower right')
    plt.show()

    return pd.DataFrame(scores_list)


scores = train_model(X_train_res_scaled, X_val_scaled, y_train_res, y_val)
print("\nEvaluation Results:")
print(scores[[
    "Model", "Training Time (s)", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "CV Score"
]])


def visualize_scores(df):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, metric in enumerate(metrics):
        axes[i].bar(df["Model"], df[metric], color='skyblue')
        axes[i].set_title(metric)
        axes[i].set_xticklabels(df["Model"], rotation=45)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel(metric)

    plt.tight_layout()
    plt.show()


visualize_scores(scores)


# ----------------------------------------------------------------------------
# SIMULATE REAL-TIME INFERENCE (No Confusion Matrix)
# ----------------------------------------------------------------------------
def simulate_real_time_inference(model, X, y, label_map, interval=1.0, n_samples=10):
    print(
        f"\n--- Simulating Real-Time Inference (LightGBM) with interval={interval}s for {n_samples} random samples ---")
    total_data = X.shape[0]

    rand_indices = np.random.choice(total_data, n_samples, replace=False)

    true_list = []
    pred_list = []

    for i, idx in enumerate(rand_indices):
        sample_features = X[idx].reshape(1, -1)
        pred_class_idx = model.predict(sample_features)[0]
        pred_class_label = label_map[pred_class_idx]
        true_label = label_map[y[idx]]

        true_list.append(true_label)
        pred_list.append(pred_class_label)

        print(f"Sample {i} (Index {idx}): True={true_label}, Pred={pred_class_label}")
        time.sleep(interval)

    correct = sum(t == p for t, p in zip(true_list, pred_list))
    accuracy_rt = correct / n_samples

    print(f"\nReal-Time Simulation Summary ({n_samples} samples):")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy_rt:.2f}")
    print("\n(No confusion matrix generated)")


# ----------------------------------------------------------------------------
# RETRAIN LIGHTGBM FOR REAL-TIME SIMULATION DEMO
# ----------------------------------------------------------------------------
print("\n--- Retraining LightGBM for Real-Time Simulation Demo ---")
lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=10,
    random_state=42
)
lgb_model.fit(
    X_train_res_scaled,
    y_train_res,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=10, verbose=True)]
)

simulate_real_time_inference(
    model=lgb_model,
    X=X_test_scaled,
    y=y_test,
    label_map=label_map,
    interval=1.0,
    n_samples=100
)

# ----------------------------------------------------------------------------
# OUTPUT FINAL DATA SUMMARY
# ----------------------------------------------------------------------------
print("\n--- DATA SUMMARY ---")
print(f"Original Training Dataset: {train_df.shape[0]} rows, {train_df.shape[1]} fields")
print(f"Original Test Dataset: {test_df.shape[0]} rows, {test_df.shape[1]} fields")

print("\nLabel Distribution BEFORE ADASYN:")
unique_vals, counts_vals = np.unique(y_train, return_counts=True)
for u, c in zip(unique_vals, counts_vals):
    print(f"Class {label_map[u]}: {c} samples")

print("\nLabel Distribution AFTER ADASYN:")
unique_after, counts_after = np.unique(y_train_res, return_counts=True)
for u, c in zip(unique_after, counts_after):
    print(f"Class {label_map[u]}: {c} samples")

total_rows = train_df.shape[0] + test_df.shape[0]
train_pct = len(X_train) / total_rows * 100
val_pct = len(X_val) / total_rows * 100
test_pct = test_df.shape[0] / total_rows * 100
print("\nSplit Percentages (of entire dataset):")
print(f"Training set: {train_pct:.2f}%")
print(f"Validation set: {val_pct:.2f}%")
print(f"Test set: {test_pct:.2f}%")

print("\n--- END OF SCRIPT ---")
