"""
deep_learning.py
Tugas: Pertemuan 12 - Deep Learning (INDIVIDU)
Nama : Muhammad Raka Razzani
NIM  : 434241056
Dataset: BlaBla.xlsx (Variabel Input: UMUR_TAHUN, B..M | Output: N)

Program ini membangun model Neural Network (Sequential MLP)
untuk klasifikasi biner pada dataset BlaBla.xlsx.
"""

# ========================================================================
# 1. IMPORT LIBRARY
# ========================================================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

if os.environ.get("DEEP_LEARNING_HEADLESS") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Set random seed untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ========================================================================
# 2. LOAD DATASET
# ========================================================================
base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
candidate_paths = [
    os.path.join(base_dir, "dataset-csv-input", "BlaBla.xlsx"),
    os.path.join(base_dir, "BlaBla.xlsx"),
]
data_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError(
        "File 'BlaBla.xlsx' tidak ditemukan di 'dataset-csv-input/' maupun direktori kerja saat ini."
    )

result_dir = os.path.join(base_dir, "dataset-csv-result")
os.makedirs(result_dir, exist_ok=True)

df = pd.read_excel(data_path)

# Buang kolom kosong 'Unnamed: ...' dan kolom formula Excel 'A'
# (Binning Umur akan dilakukan langsung di Python dari UMUR_TAHUN sesuai Langkah 4a modul)
unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
cols_to_drop = unnamed_cols + (["A"] if "A" in df.columns else [])
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Pastikan kolom target N bertipe integer (0 / 1)
df["N"] = pd.to_numeric(df["N"], errors="raise").astype(int)

print("=" * 70)
print("DEEP LEARNING - KLASIFIKASI BINER DATASET BlaBla.xlsx")
print("=" * 70)
print(f"\n[1] Dataset dimuat dari: {data_path}")
print(f"    Shape (setelah drop {cols_to_drop}): {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"    Kolom: {list(df.columns)}")
print("\n    Preview 5 baris pertama:")
print(df.head())


# ========================================================================
# 3. PISAHKAN FITUR (X) DAN LABEL (y)
# ========================================================================
# Variabel Input  : 13 fitur (UMUR_TAHUN, B, C, D, E, F, G, H, I, J, K, L, M)
# Variabel Output : N (target biner 0 / 1)

X = df.drop(columns=["N"]).copy()
y = df["N"].copy()
class_counts = y.value_counts().sort_index().to_dict()

print("\n[2] Pemisahan Fitur (X) dan Label (y)")
print(f"    X (fitur) : {X.shape} -> {X.shape[1]} fitur ({list(X.columns)})")
print(f"    y (label) : {y.shape} -> target biner N (0/1)")
print(f"\n    Distribusi kelas y: {class_counts}")
print(f"    Rasio (%): {y.value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()}")


# ========================================================================
# 4. PREPROCESSING DATA (MISSING VALUE, OUTLIER, BINNING UMUR)
# ========================================================================
print("\n[3] Preprocessing Data")

# a. Cek & Tangani Nilai Non-Numerik / Missing Value
print("\n    a. Cek Missing Value & Nilai Non-Numerik:")
non_numeric_age = pd.to_numeric(X["UMUR_TAHUN"], errors="coerce").isnull() & X["UMUR_TAHUN"].notnull()
if non_numeric_age.any():
    bad_vals = X.loc[non_numeric_age, "UMUR_TAHUN"].tolist()
    print(f"       Ditemukan {int(non_numeric_age.sum())} nilai non-numerik di UMUR_TAHUN: {bad_vals} -> diubah ke NaN")

X["UMUR_TAHUN"] = pd.to_numeric(X["UMUR_TAHUN"], errors="coerce")
missing = X.isnull().sum()
total_missing = int(missing.sum()) + int(y.isnull().sum())
print(f"       Total missing value pada X & y: {total_missing}")
if total_missing > 0:
    print(missing[missing > 0])
    X = X.fillna(X.median(numeric_only=True))
    print(f"       -> Missing value diisi dengan MEDIAN (UMUR_TAHUN median = {X['UMUR_TAHUN'].median():.1f})")

# b. Handle Outlier pada fitur numerik kontinu (UMUR_TAHUN) dengan IQR Clipping
# Catatan: Fitur B..M bernilai biner (0/1) dan konstanta (M=1), sehingga tidak dikenai IQR.
print("\n    b. Handle Outlier (IQR Clipping pada fitur numerik kontinu UMUR_TAHUN):")
q1 = X["UMUR_TAHUN"].quantile(0.25)
q3 = X["UMUR_TAHUN"].quantile(0.75)
iqr = q3 - q1
lower_bound = max(0.0, q1 - 1.5 * iqr)
upper_bound = q3 + 1.5 * iqr
n_outlier_age = int(((X["UMUR_TAHUN"] < lower_bound) | (X["UMUR_TAHUN"] > upper_bound)).sum())
X["UMUR_TAHUN"] = X["UMUR_TAHUN"].clip(lower_bound, upper_bound)
print(f"       UMUR_TAHUN : Q1={q1:.1f}, Q3={q3:.1f}, batas=[{lower_bound:.1f}, {upper_bound:.1f}] -> {n_outlier_age} outlier di-clip")
print("       Fitur B..M : SKIP IQR (sudah berupa fitur biner 0/1 & konstanta M=1)")

# Cek informasi baris identik (dipertahankan karena tidak ada kolom ID & fitur mayoritas biner survei)
n_identical = int(pd.concat([X, y], axis=1).duplicated().sum())
print(f"       Info pola identik (tanpa ID): {n_identical} baris (dipertahankan sesuai profil responden)")

# c. Kategori / Encoding Variabel Umur (Binning sesuai Langkah 4a modul)
# Aturan modul: 1 = <= 20 | 2 = 21 - 30 | 3 = 31 - 40 | 4 = 41 - 50 | 5 = > 50
print("\n[4] Kategori / Binning Variabel Umur (UMUR_TAHUN)")
print(f"    Usia minimum (tahun): {X['UMUR_TAHUN'].min():.0f}")
print(f"    Usia maksimum (tahun): {X['UMUR_TAHUN'].max():.0f}")

age_bins = [-np.inf, 20, 30, 40, 50, np.inf]
age_labels = [1, 2, 3, 4, 5]
X["UMUR_TAHUN"] = pd.cut(X["UMUR_TAHUN"], bins=age_bins, labels=age_labels, right=True).astype(int)

age_bin_names = {
    1: "<= 20",
    2: "21 - 30",
    3: "31 - 40",
    4: "41 - 50",
    5: "> 50",
}
age_dist = X["UMUR_TAHUN"].value_counts().reindex(age_labels, fill_value=0)
print("\n    Distribusi kelompok umur setelah binning:")
for code, count in age_dist.items():
    print(f"       Kategori {code} ({age_bin_names[code]:7s}): {count:5d} data")

# Simpan dataset bersih ke folder dataset-csv-result
df_clean = pd.concat([X, y], axis=1)
clean_csv_path = os.path.join(result_dir, "blabla_clean.csv")
df_clean.to_csv(clean_csv_path, index=False)
print(f"\n    -> Dataset bersih disimpan ke: {clean_csv_path}")

# Visualisasi ringkasan preprocessing (untuk dokumentasi laporan Langkah 4)
fig_prep, axes_prep = plt.subplots(1, 2, figsize=(13, 4.8))

bin_labels_plot = [f"{k}\n({age_bin_names[k]})" for k in age_labels]
bars_age = axes_prep[0].bar(bin_labels_plot, age_dist.values, color="#1565C0", alpha=0.85)
axes_prep[0].set_title("Distribusi Kategori Umur Setelah Binning")
axes_prep[0].set_xlabel("Kategori Umur")
axes_prep[0].set_ylabel("Jumlah Data")
axes_prep[0].grid(axis="y", alpha=0.3)
for bar in bars_age:
    h = int(bar.get_height())
    axes_prep[0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:,}", ha="center", va="bottom", fontsize=9)

target_labels_plot = ["Kelas 0", "Kelas 1"]
target_vals = [class_counts.get(0, 0), class_counts.get(1, 0)]
bars_target = axes_prep[1].bar(target_labels_plot, target_vals, color=["#2E7D32", "#C62828"], alpha=0.85)
axes_prep[1].set_title("Distribusi Kelas Target (N)")
axes_prep[1].set_xlabel("Kelas Target")
axes_prep[1].set_ylabel("Jumlah Data")
axes_prep[1].grid(axis="y", alpha=0.3)
for bar in bars_target:
    h = int(bar.get_height())
    axes_prep[1].text(bar.get_x() + bar.get_width() / 2, h, f"{h:,}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
prep_viz_path = os.path.join(result_dir, "viz_preprocessing_overview.png")
plt.savefig(prep_viz_path, dpi=120, bbox_inches="tight")
print(f"    -> Visualisasi preprocessing disimpan: {prep_viz_path}")
plt.show()


# ========================================================================
# 5. SPLIT DATA (80% TRAIN, 20% TEST) & FEATURE SCALING
# ========================================================================
# Split dilakukan sebelum StandardScaler untuk mencegah data leakage dari data uji
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("\n[5] Split Data 80:20 (stratified) & Feature Scaling (StandardScaler)")
print(f"    X_train : {X_train.shape}")
print(f"    X_test  : {X_test.shape}")
print(f"    y_train : {y_train.shape} -> {y_train.value_counts().sort_index().to_dict()}")
print(f"    y_test  : {y_test.shape} -> {y_test.value_counts().sort_index().to_dict()}")


# ========================================================================
# 6. BANGUN MODEL KLASIFIKASI (SEQUENTIAL MLP)
# ========================================================================
print("\n[6] Bangun Arsitektur Neural Network")

model = Sequential([
    Input(shape=(X_train.shape[1],)),   # Input layer: 13 fitur (UMUR_TAHUN, B..M)
    Dense(16, activation="relu"),       # Hidden Layer 1: 16 neuron, ReLU
    Dense(8, activation="relu"),        # Hidden Layer 2: 8 neuron, ReLU
    Dense(1, activation="sigmoid"),     # Output Layer: 1 neuron, Sigmoid
])

print("\n    Arsitektur Model:")
model.summary()


# ========================================================================
# 7. COMPILE DAN TRAIN MODEL
# ========================================================================
print("\n[7] Compile Model")
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print("    Optimizer : adam")
print("    Loss      : binary_crossentropy")
print("    Metrik    : accuracy")

print("\n[8] Training Model (100 epochs, validation_split=0.2, batch_size=32)")
history = model.fit(
    X_train,
    y_train.to_numpy(),
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=1,
)


# ========================================================================
# 8. EVALUASI MODEL
# ========================================================================
print("\n[9] Evaluasi Model pada Data Testing")
loss, acc = model.evaluate(X_test, y_test.to_numpy(), verbose=0)
print(f"    Loss               : {loss:.4f}")
print(f"    Akurasi di data uji: {acc * 100:.2f}%")


# ========================================================================
# 9. CONFUSION MATRIX
# ========================================================================
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
acc_check = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
print("\n[10] Confusion Matrix:")
print(f"     Accuracy Score (sklearn): {acc_check * 100:.2f}%")
print("                    Predicted 0   Predicted 1")
print(f"     Actual 0         {cm[0][0]:5d}         {cm[0][1]:5d}")
print(f"     Actual 1         {cm[1][0]:5d}         {cm[1][1]:5d}")


# ========================================================================
# 10. VISUALISASI CONFUSION MATRIX
# ========================================================================
plt.figure(figsize=(6, 4.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Kelas 0", "Kelas 1"],
    yticklabels=["Kelas 0", "Kelas 1"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Deep Learning (BlaBla.xlsx)")
plt.tight_layout()
cm_viz_path = os.path.join(result_dir, "viz_confusion_matrix.png")
plt.savefig(cm_viz_path, dpi=120, bbox_inches="tight")
print(f"\n[11] Confusion Matrix disimpan: {cm_viz_path}")
plt.show()


# ========================================================================
# 11. CLASSIFICATION REPORT & EKSPOR HASIL CSV DEEP LEARNING
# ========================================================================
print("\n[12] Classification Report:")
report_text = classification_report(
    y_test,
    y_pred,
    target_names=["Kelas 0", "Kelas 1"],
)
print(report_text)

# Simpan hasil prediksi pada data uji ke CSV
pred_df = X_test_raw.copy()
pred_df["N_actual"] = y_test.values
pred_df["N_pred_proba"] = np.round(y_pred_proba.flatten(), 6)
pred_df["N_pred"] = y_pred
pred_df["prediksi_tepat"] = (pred_df["N_actual"] == pred_df["N_pred"]).astype(int)
pred_csv_path = os.path.join(result_dir, "hasil_prediksi_deeplearning.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"    -> Hasil prediksi data uji disimpan: {pred_csv_path}")

# Simpan ringkasan metrik evaluasi ke CSV
report_dict = classification_report(
    y_test,
    y_pred,
    target_names=["Kelas 0", "Kelas 1"],
    output_dict=True,
)
eval_rows = []
for label, metrics in report_dict.items():
    if isinstance(metrics, dict):
        eval_rows.append({"kategori": label, **metrics, "nilai_global": np.nan})
eval_rows.append({"kategori": "test_loss", "nilai_global": loss})
eval_rows.append({"kategori": "test_accuracy", "nilai_global": acc})
eval_csv_path = os.path.join(result_dir, "evaluasi_deeplearning.csv")
pd.DataFrame(eval_rows).to_csv(eval_csv_path, index=False)
print(f"    -> Ringkasan evaluasi disimpan: {eval_csv_path}")

# Simpan riwayat training per epoch ke CSV
history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
history_csv_path = os.path.join(result_dir, "training_history_deeplearning.csv")
history_df.to_csv(history_csv_path, index=False)
print(f"    -> Riwayat training disimpan: {history_csv_path}")


# ========================================================================
# 12. VISUALISASI TRAINING HISTORY
# ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="#2E7D32")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#C62828")
axes[0].set_title("Model Accuracy per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(history.history["loss"], label="Train Loss", color="#2E7D32")
axes[1].plot(history.history["val_loss"], label="Validation Loss", color="#C62828")
axes[1].set_title("Model Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
hist_viz_path = os.path.join(result_dir, "viz_training_history.png")
plt.savefig(hist_viz_path, dpi=120, bbox_inches="tight")
print(f"\n[13] Training History disimpan: {hist_viz_path}")
plt.show()


# ========================================================================
# RINGKASAN AKHIR
# ========================================================================
print("\n" + "=" * 70)
print("  RINGKASAN HASIL DEEP LEARNING")
print("=" * 70)
print(f"  Dataset         : BlaBla.xlsx ({len(df)} baris x {X.shape[1]} fitur)")
print(f"  Arsitektur      : Sequential MLP ({X_train.shape[1]} -> 16 -> 8 -> 1)")
print("  Optimizer       : Adam")
print("  Loss Function   : Binary Crossentropy")
print("  Epochs          : 100")
print(f"  Akurasi Testing : {acc * 100:.2f}%")
print("\n  File Output (di folder dataset-csv-result/):")
print("    - blabla_clean.csv")
print("    - hasil_prediksi_deeplearning.csv")
print("    - evaluasi_deeplearning.csv")
print("    - training_history_deeplearning.csv")
print("    - viz_preprocessing_overview.png")
print("    - viz_confusion_matrix.png")
print("    - viz_training_history.png")
print("=" * 70 + "\n")
