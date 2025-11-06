# General Imports
import numpy as np
import pandas as pd
import sys
import math
import warnings
warnings.filterwarnings('ignore')

# Visualisation and graphs Imports
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# PennyLane Imports
import pennylane as qml
from pennylane.templates import AmplitudeEmbedding, AngleEmbedding

print("All packages imported successfully!")

# Load your gene expression data
print("\n=== Loading Gene Expression Data ===")

# Load the transposed data (patients as rows, genes as columns)
try:
    # Try loading the transposed CSV
    df = pd.read_csv('transposed_50_genes_22_patients.csv', index_col=0)
    print("Loaded transposed 50 genes data successfully!")
    print(f"Data shape: {df.shape}")
    print(f"Patients: {df.shape[0]}, Genes: {df.shape[1]}")
    
except FileNotFoundError:
    # If transposed file doesn't exist, load the original and transpose it
    print("Transposed file not found, loading original data...")
    df = pd.read_csv('filtered_50_genes_22_patients.csv')
    # Set Gene Description as index and transpose
    df = df.set_index('Gene Description').T
    # Remove the accession row if it exists
    if 'Gene Accession Number' in df.index:
        df = df.drop('Gene Accession Number')
    print(f"Transposed data shape: {df.shape}")

print("\nData overview:")
print(df.head())
print(f"\nData types: {df.dtypes.unique()}")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Load real patient labels
print("\n=== Loading Real Patient Labels ===")

# Your actual patient labels (ALL = 0, AML = 1)
real_labels_dict = {
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
    11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0,
    21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 1, 29: 1, 30: 1,
    31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 0, 40: 0,
    41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 1,
    51: 1, 52: 1, 53: 1, 54: 1, 55: 0, 56: 0, 57: 1, 58: 1, 59: 0, 60: 1,
    61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 1, 67: 0, 68: 0, 69: 0, 70: 0,
    71: 0, 72: 0
}

# Map patient IDs to your current dataset
current_patient_ids = df.index.astype(int)
labels = [real_labels_dict[pid] for pid in current_patient_ids]
labels = np.array(labels)

print(f"Current patient IDs: {list(current_patient_ids)}")
print(f"Real labels: {labels}")
print(f"Class distribution: ALL (0): {np.sum(labels == 0)}, AML (1): {np.sum(labels == 1)}")

# Use these as your features and labels
features = df.values.astype(float)  # Gene expression values

print(f"\nFinal dataset shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Data Exploration and Visualization
print("\n=== Data Exploration ===")

# Create first visualization: Gene Expression Analysis
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
# Plot distribution of a few representative genes
gene_indices = [0, 12, 25, 37]  # Sample across the 50 genes
gene_names = [df.columns[i][:20] + '...' for i in gene_indices]

for idx, name in zip(gene_indices, gene_names):
    plt.hist(features[:, idx], alpha=0.7, bins=20, label=name)
plt.xlabel('Expression Value')
plt.ylabel('Frequency')
plt.title('Distribution of Gene Expression Values\n(50 Genes)')
plt.legend()

plt.subplot(2, 2, 2)
# FULL Correlation heatmap of ALL 50 genes
print("Generating full correlation matrix for 50 genes...")
correlation_matrix = np.corrcoef(features.T)

# Create shortened gene names for readability
short_gene_names = []
for i, gene_name in enumerate(df.columns):
    # Extract meaningful parts of gene names
    if len(gene_name) > 20:
        # Try to keep the most descriptive part
        parts = gene_name.split()
        if len(parts) > 2:
            short_name = ' '.join(parts[:2]) + '...'
        else:
            short_name = gene_name[:18] + '...'
    else:
        short_name = gene_name
    short_gene_names.append(short_name)

# Create the heatmap
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Gene-Gene Correlation Matrix\n(All 50 Genes)')
plt.xlabel('Genes')
plt.ylabel('Genes')
# Note: With 50 genes, individual labels would be too crowded

plt.subplot(2, 2, 3)
# Expression by class for representative genes
# Find genes with high differentiation between classes
all_indices = np.where(labels == 0)[0]
aml_indices = np.where(labels == 1)[0]

# Calculate t-statistics to find most differentiating genes
t_stats = []
for i in range(features.shape[1]):
    all_vals = features[all_indices, i]
    aml_vals = features[aml_indices, i]
    t_stat = (np.mean(aml_vals) - np.mean(all_vals)) / (np.std(np.concatenate([all_vals, aml_vals])) + 1e-8)
    t_stats.append(t_stat)

t_stats = np.array(t_stats)
most_all_favoring = np.argmin(t_stats)  # Most negative t-statistic
most_aml_favoring = np.argmax(t_stats)  # Most positive t-statistic

for class_label, class_name in [(0, 'ALL'), (1, 'AML')]:
    class_data = features[labels == class_label]
    plt.scatter(class_data[:, most_all_favoring], class_data[:, most_aml_favoring], 
                alpha=0.6, label=f'{class_name}', s=50)
plt.xlabel(f'{short_gene_names[most_all_favoring]}\n(ALL-favoring)')
plt.ylabel(f'{short_gene_names[most_aml_favoring]}\n(AML-favoring)')
plt.title('Most Differentiating Genes by Cancer Type')
plt.legend()

plt.subplot(2, 2, 4)
# Box plot of gene expression by class for most differentiating genes
all_data = features[labels == 0]
aml_data = features[labels == 1]

plt.boxplot([all_data[:, most_all_favoring], aml_data[:, most_all_favoring], 
             all_data[:, most_aml_favoring], aml_data[:, most_aml_favoring]], 
            labels=['ALL\n(ALL-gene)', 'AML\n(ALL-gene)', 'ALL\n(AML-gene)', 'AML\n(AML-gene)'])
plt.xlabel('Cancer Type and Gene Preference')
plt.ylabel('Expression Value')
plt.title('Expression of Most Differentiating Genes')

plt.tight_layout()
plt.savefig('gene_expression_analysis_50_genes.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Data exploration visualizations completed successfully!")

# Dataset Preprocessing - USE ALL 50 GENES
print("\n=== Data Preprocessing ===")

# Scale the features to [0, 1] range for quantum processing
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

print(f"Using all {features.shape[1]} genes")
print(f"Features shape: {features_scaled.shape}")
print("✓ Data preprocessing completed successfully!")

# Split dataset
print("\n=== Dataset Splitting ===")
# Use the scaled features for processing (all 50 genes)
X = features_scaled
y = labels

train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, train_size=0.75, random_state=42, stratify=y
)

print(f"Training set size: {len(train_features)}")
print(f"Test set size: {len(test_features)}")
print(f"Training class distribution - ALL: {np.sum(train_labels == 0)}, AML: {np.sum(train_labels == 1)}")
print(f"Test class distribution - ALL: {np.sum(test_labels == 0)}, AML: {np.sum(test_labels == 1)}")

# Classical SVM Benchmark
print("\n=== Classical SVM Benchmark ===")
svc = SVC(random_state=42)
svc.fit(train_features, train_labels)

train_score_classical = svc.score(train_features, train_labels)
test_score_classical = svc.score(test_features, test_labels)

print(f"Classical SVC on training dataset: {train_score_classical:.3f}")
print(f"Classical SVC on test dataset: {test_score_classical:.3f}")

# Cross-validation for more reliable results
print("\n=== Cross-Validation Analysis ===")
cv_scores_classical = cross_val_score(svc, X, y, cv=5, scoring='accuracy')
print(f"Classical SVC - 5-fold CV scores: {cv_scores_classical}")
print(f"Classical SVC - Mean CV accuracy: {cv_scores_classical.mean():.3f} (+/- {cv_scores_classical.std() * 2:.3f})")

# Classical kernels comparison
print("\n=== Classical Kernels Comparison ===")
classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

classical_results = []
classical_models = []

for kernel in classical_kernels:
    classical_svc = SVC(kernel=kernel, random_state=42)
    classical_svc.fit(train_features, train_labels)
    classical_score = classical_svc.score(test_features, test_labels)
    classical_results.append(classical_score)
    classical_models.append(classical_svc)
    print(f'{kernel} kernel test score: {classical_score:.3f}')

# PennyLane Quantum Kernels - USING AMPLITUDE AND ANGLE EMBEDDING
print("\n=== PennyLane Quantum Kernels ===")
print("NOTE: Using amplitude embedding AND angle encoding")
print(f"Using both encoding methods with {features.shape[1]} genes")


# Replace the kernel matrix calculation sections with these versions:

def amplitude_embedding_pennylane_all_features_full():
    """Amplitude embedding that uses all features - FULL MATRICES"""
    qubits = 6
    dev = qml.device("default.qubit", wires=qubits)
    projector = np.zeros((2**qubits, 2**qubits))
    projector[0, 0] = 1
    
    @qml.qnode(dev)
    def kernel(x1, x2):
        x1_normalized = x1 / np.linalg.norm(x1)
        x2_normalized = x2 / np.linalg.norm(x2)
        
        AmplitudeEmbedding(x1_normalized, wires=range(qubits), normalize=True, pad_with=0.0)
        qml.adjoint(AmplitudeEmbedding)(x2_normalized, wires=range(qubits), normalize=True, pad_with=0.0)
        return qml.expval(qml.Hermitian(projector, wires=range(qubits)))
    
    def kernel_matrix(A, B):
        # USE FULL MATRICES - no reduction
        nA = len(A)
        nB = len(B)
        
        K = np.zeros((nA, nB))
        print(f"Computing FULL amplitude kernel matrix: {nA} x {nB}")
        
        for i in range(nA):
            if i % 5 == 0:  # Progress indicator
                print(f"  Progress: {i+1}/{nA}")
            for j in range(nB):
                try:
                    K[i, j] = kernel(A[i], B[j])
                except Exception as e:
                    print(f"Error at position ({i},{j}): {e}")
                    K[i, j] = 0.0
        return K
    
    return kernel_matrix

def angle_embedding_pennylane_all_features_full():
    """Angle embedding that uses all features - FULL MATRICES"""
    qubits = min(8, features.shape[1])
    dev = qml.device("default.qubit", wires=qubits)
    projector = np.zeros((2**qubits, 2**qubits))
    projector[0, 0] = 1
    
    @qml.qnode(dev)
    def kernel(x1, x2):
        x1_subset = x1[:qubits]
        x2_subset = x2[:qubits]
        x1_scaled = x1_subset * np.pi
        x2_scaled = x2_subset * np.pi
        
        AngleEmbedding(x1_scaled, wires=range(qubits), rotation='Y')
        qml.adjoint(AngleEmbedding)(x2_scaled, wires=range(qubits), rotation='Y')
        return qml.expval(qml.Hermitian(projector, wires=range(qubits)))
    
    def kernel_matrix(A, B):
        # USE FULL MATRICES - no reduction
        nA = len(A)
        nB = len(B)
        
        K = np.zeros((nA, nB))
        print(f"Computing FULL angle embedding kernel matrix: {nA} x {nB}")
        print(f"Using {qubits} qubits for angle encoding")
        
        for i in range(nA):
            if i % 5 == 0:  # Progress indicator
                print(f"  Progress: {i+1}/{nA}")
            for j in range(nB):
                try:
                    K[i, j] = kernel(A[i], B[j])
                except Exception as e:
                    print(f"Error at position ({i},{j}): {e}")
                    K[i, j] = 0.0
        return K
    
    return kernel_matrix

# Then update the calculation section:
print("\nCalculating FULL amplitude AND angle embedding kernel matrices...")

# Use FULL datasets (no reduction)
train_full = train_features  # Use all 16 training samples
test_full = test_features    # Use all 6 test samples
train_labels_full = train_labels
test_labels_full = test_labels

print(f"Quantum processing with FULL datasets: {len(train_full)} training, {len(test_full)} test samples")

# Amplitude embedding kernel - FULL
print("Computing FULL amplitude embedding kernel...")
amplitude_kernel_full = amplitude_embedding_pennylane_all_features_full()
matrix_train_amp_full = amplitude_kernel_full(train_full, train_full)
matrix_test_amp_full = amplitude_kernel_full(test_full, train_full)

print(f"✓ FULL Amplitude embedding kernel matrices calculated: {matrix_train_amp_full.shape}, {matrix_test_amp_full.shape}")

# Angle embedding kernel - FULL  
print("Computing FULL angle embedding kernel...")
angle_kernel_full = angle_embedding_pennylane_all_features_full()
matrix_train_angle_full = angle_kernel_full(train_full, train_full)
matrix_test_angle_full = angle_kernel_full(test_full, train_full)

print(f"✓ FULL Angle embedding kernel matrices calculated: {matrix_train_angle_full.shape}, {matrix_test_angle_full.shape}")





# Visualize kernel matrices
print("\nGenerating kernel matrix visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Amplitude embedding kernels
im1 = axes[0, 0].imshow(matrix_train_amp_full, interpolation='nearest', origin='upper', cmap='Oranges_r')
axes[0, 0].set_title("Amplitude Embedding - Training Kernel\n(50 Genes, 6 Qubits)")
axes[0, 0].set_xlabel('Training Samples')
axes[0, 0].set_ylabel('Training Samples')
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(matrix_test_amp_full, interpolation='nearest', origin='upper', cmap='Purples_r')
axes[0, 1].set_title("Amplitude Embedding - Testing Kernel\n(50 Genes, 6 Qubits)")
axes[0, 1].set_xlabel('Training Samples')
axes[0, 1].set_ylabel('Test Samples')
plt.colorbar(im2, ax=axes[0, 1])

# Angle embedding kernels
im3 = axes[1, 0].imshow(matrix_train_angle_full, interpolation='nearest', origin='upper', cmap='Blues_r')
axes[1, 0].set_title("Angle Embedding - Training Kernel\n(8 Genes, 8 Qubits)")
axes[1, 0].set_xlabel('Training Samples')
axes[1, 0].set_ylabel('Training Samples')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(matrix_test_angle_full, interpolation='nearest', origin='upper', cmap='Greens_r')
axes[1, 1].set_title("Angle Embedding - Testing Kernel\n(8 Genes, 8 Qubits)")
axes[1, 1].set_xlabel('Training Samples')
axes[1, 1].set_ylabel('Test Samples')
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('kernel_matrices_50_genes.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Kernel matrix visualizations completed!")

# Quantum SVM Evaluation
print("\n=== Quantum SVM Evaluation ===")

def evaluate_qsvm(train_kernel, train_labels, test_kernel, test_labels, kernel_name):
    """Evaluate QSVM with given kernel matrix"""
    qsvc = SVC(kernel='precomputed')
    qsvc.fit(train_kernel, train_labels)
    train_score = qsvc.score(train_kernel, train_labels)
    test_score = qsvc.score(test_kernel, test_labels)
    
    print(f"{kernel_name}:")
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Test accuracy:  {test_score:.3f}")
    
    # Additional metrics
    y_pred = qsvc.predict(test_kernel)
    report = classification_report(test_labels, y_pred, output_dict=True, zero_division=0)
    print(f"  Precision: {report['weighted avg']['precision']:.3f}")
    print(f"  Recall:    {report['weighted avg']['recall']:.3f}")
    
    return train_score, test_score, qsvc

# Evaluate both quantum kernels
print("\nQuantum Kernel Performance:")
amp_train, amp_test, amp_svc = evaluate_qsvm(matrix_train_amp_full, train_labels_full, 
                                             matrix_test_amp_full, test_labels_full, 
                                             "Amplitude Embedding")

angle_train, angle_test, angle_svc = evaluate_qsvm(matrix_train_angle_full, train_labels_full, 
                                                   matrix_test_angle_full, test_labels_full, 
                                                   "Angle Encoding")

# DIAGNOSTIC ANALYSIS
print("\n" + "="*60)
print("DIAGNOSTIC ANALYSIS")
print("="*60)

# Check what predictions are actually being made
print("\n=== Detailed Prediction Analysis ===")

# Majority class baseline
majority_class = np.argmax(np.bincount(train_labels))
majority_predictions = np.full_like(test_labels_full, majority_class)
majority_accuracy = np.mean(majority_predictions == test_labels_full)
print(f"Majority class baseline (always predict {['ALL', 'AML'][majority_class]}): {majority_accuracy:.3f}")

# Check individual model predictions
models_info = [
    ('Classical SVC', svc),
    ('Amplitude QSVM', amp_svc),
    ('Angle QSVM', angle_svc)
]

for name, model in models_info:
    if name == 'Classical SVC':
        predictions = model.predict(test_full)
    elif name == 'Amplitude QSVM':
        predictions = model.predict(matrix_test_amp_full)
    else:  # Angle QSVM
        predictions = model.predict(matrix_test_angle_full)
    correct = predictions == test_labels_full
    print(f"\n{name}:")
    print(f"  Predictions: {predictions} ({['ALL', 'AML'][predictions[0]]} etc.)")
    print(f"  Actual:      {test_labels_full} ({['ALL', 'AML'][test_labels_full[0]]} etc.)")
    print(f"  Correct:     {correct}")
    print(f"  Accuracy:    {np.mean(correct):.3f}")

# Confusion matrices
print("\n=== Confusion Matrices ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(models_info):
    if name == 'Classical SVC':
        predictions = model.predict(test_full)
    elif name == 'Amplitude QSVM':
        predictions = model.predict(matrix_test_amp_full)
    else:  # Angle QSVM
        predictions = model.predict(matrix_test_angle_full)
    cm = confusion_matrix(test_labels_full, predictions)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues',
                xticklabels=['ALL', 'AML'], yticklabels=['ALL', 'AML'])
    accuracy = np.mean(predictions == test_labels_full)
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices_50_genes.png', dpi=300, bbox_inches='tight')
plt.show()

# Results Visualization
print("\nGenerating results comparison...")

# Performance comparison bar plot
methods = ['Amplitude QSVM', 'Angle QSVM'] + [f'Classical {k}' for k in classical_kernels]
scores = [amp_test, angle_test] + classical_results
colors = ['#A23B72', '#2E86AB'] + ['#F18F01'] * 4

plt.figure(figsize=(14, 6))
bars = plt.bar(methods, scores, color=colors, alpha=0.7)
plt.axhline(y=test_score_classical, color='red', linestyle='--', 
            label=f'Classical SVC Baseline: {test_score_classical:.3f}')
plt.axhline(y=majority_accuracy, color='green', linestyle='--',
            label=f'Majority Class Baseline: {majority_accuracy:.3f}')
plt.ylabel('Test Accuracy')
plt.title('Gene Expression: Quantum vs Classical SVM Performance\n(ALL vs AML Classification - 50 Genes)')
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.legend()

# Add value labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('performance_comparison_50_genes.png', dpi=300, bbox_inches='tight')
plt.show()

# Decision boundary visualization
print("\nGenerating decision boundary visualizations...")

def plot_decision_boundary(clf, X, y, title, ax):
    """Plot decision boundary for classifier"""
    # For visualization, use first 2 features if available
    if X.shape[1] >= 2:
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        if hasattr(clf, 'kernel') and clf.kernel == 'precomputed':
            # For QSVM with precomputed kernels, we can't easily plot decision boundaries
            ax.text(0.5, 0.5, "Decision boundary\nnot available for\nprecomputed kernels", 
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black', s=60)
        else:
            # For classical SVM
            if X.shape[1] > 2:
                # Pad with feature means for other dimensions
                mesh_full = np.tile(np.mean(X, axis=0), (len(mesh_points), 1))
                mesh_full[:, :2] = mesh_points
            else:
                mesh_full = mesh_points
                
            Z = clf.predict(mesh_full)
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black', s=60)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        return scatter
    else:
        ax.text(0.5, 0.5, "Cannot plot decision boundary\nfor 1D data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return None

# Plot decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Classical SVM decision boundary
scatter1 = plot_decision_boundary(svc, train_full, train_labels_full, 
                                 f'Classical SVM (Test Acc: {test_score_classical:.3f})', axes[0])

# Amplitude embedding info
axes[1].text(0.5, 0.5, f"Amplitude QSVM\nTest Accuracy: {amp_test:.3f}\n• 50 genes → 6 qubits\n• 2^6 = 64 amplitude encoding\n• Precomputed kernel\n• Decision boundary not available", 
             ha='center', va='center', transform=axes[1].transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
axes[1].set_title('Amplitude Embedding QSVM')

# Angle embedding info
axes[2].text(0.5, 0.5, f"Angle QSVM\nTest Accuracy: {angle_test:.3f}\n• 8 genes → 8 qubits\n• Y-rotation encoding\n• Direct feature mapping\n• Decision boundary not available", 
             ha='center', va='center', transform=axes[2].transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
axes[2].set_title('Angle Encoding QSVM')

plt.tight_layout()
plt.savefig('decision_boundaries_50_genes.png', dpi=300, bbox_inches='tight')
plt.show()

# Final Summary
print("\n" + "="*60)
print("FINAL SUMMARY - GENE EXPRESSION ANALYSIS (50 GENES)")
print("="*60)
print(f"Dataset: {df.shape[0]} patients, {df.shape[1]} genes")
print(f"Cancer types - ALL: {np.sum(labels == 0)}, AML: {np.sum(labels == 1)}")
print(f"Quantum Implementations:")
print(f"  - Amplitude Embedding (6 qubits, 50 genes)")
print(f"  - Angle Encoding (8 qubits, 8 genes)")
print("\nPerformance Results:")
print(f"Majority Class Baseline: {majority_accuracy:.3f}")
print(f"Classical SVM Baseline: {test_score_classical:.3f}")
print(f"Amplitude Embedding QSVM: {amp_test:.3f}")
print(f"Angle Encoding QSVM: {angle_test:.3f}")
print(f"Best Classical Kernel: {classical_kernels[np.argmax(classical_results)]} ({np.max(classical_results):.3f})")

print("\nPerformance Comparison:")
if max(amp_test, angle_test) > majority_accuracy:
    print("✓ At least one quantum method beats majority class baseline!")
else:
    print("✗ Quantum methods perform at baseline level")

if max(amp_test, angle_test) >= test_score_classical:
    print("✓ At least one quantum method matches or exceeds classical performance!")
else:
    print("✗ Quantum methods underperform classical")

print(f"\nQuantum Advantage: {'YES' if max(amp_test, angle_test) > test_score_classical else 'NO'}")

print("\nEncoding Comparison:")
print(f"Amplitude vs Angle: {'Amplitude better' if amp_test > angle_test else 'Angle better' if angle_test > amp_test else 'Equal performance'}")

print("\nAll visualizations have been saved:")
print("1. gene_expression_analysis_50_genes.png - Data exploration")
print("2. quantum_circuits_50_genes.png - Circuit diagrams") 
print("3. kernel_matrices_50_genes.png - Quantum kernel matrices")
print("4. performance_comparison_50_genes.png - Accuracy comparison")
print("5. decision_boundaries_50_genes.png - Model decision regions")
print("6. confusion_matrices_50_genes.png - Prediction performance")
print("7. (Cross-validation results in console)")

# Save results to file
results_df = pd.DataFrame({
    'Method': methods,
    'Test_Accuracy': scores,
    'Quantum': [True, True] + [False] * len(classical_kernels),
    'Encoding_Type': ['Amplitude', 'Angle'] + ['Classical'] * len(classical_kernels)
})
results_df.to_csv('quantum_svm_50_genes_results.csv', index=False)
print(f"\nResults saved to: quantum_svm_50_genes_results.csv")