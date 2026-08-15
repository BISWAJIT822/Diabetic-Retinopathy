import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Load EfficientNet-B3 (Native 300x300 resolution)
base_model = EfficientNetB3(
    weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False

# 2. Build the top layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')  # Your 5 DR stages
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Setup Data Flow (Point this to your local folder)
train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DRdataset_Processed')) 

if not os.path.exists(train_dir):
    print(f"ERROR: The path {train_dir} does not exist. Please check your folder structure.")
    exit()

# 1. Setup the Generator with a 20% split for validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 2. This is the TRAINING part (80% of your data)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse',
    subset='training',
    shuffle=True  # We shuffle during training
)

# 3. The VALIDATION part (20% of your data)
# NOTE: For evaluation/confusion matrix, we must set shuffle=False
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    shuffle=False  # IMPORTANT: Set to False for accurate evaluation
)

print(f"Found {train_generator.samples} training images and {validation_generator.samples} validation images.")

if train_generator.samples == 0:
    print("ERROR: No images found in training directory classes. Check your subfolder structure.")
    exit()

print("Both generators are now defined!")

# --- Dataset Balancing & Class Distribution Plot ---

# Calculate Class Weights to handle dataset imbalance
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

print("\nCalculating class weights to balance dataset training...")
train_classes = train_generator.classes
classes = np.unique(train_classes)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_classes
)
class_weights_dict = dict(zip(classes, class_weights))
print(f"Calculated class weights: {class_weights_dict}")

# Create output directory for evaluation and visualization results
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation_results'))
os.makedirs(output_dir, exist_ok=True)

# Plot and Save Class Distribution Graph
print("\nPlotting class distribution...")
class_labels_list = list(train_generator.class_indices.keys())
class_counts = np.bincount(train_classes)

plt.figure(figsize=(10, 6))
# Colors matching the class distribution chart: green, orange/yellow, coral, red, purple
colors = ['#66BB6A', '#FFA726', '#FF7043', '#EF5350', '#AB47BC']
colors = colors[:len(class_labels_list)]

bars = plt.bar(class_labels_list, class_counts, color=colors, edgecolor='none', width=0.4)
plt.ylabel('Count', fontsize=12)
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.grid(True, axis='y', linestyle=':', alpha=0.6)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + (max(class_counts)*0.01), f'{int(height):,}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
dist_path = os.path.join(output_dir, 'class_distribution.png')
plt.savefig(dist_path, dpi=300)
print(f"Saved Class Distribution plot to: {dist_path}")
plt.show()

# --- Training Logic ---

# 4. Start the training process
print("\nStarting initial training (Top Layers)...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weights_dict,
    verbose=1
)

# 5. Fine-Tuning
print("\nStarting fine-tuning...")
base_model.trainable = True

# Re-compile with a VERY SMALL learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    class_weight=class_weights_dict
)

# 6. Save the model locally
model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DR_EfficientNetB3_Model.keras'))
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# 7. Evaluation and Visualization
print("\nCreating output directory for evaluation results...")
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation_results'))
os.makedirs(output_dir, exist_ok=True)
print(f"Evaluation results will be saved to: {output_dir}")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

# 7.1. Plot Training & Validation Loss and Accuracy curves (Training vs Testing)
if 'history' in locals() or 'fine_tune_history' in locals():
    print("\nPlotting training history (Loss & Accuracy)...")
    
    acc, val_acc, loss, val_loss = [], [], [], []
    
    if 'history' in locals() and history is not None:
        acc += history.history['accuracy']
        val_acc += history.history['val_accuracy']
        loss += history.history['loss']
        val_loss += history.history['val_loss']
        
    if 'fine_tune_history' in locals() and fine_tune_history is not None:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']
        loss += fine_tune_history.history['loss']
        val_loss += fine_tune_history.history['val_loss']
        
    epochs_range = range(1, len(loss) + 1)
    
    plt.figure(figsize=(15, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#1976D2', linewidth=2.5, marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#00897B', linewidth=2.5, marker='o')
    plt.title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(epochs_range)
    plt.ylim([0.4, 1.02])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=11)
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#E53935', linewidth=2.5, marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#FF7043', linewidth=2.5, marker='o')
    plt.title('Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epochs_range)
    plt.ylim([0.2, max(max(loss), max(val_loss)) * 1.1])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=11)
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    loss_acc_path = os.path.join(output_dir, 'loss_accuracy_curves.png')
    plt.savefig(loss_acc_path, dpi=300, bbox_inches='tight')
    print(f"Saved Loss & Accuracy curves to: {loss_acc_path}")
    plt.show()

# 7.2. Evaluate model on Validation Generator
print("\nEvaluating model on validation dataset...")
eval_loss, eval_accuracy = model.evaluate(validation_generator, steps=len(validation_generator))
print(f"Validation Loss: {eval_loss:.4f}")
print(f"Validation Accuracy: {eval_accuracy:.4f}")

# Save overall stats
summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=== Model Evaluation Summary ===\n")
    f.write(f"Validation Loss: {eval_loss:.6f}\n")
    f.write(f"Validation Accuracy: {eval_accuracy:.6f}\n")
print(f"Saved Evaluation Summary to: {summary_path}")

# 7.3. Generate Predictions
print("\nGenerating model predictions...")
validation_generator.reset()
preds = model.predict(validation_generator, steps=len(validation_generator))
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())
num_classes = len(class_labels)

# Verify lengths match
if len(y_true) != len(y_pred):
    print(f"Warning: Length mismatch. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
    y_true = y_true[:len(y_pred)]

# 7.4. Configure and Plot Confusion Matrix
print("\nConfiguring and plotting Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={"size": 12, "weight": "bold"})
plt.title('Confusion Matrix: Diabetic Retinopathy Stages', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual Stage', fontsize=12)
plt.xlabel('Predicted Stage', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
print(f"Saved Confusion Matrix to: {cm_path}")
plt.show()

# 7.5. Generate and Save Classification Report
print("\nGenerating Classification Report...")
report_str = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0)
print(report_str)

report_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("=== Classification Report ===\n")
    f.write(report_str)
print(f"Saved Classification Report to: {report_path}")

# 7.6. Plot ROC and AUC curves (One-vs-Rest)
print("\nCalculating and plotting ROC and AUC curves...")
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    if np.sum(y_true_bin[:, i]) > 0:
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        fpr[i] = np.array([0.0, 1.0])
        tpr[i] = np.array([0.0, 0.0])
        roc_auc[i] = 0.0

plt.figure(figsize=(10, 8))
colors = ['#1abc9c', '#9b59b6', '#3498db', '#e67e22', '#e74c3c']
for i in range(num_classes):
    label_name = class_labels[i] if i < len(class_labels) else f"Class {i}"
    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], linewidth=2,
             label=f'ROC curve of {label_name} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
roc_path = os.path.join(output_dir, 'roc_auc_curve.png')
plt.savefig(roc_path, dpi=300)
print(f"Saved ROC/AUC curves to: {roc_path}")
plt.show()

# 7.7. Plot F1 Score vs Threshold Curve
print("\nCalculating and plotting F1 score curves...")
plt.figure(figsize=(10, 8))
thresholds = np.linspace(0.01, 0.99, 100)

for i in range(num_classes):
    f1_scores = []
    y_true_class = y_true_bin[:, i]
    y_prob_class = preds[:, i]
    
    for t in thresholds:
        y_pred_thresh = (y_prob_class >= t).astype(int)
        f1_scores.append(f1_score(y_true_class, y_pred_thresh, zero_division=0))
        
    label_name = class_labels[i] if i < len(class_labels) else f"Class {i}"
    plt.plot(thresholds, f1_scores, color=colors[i % len(colors)], linewidth=2,
             label=f'{label_name}')

plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.title('F1 Score vs. Confidence Threshold per Class', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
f1_path = os.path.join(output_dir, 'f1_curve.png')
plt.savefig(f1_path, dpi=300)
print(f"Saved F1 curves to: {f1_path}")
plt.show()

# 7.8. Calculate and Plot Accuracy of Each Class
print("\nCalculating and plotting accuracy of each class...")
class_accuracies = []
class_acc_report = []

class_acc_report.append("=== Class-wise Accuracy Report ===\n")
for i in range(num_classes):
    class_mask = (y_true == i)
    total_class_samples = np.sum(class_mask)
    
    if total_class_samples > 0:
        correct_class_preds = np.sum((y_pred == i) & class_mask)
        class_acc = correct_class_preds / total_class_samples
    else:
        class_acc = 0.0
        correct_class_preds = 0
        
    class_accuracies.append(class_acc)
    label_name = class_labels[i] if i < len(class_labels) else f"Class {i}"
    class_acc_report.append(f"{label_name}: Accuracy = {class_acc:.4f} ({correct_class_preds}/{total_class_samples} samples)\n")

class_acc_report_str = "".join(class_acc_report)
print(class_acc_report_str)

# Save class accuracy report
class_acc_path = os.path.join(output_dir, 'class_accuracy_report.txt')
with open(class_acc_path, 'w') as f:
    f.write(class_acc_report_str)
print(f"Saved Class-wise Accuracy Report to: {class_acc_path}")

# Plot bar chart for class-wise accuracies
plt.figure(figsize=(10, 6))
bars = plt.bar(class_labels, class_accuracies, color='#3498db', edgecolor='black', alpha=0.8)
plt.ylim([0.0, 1.1])
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Diabetic Retinopathy Stage (Class)', fontsize=12)
plt.title('Accuracy of Each Class', fontsize=14, fontweight='bold')
plt.grid(True, axis='y', linestyle=':', alpha=0.6)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.2%}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
class_acc_img_path = os.path.join(output_dir, 'class_accuracy_bar_chart.png')
plt.savefig(class_acc_img_path, dpi=300)
print(f"Saved Class-wise Accuracy Bar Chart to: {class_acc_img_path}")
plt.show()

# 7.9. Plot Overall KPI Metric Cards (Visual KPI Grid)
print("\nPlotting overall KPI metrics cards...")
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score

# Compute overall metric values
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)

fig, axes = plt.subplots(3, 2, figsize=(10, 8))
fig.patch.set_facecolor('#F8F9FA')

kpi_metrics = [
    {"value": f"{accuracy:.1%}", "label": "Accuracy", "color": "#1976D2"},
    {"value": f"{macro_f1:.3f}", "label": "Macro F1", "color": "#00897B"},
    {"value": f"{weighted_f1:.3f}", "label": "Weighted F1", "color": "#FF7043"},
    {"value": f"{macro_precision:.3f}", "label": "Macro Precision", "color": "#AB47BC"},
    {"value": f"{macro_recall:.3f}", "label": "Macro Recall", "color": "#E53935"},
    {"value": f"{kappa:.3f}", "label": "Kappa Score", "color": "#66BB6A"}
]

for ax, m in zip(axes.flat, kpi_metrics):
    ax.set_facecolor(m["color"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Value label
    ax.text(0.5, 0.6, m["value"], fontsize=28, color="white", weight="bold", ha="center", va="center")
    # Metric label
    ax.text(0.5, 0.25, m["label"], fontsize=14, color="white", weight="normal", ha="center", va="center")

plt.tight_layout(pad=3.0)
kpi_path = os.path.join(output_dir, 'overall_metrics_kpi_cards.png')
plt.savefig(kpi_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Saved Overall KPI Cards plot to: {kpi_path}")
plt.show()

# 7.10. Plot Per-Class Metrics Bar Chart (Grouped Precision, Recall, F1)
print("\nPlotting Per-Class Metrics grouped bar chart...")
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes), zero_division=0)

x = np.arange(num_classes)
width = 0.25

plt.figure(figsize=(12, 7))
rects1 = plt.bar(x - width, precision, width, label='Precision', color='#1976D2')
rects2 = plt.bar(x, recall, width, label='Recall', color='#00897B')
rects3 = plt.bar(x + width, f1, width, label='F1 Score', color='#FF7043')

plt.ylabel('Score', fontsize=12)
plt.title('Per-Class Metrics', fontsize=16, fontweight='bold', pad=15)
plt.xticks(x, class_labels, fontsize=11)
plt.ylim([0.5, 1.02])
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11, frameon=False)
plt.grid(True, axis='y', linestyle='-', alpha=0.2)

# Function to add numeric labels on top of bars
def autolabel_metrics(rects):
    for rect in rects:
        height = rect.get_height()
        if height >= 0.5:
            plt.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel_metrics(rects1)
autolabel_metrics(rects2)
autolabel_metrics(rects3)

plt.tight_layout()
per_class_path = os.path.join(output_dir, 'per_class_metrics.png')
plt.savefig(per_class_path, dpi=300)
print(f"Saved Per-Class Metrics plot to: {per_class_path}")
plt.show()
