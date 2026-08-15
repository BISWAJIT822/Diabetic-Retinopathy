# Diabetic Retinopathy Evaluation & Optimization Guide

This directory is automatically created and populated when you run `EfficientNetB3.py`. It contains complete performance evaluations, high-resolution visualization graphs, and metric reports.

---

## ⚖️ Dataset Balancing (Improve Accuracy)
To handle severe class imbalance (where certain stages of Diabetic Retinopathy have significantly fewer samples than others), the script now dynamically computes **balanced class weights** using:
$$\text{Class Weight} = \frac{\text{Total Samples}}{\text{Number of Classes} \times \text{Class Samples}}$$
These weights are passed to the `class_weight` parameter of `model.fit()` in both training phases (initial top layers and fine-tuning). This ensures the model learns equally from all stages, directly improving the macro and weighted metrics (F1-score, sensitivity/recall) on minority classes.

---

## 📊 Directory Structure of Saved Outputs

All files are saved with a high resolution of 300 DPI:

```text
evaluation_results/
├── class_distribution.png          # Bar chart of class frequencies (before training)
├── loss_accuracy_curves.png        # Training vs Validation (Testing) Loss & Accuracy per Epoch
├── overall_metrics_kpi_cards.png   # 6 KPI Cards grid (Accuracy, Macro F1, Weighted F1, etc.)
├── per_class_metrics.png           # Grouped bar chart (Precision, Recall, F1 side-by-side per stage)
├── confusion_matrix.png            # Confusion matrix count heatmap
├── roc_auc_curve.png               # One-vs-Rest ROC curve and AUC values per class
├── f1_curve.png                    # F1 Score vs Confidence Threshold curves
├── class_accuracy_bar_chart.png    # Accuracy percentage bar chart per class
├── evaluation_summary.txt          # Overall evaluation loss & accuracy numeric summary
├── classification_report.txt       # Precision, Recall, F1-score & Support text report
└── class_accuracy_report.txt       # Accuracy of each class text report
```

---

## 🔍 Visual Graphs Details

### 1. Class Distribution (`class_distribution.png`)
* **What it plots**: A bar chart of sample counts per class before training begins.
* **Colors**: Matches the exact custom color scheme of your dataset categories (Green, Orange, Coral, Red, Purple).

### 2. Accuracy & Loss vs Epochs (`loss_accuracy_curves.png`)
* **What it plots**: Training vs Validation Accuracy and Training vs Validation Loss over all epochs.
* **Style**: Circular markers on line plots, matching the colors of your reference (Blue & Green for accuracy, Red & Orange for loss).

### 3. Overall KPI Metric Cards (`overall_metrics_kpi_cards.png`)
* **What it plots**: 6 colored grids containing the critical summary metrics:
  * **Accuracy** (Blue, `#1976D2`)
  * **Macro F1** (Teal, `#00897B`)
  * **Weighted F1** (Orange, `#FF7043`)
  * **Macro Precision** (Purple, `#AB47BC`)
  * **Macro Recall** (Coral, `#E53935`)
  * **Kappa Score** (Green, `#66BB6A`)

### 4. Per-Class Metrics Grouped Bar Chart (`per_class_metrics.png`)
* **What it plots**: Grouped side-by-side bar chart showing **Precision**, **Recall**, and **F1 Score** for each of the 5 categories.
* **Colors**: Blue (Precision), Teal (Recall), and Orange (F1 Score).
* **Zoom**: Focuses on the $0.5 - 1.02$ range to easily compare high-performance thresholds.
