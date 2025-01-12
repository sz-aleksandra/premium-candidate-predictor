import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_curve, auc, f1_score, precision_score, recall_score
)
import seaborn as sns
import pickle

def load_data(base_path='data/'):
    X_train = pd.read_csv(f'{base_path}X_train.csv')
    X_val = pd.read_csv(f'{base_path}X_val.csv')
    X_test = pd.read_csv(f'{base_path}X_test.csv')
    Y_test = pd.read_csv(f'{base_path}Y_test.csv')
    Y_train = pd.read_csv(f'{base_path}Y_train.csv')
    Y_val = pd.read_csv(f'{base_path}Y_val.csv')
    return X_train, X_val,X_test, Y_train, Y_val,Y_test

def get_model_responses(scaler, model, X):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    return y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

def plot_roc_curve(y_true, y_prob, ax, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return roc_auc

def plot_precision_recall_curve(y_true, y_prob, ax, title):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, color='blue', lw=2, label=f'P-R curve (AP = {avg_precision:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    return avg_precision

def plot_model_confusion_matrices(y_true_A, y_pred_A, y_true_B, y_pred_B, model_A_name, model_B_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_confusion_matrix(y_true_A, y_pred_A, axes[0], f'{model_A_name} - Confusion Matrix')
    plot_confusion_matrix(y_true_B, y_pred_B, axes[1], f'{model_B_name} - Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_model_roc_curves(y_true_A, y_prob_A, y_true_B, y_prob_B, model_A_name, model_B_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    roc_auc_A = plot_roc_curve(y_true_A, y_prob_A, axes[0], f'{model_A_name} - ROC Curve')
    roc_auc_B = plot_roc_curve(y_true_B, y_prob_B, axes[1], f'{model_B_name} - ROC Curve')
    plt.tight_layout()
    plt.show()
    return roc_auc_A, roc_auc_B

def plot_model_pr_curves(y_true_A, y_prob_A, y_true_B, y_prob_B, model_A_name, model_B_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    avg_precision_A = plot_precision_recall_curve(y_true_A, y_prob_A, axes[0], 
                                                f'{model_A_name} - Precision-Recall Curve')
    avg_precision_B = plot_precision_recall_curve(y_true_B, y_prob_B, axes[1], 
                                                f'{model_B_name} - Precision-Recall Curve')
    plt.tight_layout()
    plt.show()
    return avg_precision_A, avg_precision_B

def calculate_model_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'TNR': recall_score((-np.array(y_true)+1), (-np.array(y_pred)+1)),
        'f1': f1_score(y_true, y_pred),
    }

def print_comparison_table(metrics_A, metrics_B, model_A_name, model_B_name):
    print(f"Metric    |{model_A_name} |{model_B_name} ")
    [print(f"{k:<10}|   {v:.3f}   |   {metrics_B[k]:.3f} ") for k, v in metrics_A.items()]