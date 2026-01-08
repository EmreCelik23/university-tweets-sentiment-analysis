#!/usr/bin/env python3
"""
evaluate_predictions.py

Evaluate model predictions against ground truth labels.
Calculates accuracy, precision, recall, F1 score and confusion matrix for each model.

Usage:
    python evaluate_predictions.py --input predictions.xlsx
    python evaluate_predictions.py -i results.csv --ground-truth labels --output eval.csv
"""

import argparse
import pandas as pd
from typing import Dict, List
import sys

# Add parent directory to path for imports (if needed)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_predictions(
    input_file: str,
    ground_truth_column: str = "labels",
    models: Dict[str, str] = None,
    output_file: str = None
) -> pd.DataFrame:
    """
    Evaluates model predictions against ground truth.
    
    Args:
        input_file: Excel/CSV with predictions
        ground_truth_column: Column name for ground truth labels
        models: Dict mapping model names to prediction column names
        output_file: Optional output file for results
    
    Returns:
        DataFrame with evaluation metrics
    """
    print(f"📂 Dosya okunuyor: {input_file}")
    
    # Load data
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    print(f"✅ {len(df)} satır bulundu\n")
    
    # Ground truth check
    if ground_truth_column not in df.columns:
        print(f"❌ Hata: '{ground_truth_column}' sütunu bulunamadı!")
        print(f"Mevcut sütunlar: {', '.join(df.columns.tolist())}")
        sys.exit(1)
    
    # Default models if not provided
    if models is None:
        models = {
            'BERTurk': 'berturk_pred',
            'Electra': 'electra_pred',
            'CNN-BiLSTM': 'cnn_bilstm_pred',
            'BiLSTM': 'bilstm_pred',
            'CNN': 'cnn_pred'
        }
    
    print("=" * 70)
    print(f"MODEL PERFORMANS KARŞILAŞTIRMASI ({ground_truth_column} ile)")
    print("=" * 70)
    
    results = []
    
    for model_name, pred_col in models.items():
        if pred_col not in df.columns:
            print(f"\n⚠️ {model_name}: Tahmin sütunu bulunamadı ({pred_col})")
            continue
        
        # Valid predictions (non-null)
        valid_mask = df[pred_col].notna() & df[ground_truth_column].notna()
        valid_df = df[valid_mask]
        
        if len(valid_df) == 0:
            print(f"\n⚠️ {model_name}: Geçerli tahmin yok")
            continue
        
        # Ground truth and predictions
        y_true = valid_df[ground_truth_column].astype(int)
        y_pred = valid_df[pred_col].astype(int)
        
        # Correct/Incorrect counts
        correct = (y_true == y_pred).sum()
        incorrect = (y_true != y_pred).sum()
        total = len(valid_df)
        
        # Accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Confusion matrix components
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Save results
        results.append({
            'Model': model_name,
            'Total': total,
            'Correct': correct,
            'Incorrect': incorrect,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        })
        
        # Detailed output
        print(f"\n{model_name}")
        print("-" * 70)
        print(f"  Toplam Tahmin : {total}")
        print(f"  ✅ Doğru      : {correct} ({accuracy*100:.2f}%)")
        print(f"  ❌ Yanlış     : {incorrect} ({(1-accuracy)*100:.2f}%)")
        print(f"\n  Confusion Matrix:")
        print(f"    True Positives  (TP): {tp}")
        print(f"    True Negatives  (TN): {tn}")
        print(f"    False Positives (FP): {fp}")
        print(f"    False Negatives (FN): {fn}")
        print(f"\n  Metrikler:")
        print(f"    Accuracy  : {accuracy:.4f}")
        print(f"    Precision : {precision:.4f}")
        print(f"    Recall    : {recall:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary table
    print("\n" + "=" * 70)
    print("ÖZET TABLO")
    print("=" * 70)
    
    if len(results_df) > 0:
        # Display without confusion matrix columns for cleaner view
        display_cols = ['Model', 'Total', 'Correct', 'Incorrect', 'Accuracy', 'Precision', 'Recall', 'F1']
        print("\n" + results_df[display_cols].to_string(index=False))
        
        # Best models
        best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
        best_f1 = results_df.loc[results_df['F1'].idxmax()]
        
        print("\n" + "=" * 70)
        print("🏆 EN İYİ MODELLER")
        print("=" * 70)
        print(f"  Accuracy'e göre  : {best_acc['Model']} ({best_acc['Accuracy']*100:.2f}%)")
        print(f"  F1 Score'a göre  : {best_f1['Model']} ({best_f1['F1']:.4f})")
        print("=" * 70)
        
        # Save to file if requested
        if output_file:
            if output_file.endswith('.xlsx') or output_file.endswith('.xls'):
                results_df.to_excel(output_file, index=False)
            else:
                results_df.to_csv(output_file, index=False)
            print(f"\n💾 Sonuçlar kaydedildi: {output_file}")
    else:
        print("\n⚠️ Hiçbir model değerlendirilemedi")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python evaluate_predictions.py --input predictions.xlsx
  python evaluate_predictions.py -i results.csv --ground-truth labels
  python evaluate_predictions.py -i results.xlsx --models berturk electra
  python evaluate_predictions.py -i results.xlsx --output eval_results.csv

Varsayılan modeller:
  - BERTurk (berturk_pred)
  - Electra (electra_pred)
  - CNN-BiLSTM (cnn_bilstm_pred)
  - BiLSTM (bilstm_pred)
  - CNN (cnn_pred)
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file with predictions (.xlsx or .csv)"
    )
    parser.add_argument(
        "--ground-truth", "-gt",
        default="labels",
        help="Ground truth column name (default: labels)"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Models to evaluate (e.g., berturk electra). Default: all 5 models"
    )
    parser.add_argument(
        "--output", "-o",
        help="Optional: Save results to CSV/Excel"
    )
    
    args = parser.parse_args()
    
    # Build models dict if specific models requested
    models = None
    if args.models:
        model_mapping = {
            'berturk': ('BERTurk', 'berturk_pred'),
            'electra': ('Electra', 'electra_pred'),
            'cnn_bilstm': ('CNN-BiLSTM', 'cnn_bilstm_pred'),
            'bilstm': ('BiLSTM', 'bilstm_pred'),
            'cnn': ('CNN', 'cnn_pred'),
        }
        models = {}
        for m in args.models:
            m_lower = m.lower()
            if m_lower in model_mapping:
                name, col = model_mapping[m_lower]
                models[name] = col
            else:
                print(f"⚠️ Warning: Unknown model '{m}', skipping")
    
    evaluate_predictions(
        input_file=args.input,
        ground_truth_column=args.ground_truth,
        models=models,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
