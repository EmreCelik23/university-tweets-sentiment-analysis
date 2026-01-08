#!/usr/bin/env python3
"""
predict_batch.py

CLI tool for batch prediction using ALL models.
Reads Excel/CSV, predicts with 5 models, outputs results with separate columns.

Usage:
    python predict_batch.py --input data.xlsx --output results.xlsx
    python predict_batch.py -i data.csv -o results.csv --text-column tweet_text
"""

import argparse
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import get_multi_model_prediction


def batch_predict(input_file: str, output_file: str, text_column: str = "text"):
    """
    Excel/CSV dosyasından text'leri okur, tüm modellerle tahmin yapar ve sonuçları kaydeder.
    
    Args:
        input_file: Giriş dosyası (.xlsx, .xls veya .csv)
        output_file: Çıkış dosyası (.xlsx, .xls veya .csv)  
        text_column: Text içeren sütun adı (varsayılan: "text")
    """
    print(f"📂 Dosya okunuyor: {input_file}")
    
    # Dosyayı oku
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    print(f"✅ {len(df)} satır bulundu")
    
    # Text sütunu kontrolü
    if text_column not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(
            f"'{text_column}' sütunu bulunamadı. Mevcut sütunlar: {available_cols}\n"
            f"--text-column parametresi ile doğru sütun adını belirtin."
        )
    
    # Sonuç sütunlarını hazırla (5 model)
    model_columns = {
        'BERTurk': ('berturk_pred', 'berturk_conf'),
        'Electra': ('electra_pred', 'electra_conf'),
        'CNN-BiLSTM': ('cnn_bilstm_pred', 'cnn_bilstm_conf'),
        'BiLSTM': ('bilstm_pred', 'bilstm_conf'),
        'CNN': ('cnn_pred', 'cnn_conf'),
    }
    
    for pred_col, conf_col in model_columns.values():
        df[pred_col] = None
        df[conf_col] = None
    
    print("\n🔮 Tahminler yapılıyor (5 model)...")
    
    # Her satır için tüm modellerle tahmin yap
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="İşleniyor"):
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        
        if not text.strip():
            continue
        
        try:
            # Tüm modellerle tahmin yap (university="Genel" → sadece text gönderir)
            results = get_multi_model_prediction(text, university="Genel")
            
            # Her model için sonuçları kaydet
            for model_name, (pred_col, conf_col) in model_columns.items():
                if results.get(model_name):
                    pred, conf = results[model_name]
                    df.at[idx, pred_col] = pred
                    df.at[idx, conf_col] = round(conf, 4)
                    
        except Exception as e:
            print(f"\n⚠️ Satır {idx} hata: {e}")
            continue
    
    # Sonuçları kaydet
    print(f"\n💾 Sonuçlar kaydediliyor: {output_file}")
    
    if output_file.endswith('.xlsx') or output_file.endswith('.xls'):
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"✅ Tamamlandı! {len(df)} satır işlendi.")
    
    # Özet istatistikler
    print("\n" + "=" * 60)
    print("ÖZET İSTATİSTİKLER")
    print("=" * 60)
    
    for model_name, (pred_col, _) in model_columns.items():
        valid_preds = df[pred_col].dropna()
        if len(valid_preds) > 0:
            pos_count = int((valid_preds == 1).sum())
            neg_count = int((valid_preds == 0).sum())
            total = pos_count + neg_count
            pos_pct = (pos_count / total * 100) if total > 0 else 0
            print(f"{model_name:12} → Pozitif: {pos_count:4} ({pos_pct:5.1f}%), Negatif: {neg_count:4}")
        else:
            print(f"{model_name:12} → Tahmin yapılamadı")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Excel/CSV dosyasından text okuyup 5 modelle tahmin yapar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python predict_batch.py --input data.xlsx --output results.xlsx
  python predict_batch.py -i tweets.csv -o predictions.csv --text-column tweet_text
  
Çıkış kolonları:
  - berturk_pred, berturk_conf
  - electra_pred, electra_conf  
  - cnn_bilstm_pred, cnn_bilstm_conf
  - bilstm_pred, bilstm_conf
  - cnn_pred, cnn_conf
  
Tahmin değerleri: 0 (negatif), 1 (pozitif)
Confidence: 0.0 - 1.0 arası güven skoru
        """
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Giriş dosyası (.xlsx, .xls veya .csv)"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Çıkış dosyası (.xlsx, .xls veya .csv)"
    )
    parser.add_argument(
        "--text-column",
        "-t",
        default="text",
        help="Text içeren sütun adı (varsayılan: 'text')"
    )
    
    args = parser.parse_args()
    
    batch_predict(
        input_file=args.input,
        output_file=args.output,
        text_column=args.text_column
    )


if __name__ == "__main__":
    main()
