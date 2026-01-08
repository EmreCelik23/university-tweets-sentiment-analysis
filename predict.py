import argparse
import os
import re
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sentencepiece as spm

# ======================================================================
# KONFİGÜRASYON
# ======================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "BERTurk": "models/berturk_model",      # HF klasörü (config.json burada)
    "Electra": "models/electra_model",      # HF klasörü (Turkish ELECTRA)
    "CNN": "models/cnn_model.pt",           # cnn_spm_best.pt'yi böyle adlandırdıysan
    "BiLSTM": "models/bilstm_model.pt",     # bilstm_spm_best.pt
    "CNN-BiLSTM": "models/hybrid_model.pt", # cnn_bilstm_spm_best.pt
}

# train_classical_spm.py ile eğittiğin SentencePiece modeli
SPM_MODEL_PATH = "models/spm_uni.model"

MAX_LEN_BERT = 256          # BERT için
MAX_LEN_DL = 40             # classical modeller için (train_classical_spm.py default)

EMBED_DIM = 200
HIDDEN_DIM = 128
NUM_FILTERS = 100
FILTER_SIZES = (3, 4, 5)

OUTPUT_COL_ORDER = [
    "type",
    "url",
    "tags",
    "text",
    "createdAt",
    "location",
    "authorUserName",
    "university",
    "group",
]

_loaded_models: Dict[str, Any] = {}
_spm: Optional[spm.SentencePieceProcessor] = None


# ======================================================================
# MODEL SINIFLARI (train_classical_spm.py ile birebir)
# ======================================================================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels=2,
                 filter_sizes=(3, 4, 5), num_filters=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):
        x = self.embed(x)              # [B, L, D]
        x = x.permute(0, 2, 1)         # [B, D, L]
        xs = []
        for conv in self.convs:
            c = F.relu(conv(x))        # [B, F, L']
            c = torch.max(c, dim=2).values
            xs.append(c)
        x = torch.cat(xs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2 * hidden_dim, num_labels)

    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        x, _ = torch.max(lstm_out, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class TextCNNBiLSTM(nn.Module):
    """
    Embedding -> BiLSTM -> CNN
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_labels=2, filter_sizes=(3, 4, 5), num_filters=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        conv_in = 2 * hidden_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=conv_in,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.permute(0, 2, 1)
        xs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            c = torch.max(c, dim=2).values
            xs.append(c)
        x = torch.cat(xs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


# ======================================================================
# YARDIMCI FONKSİYONLAR
# ======================================================================

def clean_text_minimal(s: str) -> str:
    """
    BERTurk için kullandığın hafif temizlik – buna dokunmuyoruz.
    Classical modeller için train pipeline aşağıdaki normalize_tweet.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " <url> ", s)
    s = re.sub(r"@\w+", "@user", s)
    s = re.sub(r"#([0-9a-z_çğıöşü]+)", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_tweet(text: str) -> str:
    """
    train_classical_spm.py'deki normalize_tweet ile birebir.
    Classical modellerin TRAIN sırasında kullanılan temizlik.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()

    # URL
    t = re.sub(r"http\S+|www\.\S+", "<url>", t)

    # mention
    t = re.sub(r"@\w+", "<user>", t)

    # hashtag: #yurtmagduru -> yurtmagduru
    t = re.sub(r"#", "", t)

    # sayılar -> <num>
    t = re.sub(r"\d+([\.,]\d+)?", "<num>", t)

    # çoklu boşluk -> tek boşluk
    t = re.sub(r"\s+", " ", t).strip()

    return t


def load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def save_df(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def _build_model_instance(model_name: str, vocab_size: int) -> nn.Module:
    if model_name == "CNN":
        return TextCNN(vocab_size, EMBED_DIM, num_labels=2,
                       filter_sizes=FILTER_SIZES, num_filters=NUM_FILTERS)
    if model_name == "BiLSTM":
        return TextBiLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, num_labels=2)
    if model_name == "CNN-BiLSTM":
        return TextCNNBiLSTM(
            vocab_size, EMBED_DIM, HIDDEN_DIM,
            num_labels=2, filter_sizes=FILTER_SIZES, num_filters=NUM_FILTERS,
        )
    raise ValueError(f"Bilinmeyen model adı: {model_name}")


def get_spm() -> spm.SentencePieceProcessor:
    global _spm
    if _spm is None:
        if not os.path.isfile(SPM_MODEL_PATH):
            raise FileNotFoundError(f"SentencePiece modeli bulunamadı: {SPM_MODEL_PATH}")
        _spm = spm.SentencePieceProcessor()
        _spm.load(SPM_MODEL_PATH)
        print(f"✅ SentencePiece yüklendi ({SPM_MODEL_PATH})")
    return _spm


def encode_text_dl(text: str, maxlen: int) -> torch.Tensor:
    """
    Classical modeller için TRAIN pipeline ile birebir:
      text -> normalize_tweet -> SentencePiece -> pad/truncate -> tensor
    """
    sp = get_spm()
    t = normalize_tweet(text)
    ids = sp.encode(t, out_type=int)

    if len(ids) < maxlen:
        ids = ids + [0] * (maxlen - len(ids))   # pad_id=0
    else:
        ids = ids[:maxlen]

    return torch.tensor([ids], dtype=torch.long).to(DEVICE)


# ======================================================================
# MODEL YÜKLEME
# ======================================================================

def get_model(model_name: str) -> Any:
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"MODEL_PATHS içinde {model_name} tanımlı değil.")

    # ---------- BERTURK / ELECTRA ----------
    if model_name in ["BERTurk", "Electra"]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{model_name} klasörü bulunamadı: {path}")

        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        model.to(DEVICE)

        container = {"type": "hf", "model": model, "tokenizer": tok}
        _loaded_models[model_name] = container
        return container

    # ---------- DL MODELLER (CNN / BiLSTM / CNN-BiLSTM) ----------
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {path}")

    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        tensor_items = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        state_dict = tensor_items if tensor_items else ckpt
    else:
        raise RuntimeError(f"{model_name} checkpoint yapısı dict değil.")

    sp = get_spm()
    vocab_size = sp.get_piece_size()

    model = _build_model_instance(model_name, vocab_size)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    container = {"type": "pt", "model": model}
    _loaded_models[model_name] = container
    return container


# ======================================================================
# TAHMİN MOTORU (Streamlit'in kullandığı)
# ======================================================================

def _predict_with_bert(text: str, university: str, model_name: str = "BERTurk") -> Tuple[int, float]:
    m = get_model(model_name)
    tok = m["tokenizer"]
    model = m["model"]

    clean_txt = clean_text_minimal(text)
    
    # Sadece text kullan, üniversite bilgisini ekleme
    inp = clean_txt

    enc = tok(
        inp,
        truncation=True,
        padding=True,
        max_length=MAX_LEN_BERT,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        
        # Electra için threshold tuning kullan (varsa)
        if model_name == "Electra":
            threshold = _load_electra_threshold()
            prob_pos = float(probs[1])
            pred = 1 if prob_pos >= threshold else 0
            conf = prob_pos if pred == 1 else float(probs[0])
        else:
            pred = int(probs.argmax())
            conf = float(probs.max())

    # 0=olumsuz, 1=olumlu
    return pred, conf


def _load_electra_threshold() -> float:
    """
    Electra için inference_threshold.json'dan threshold yükle.
    Dosya yoksa default 0.5 kullan.
    """
    threshold_path = os.path.join(MODEL_PATHS.get("Electra", "models/electra_model"), "inference_threshold.json")
    if os.path.exists(threshold_path):
        try:
            import json
            with open(threshold_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("threshold", 0.5))
        except Exception:
            pass
    return 0.5


def _predict_with_dl(model_name: str, text: str) -> Tuple[int, float]:
    """
    Classical modeller için:
      - ham text al
      - encode_text_dl -> normalize_tweet + SentencePiece + pad
    """
    m = get_model(model_name)
    model = m["model"]

    tensor_inp = encode_text_dl(text, MAX_LEN_DL)

    with torch.no_grad():
        logits = model(tensor_inp)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())
        conf = float(probs.max())

    return pred, conf


def get_multi_model_prediction(
    text: str, university: str = "Genel"
) -> Dict[str, Optional[Tuple[int, float]]]:
    """
    Streamlit UI bu fonksiyonu çağırıyor.
    Dönüş:
      {
        "BERTurk":    (pred, conf),
        "Electra":    (pred, conf) veya None,
        "CNN-BiLSTM": (pred, conf) veya None,
        "BiLSTM":     (pred, conf) veya None,
        "CNN":        (pred, conf) veya None
      }
    """
    results: Dict[str, Optional[Tuple[int, float]]] = {}

    # Transformer modeller: BERTurk, Electra
    try:
        bert_pred, bert_conf = _predict_with_bert(text, university)
        results["BERTurk"] = (bert_pred, bert_conf)
    except Exception:
        results["BERTurk"] = None
    
    try:
        electra_pred, electra_conf = _predict_with_bert(text, university, model_name="Electra")
        results["Electra"] = (electra_pred, electra_conf)
    except Exception:
        results["Electra"] = None

    # Klasik modeller: hata varsa logla, sonuç yerine None koy
    for name in ["CNN-BiLSTM", "BiLSTM", "CNN"]:
        try:
            p, c = _predict_with_dl(name, text)
            results[name] = (p, c)
        except Exception:
            results[name] = None

    return results


# ======================================================================
# CLI: TOPLU ETİKETLEME (sadece BERTurk ile 0/1 tags)
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="xlsx/csv giriş (en az: text, university)")
    ap.add_argument("--out", required=True, help="xlsx/csv çıkış")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=256)
    args = ap.parse_args()

    df = load_df(args.inp)

    needed = {"text", "university"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Eksik kolon(lar): {missing}. Gerekli kolonlar: {needed}")

    texts = [clean_text_minimal(x) for x in df["text"].astype(str).tolist()]
    # Sadece text kullan, üniversite bilgisini ekleme
    inputs = texts

    m = get_model("BERTurk")
    tok = m["tokenizer"]
    model = m["model"]
    model.eval()

    preds, confs = [], []

    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(inputs), args.batch), desc="Predict")
    except Exception:
        iterator = range(0, len(inputs), args.batch)

    with torch.no_grad():
        for i in iterator:
            chunk = inputs[i:i + args.batch]
            enc = tok(
                chunk,
                truncation=True,
                padding=True,
                max_length=args.maxlen,
                return_tensors="pt",
            ).to(DEVICE)

            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            pred_i = probs.argmax(axis=1)
            mx = probs.max(axis=1)

            preds.extend(pred_i.tolist())
            confs.extend(mx.tolist())

    out_df = df.copy()
    out_df["tags"] = preds      # 0 / 1
    out_df["confidence"] = confs

    for col in OUTPUT_COL_ORDER:
        if col not in out_df.columns:
            out_df[col] = pd.NA

    out_df = out_df[OUTPUT_COL_ORDER + [c for c in out_df.columns if c not in OUTPUT_COL_ORDER]]

    save_df(out_df, args.out)
    print(f"✅ Kaydedildi -> {args.out}")


if __name__ == "__main__":
    main()