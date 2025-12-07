# preprocessing.py
# Pipeline ringan untuk 3 model klasik: cleaning → normalisasi → stopwords → stemming
# - Inisialisasi stopwords, kamus kata baku, dan stemmer hanya SEKALI (lebih cepat)
# - Kolom Excel kamus fleksibel (mis.: tidak_baku/baku atau salah/benar)
# - Aman jika file kamus hilang (fallback ke kamus kosong)
# - Tidak bergantung pada transformers/torch

import os
import re
import string
import pandas as pd

# ===== NLTK stopwords (muat sekali, download jika belum ada) =====
try:
    from nltk.corpus import stopwords
    STOP_ID = set(stopwords.words("indonesian"))
    STOP_EN = set(stopwords.words("english"))
except Exception:
    # Coba download jika belum tersedia
    import nltk
    try:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        STOP_ID = set(stopwords.words("indonesian"))
        STOP_EN = set(stopwords.words("english"))
    except Exception:
        # Fallback kalau tetap gagal (offline)
        STOP_ID, STOP_EN = set(), set()

# ===== Sastrawi stemmer (buat sekali) =====
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _STEMMER = StemmerFactory().create_stemmer()
except Exception:
    _STEMMER = None  # kalau tidak terpasang, stemming akan dilewati

# ===== Lokasi proyek & file kamus =====
APP_DIR = os.path.dirname(os.path.abspath(__file__))
KAMUS_PATH = os.path.join(APP_DIR, "kamuskatabaku.xlsx")

def _load_kamus_excel(path: str) -> dict:
    """
    Muat kamus kata baku dari Excel dengan berbagai kemungkinan nama kolom.
    Preferensi nama kolom:
      - sisi kiri:  ('tidak_baku','tidak baku','salah','nonbaku','slang')
      - sisi kanan: ('baku','kata_baku','kata baku','benar','normal')
    Fallback: pakai dua kolom pertama jika deteksi gagal.
    """
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_excel(path)
        cols_l = [c.lower() for c in df.columns]
        left_idx = right_idx = None

        left_aliases  = {"tidak_baku", "tidak baku", "salah", "nonbaku", "slang"}
        right_aliases = {"baku", "kata_baku", "kata baku", "benar", "normal"}

        for i, c in enumerate(cols_l):
            if left_idx is None and c in left_aliases:
                left_idx = i
            if right_idx is None and c in right_aliases:
                right_idx = i
        if left_idx is None or right_idx is None:
            # fallback: dua kolom pertama
            if len(df.columns) >= 2:
                left_idx, right_idx = 0, 1
            else:
                return {}

        left_col, right_col = df.columns[left_idx], df.columns[right_idx]
        df = df.dropna(subset=[left_col, right_col])
        mapping = {
            str(a).strip().lower(): str(b).strip().lower()
            for a, b in zip(df[left_col], df[right_col])
            if str(a).strip()
        }
        return mapping
    except Exception:
        return {}

_SLANG_MAP = _load_kamus_excel(KAMUS_PATH)

# ===== Regex kompilasi sekali =====
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION  = re.compile(r"@\w+", re.UNICODE)
_HASHTAG  = re.compile(r"#\w+", re.UNICODE)
_HTMLTAG  = re.compile(r"<[^>]+>")  # jaga-jaga kalau ada HTML
_NUM_RE   = re.compile(r"\d+")
_PUNC_TBL = str.maketrans("", "", string.punctuation)
_NONWORD  = re.compile(r"[^\w\s]", re.UNICODE)
_WS_MULTI = re.compile(r"\s+")

# ===== Stopword tambahan ala percakapan =====
CUSTOM_STOP = {
    "iya","yaa","ya","ga","gak","gaa","nggak","lah","kah","sih","loh",
    "nya","ku","mu","nih","tuh","woi","woy","woii","deh","dong","nah",
    "oke","ok","okey","sip","nah","kan","kayak","kayaknya","banget"
}
STOP_ALL = (STOP_ID | STOP_EN | CUSTOM_STOP)

def _clean_basic(text: str) -> str:
    """Bersihkan URL/mention/hashtag/HTML, angka, tanda baca, dan whitespace ganda."""
    s = text.lower()
    s = _HTMLTAG.sub(" ", s)
    s = _URL_RE.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = _HASHTAG.sub(" ", s)
    s = _NUM_RE.sub(" ", s)
    s = _NONWORD.sub(" ", s)  # buang non-alfanumerik kecuali spasi/underscore
    s = s.translate(_PUNC_TBL)  # hapus punctuation standar
    s = _WS_MULTI.sub(" ", s).strip()
    return s

def _normalize_slang(s: str) -> str:
    """Mapping slang → baku berdasar Excel (jika tersedia)."""
    if not _SLANG_MAP:
        return s
    tokens = s.split()
    tokens = [_SLANG_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)

def _remove_stopwords(s: str) -> str:
    if not STOP_ALL:
        return s
    tokens = [t for t in s.split() if t not in STOP_ALL]
    return " ".join(tokens)

def _stem(s: str) -> str:
    if _STEMMER is None or not s:
        return s
    # Sastrawi menerima kalimat langsung (lebih cepat daripada per-token)
    return _STEMMER.stem(s)

def preprocess_text(text: str) -> str:
    """
    Pipeline final:
      raw → clean → slang-normalize → stopwords → stemming → strip
    Output dipakai langsung ke TF-IDF.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    s = _clean_basic(text)
    s = _normalize_slang(s)
    s = _remove_stopwords(s)
    s = _stem(s)
    s = _WS_MULTI.sub(" ", s).strip()
    return s
