# app.py (versi klasik: TF-IDF + LogReg / NB / SVM)
from flask import Flask, render_template, request
import os
import pickle
import joblib
import pandas as pd

from preprocessing import preprocess_text  # fungsi final milik kamu

# Wordcloud & Matplotlib
from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================
# Inisialisasi Flask
# =========================
app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
STATIC_DIR = os.path.join(APP_DIR, "static")
DATA_PATH = os.path.join(APP_DIR, "data", "label.xlsx")


# =========================
# Helper: loader pickle/joblib
# =========================
def _load(path):
    # Coba pickle dulu, kalau gagal coba joblib
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


# =========================
# Load TF-IDF & 3 model klasik
# =========================
tfidf = _load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
logreg = _load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
nb     = _load(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
svm    = _load(os.path.join(MODEL_DIR, "svm_model.pkl"))


# =========================
# Normalisasi label -> "Positive"/"Negative"
# =========================
def to_binary_label(y):
    """
    Apapun output model akan dinormalkan ke:
    - Positive
    - Negative
    """
    if y is None:
        return "Negative"
    s = str(y).strip().lower()
    if s in {"positive", "positif", "1", "true", "label_1"}:
        return "Positive"
    # Selain itu kita anggap Negative (termasuk 'negatif', 'netral', 0, dll)
    return "Negative"


def predict_all(text_raw: str):
    """
    Preprocess -> TF-IDF -> prediksi 3 model
    Return:
      predictions: dict {logreg, nb, svm} => 'Positive'/'Negative'
      probas: dict kemungkinan Positive jika model mendukung predict_proba, else None
      text_norm: teks setelah preprocessing
    """
    text_norm = preprocess_text(text_raw)
    X = tfidf.transform([text_norm])

    # Prediksi label
    y_logreg = to_binary_label(logreg.predict(X)[0])
    y_nb     = to_binary_label(nb.predict(X)[0])
    y_svm    = to_binary_label(svm.predict(X)[0])

    # Ubah teks label yang ditampilkan ke user
    def translate_label(label):
        if label == "Positive":
            return "Not Suicide"
        elif label == "Negative":
            return "Suicide"
        else:
            return label

    y_logreg = translate_label(y_logreg)
    y_nb     = translate_label(y_nb)
    y_svm    = translate_label(y_svm)


    def get_positive_proba(model):
        """Ambil probabilitas kelas Positive kalau model punya predict_proba."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            # Cari index kelas yang bermakna 'Positive'
            if hasattr(model, "classes_"):
                idx_pos = None
                for i, cls in enumerate(model.classes_):
                    if to_binary_label(cls) == "Positive":
                        idx_pos = i
                        break
                if idx_pos is not None:
                    return float(proba[idx_pos])
            # fallback: ambil max
            return float(max(proba))
        return None

    probas = {
        "logreg": get_positive_proba(logreg),
        "nb":     get_positive_proba(nb),
        "svm":    get_positive_proba(svm),
    }

    predictions = {
        "logreg": y_logreg,
        "nb":     y_nb,
        "svm":    y_svm,
    }

    return predictions, probas, text_norm


# =========================
# Load dataset untuk visualisasi
# =========================
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
else:
    # Jika tidak ada, buat df kosong agar halaman lain tetap hidup
    df = pd.DataFrame(columns=["Sentimen", "text_stemming"])


# =========================
# Routes
# =========================
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    predictions = None
    probas = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form.get("text_input", "")
        if text_input.strip():
            predictions, probas, _ = predict_all(text_input)

    # index.html diharapkan menampilkan 'predictions' & 'probas'
    return render_template(
        "index.html",
        predictions=predictions,
        probas=probas,
        text_input=text_input
    )


@app.route("/visualisasi")
def visualisasi():
    """
    Visualisasi 2 label (Positive/Negative)
    - Label sumber: df['text_label']
    - Korpus wordcloud: df['text_stemming']
    - Output gambar: static/wordcloud_positive.png & static/wordcloud_negative.png
    """
    import pandas as pd
    import os
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt

    # --- Mapping label ke biner ---
    def map_to_binary_label(s):
        if pd.isna(s): return "Negative"
        s = str(s).strip().lower()
        return "Positive" if s in {"positive", "positif"} else "Negative"

    # Pastikan kolom ada
    has_label = "text_label" in df.columns
    has_text  = "text_stemming" in df.columns

    if has_label:
        df_vis = df.copy()
        df_vis["Sentimen_Binary"] = df_vis["text_label"].map(map_to_binary_label)
    else:
        df_vis = pd.DataFrame(columns=["Sentimen_Binary", "text_stemming"])

    # --- Hitung jumlah (urutan: Positive, Negative) ---
    count_pos = int((df_vis["Sentimen_Binary"] == "Positive").sum())
    count_neg = int((df_vis["Sentimen_Binary"] == "Negative").sum())
    pie_data = [count_pos, count_neg]
    bar_data = pie_data

    # --- Wordcloud helper ---
    os.makedirs(STATIC_DIR, exist_ok=True)

    def plot_wordcloud_binary(dataframe, label, colormap, filename):
        if not has_text:
            return  # kolom text_stemming tidak ada
        subset = dataframe[dataframe["Sentimen_Binary"] == label]
        if subset.empty:
            # hapus gambar lama bila ada
            path_old = os.path.join(STATIC_DIR, filename)
            if os.path.exists(path_old):
                try: os.remove(path_old)
                except Exception: pass
            return
        corpus = " ".join(subset["text_stemming"].fillna("").astype(str))
        if not corpus.strip():
            return
        wc = WordCloud(
            width=800, height=400, background_color="white",
            colormap=colormap, stopwords=STOPWORDS, collocations=False
        ).generate(corpus)
        path = os.path.join(STATIC_DIR, filename)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(path, format="png")
        plt.close()

    # --- Buat wordcloud Positive & Negative ---
    plot_wordcloud_binary(df_vis, "Positive", "Greens", "wordcloud_positive.png")
    plot_wordcloud_binary(df_vis, "Negative", "Reds",   "wordcloud_negative.png")

    return render_template("visualisasi.html", pie_data=pie_data, bar_data=bar_data)


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


# =========================
# Run server
# =========================

if __name__ == "__main__":
    # Ubah port bila perlu, debug=True saat dev
    app.run()

