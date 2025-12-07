from flask import Flask, render_template, request
import os
import pickle
import joblib
import pandas as pd

from preprocessing import preprocess_text

# =========================
# PATH FIX SESUAI STRUKTUR BARU
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))            # /deploy
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))         # /WEBSITE - COPY
WEB_DIR = os.path.join(BASE_DIR, "web")                         # /web

MODEL_DIR = os.path.join(WEB_DIR, "model")
STATIC_DIR = os.path.join(WEB_DIR, "static")
TEMPLATE_DIR = os.path.join(WEB_DIR, "templates")
DATA_PATH = os.path.join(WEB_DIR, "data", "label.xlsx")

# Flask pakai templates & static yg ada di folder /web
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# =========================
# Helper loader
# =========================
def _load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)

# =========================
# Load Model
# =========================
tfidf = _load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
logreg = _load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
nb     = _load(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
svm    = _load(os.path.join(MODEL_DIR, "svm_model.pkl"))

# =========================
# Label Normalization
# =========================
def to_binary_label(y):
    if y is None:
        return "Negative"
    s = str(y).strip().lower()
    if s in {"positive", "positif", "1", "true", "label_1"}:
        return "Positive"
    return "Negative"

def predict_all(text_raw: str):
    text_norm = preprocess_text(text_raw)
    X = tfidf.transform([text_norm])

    # prediksi
    y_logreg = to_binary_label(logreg.predict(X)[0])
    y_nb     = to_binary_label(nb.predict(X)[0])
    y_svm    = to_binary_label(svm.predict(X)[0])

    # translate ke tampilan user
    def translate(x):
        return "Not Suicide" if x == "Positive" else "Suicide"

    y_logreg = translate(y_logreg)
    y_nb     = translate(y_nb)
    y_svm    = translate(y_svm)

    # probabilitas
    def get_positive_proba(model):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            if hasattr(model, "classes_"):
                for i, cls in enumerate(model.classes_):
                    if to_binary_label(cls) == "Positive":
                        return float(proba[i])
            return float(max(proba))
        return None

    probas = {
        "logreg": get_positive_proba(logreg),
        "nb": get_positive_proba(nb),
        "svm": get_positive_proba(svm),
    }

    predictions = {
        "logreg": y_logreg,
        "nb": y_nb,
        "svm": y_svm,
    }

    return predictions, probas, text_norm

# =========================
# Load dataset visualisasi
# =========================
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
else:
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

    return render_template("index.html",
                           predictions=predictions,
                           probas=probas,
                           text_input=text_input)

@app.route("/visualisasi")
def visualisasi():
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import os

    # mapping
    def map_binary(s):
        if pd.isna(s): return "Negative"
        s = str(s).strip().lower()
        return "Positive" if s in {"positive", "positif"} else "Negative"

    has_label = "text_label" in df.columns
    has_text = "text_stemming" in df.columns

    if has_label:
        df_vis = df.copy()
        df_vis["Sentimen_Binary"] = df_vis["text_label"].map(map_binary)
    else:
        df_vis = pd.DataFrame(columns=["Sentimen_Binary", "text_stemming"])

    count_pos = int((df_vis["Sentimen_Binary"] == "Positive").sum())
    count_neg = int((df_vis["Sentimen_Binary"] == "Negative").sum())

    pie_data = [count_pos, count_neg]
    bar_data = pie_data

    os.makedirs(STATIC_DIR, exist_ok=True)

    # wordcloud function
    def make_wc(label, cmap, fname):
        if not has_text: return
        subset = df_vis[df_vis["Sentimen_Binary"] == label]
        if subset.empty:
            old = os.path.join(STATIC_DIR, fname)
            if os.path.exists(old): os.remove(old)
            return
        corpus = " ".join(subset["text_stemming"].fillna("").astype(str))
        if not corpus.strip(): return
        wc = WordCloud(width=800, height=400, colormap=cmap,
                       background_color="white",
                       stopwords=STOPWORDS, collocations=False).generate(corpus)
        path = os.path.join(STATIC_DIR, fname)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(path, format="png")
        plt.close()

    make_wc("Positive", "Greens", "wordcloud_positive.png")
    make_wc("Negative", "Reds", "wordcloud_negative.png")

    return render_template("visualisasi.html",
                           pie_data=pie_data,
                           bar_data=bar_data)

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

# =========================
# Local Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
