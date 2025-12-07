from flask import Flask, render_template, request
import os
import pickle
import joblib
import pandas as pd

# ======================================
# IMPORT SETELAH PINDAH KE DALAM /web
# ======================================
from web.preprocessing import preprocess_text

# ======================================
# PATH CONFIG
# ======================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # root repo
WEB_DIR = os.path.join(BASE_DIR, "web")                # folder web

MODEL_DIR = os.path.join(WEB_DIR, "model")
STATIC_DIR = os.path.join(WEB_DIR, "static")
TEMPLATE_DIR = os.path.join(WEB_DIR, "templates")
DATA_PATH = os.path.join(WEB_DIR, "data", "label.xlsx")

# ======================================
# FLASK APP
# ======================================
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


# ======================================
# HELPER LOAD MODEL
# ======================================
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


def to_binary_label(y):
    if y is None:
        return "Negative"
    s = str(y).strip().lower()
    if s in {"positive", "positif", "1", "true", "label_1"}:
        return "Positive"
    return "Negative"


# ======================================
# LOAD MODELS
# ======================================
tfidf = load_model(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
logreg = load_model(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
nb = load_model(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
svm = load_model(os.path.join(MODEL_DIR, "svm_model.pkl"))


# ======================================
# PREDICTION LOGIC
# ======================================
def predict_all(text_raw):
    text_norm = preprocess_text(text_raw)
    X = tfidf.transform([text_norm])

    def translate_label(y):
        return "Not Suicide" if to_binary_label(y) == "Positive" else "Suicide"

    predictions = {
        "logreg": translate_label(logreg.predict(X)[0]),
        "nb": translate_label(nb.predict(X)[0]),
        "svm": translate_label(svm.predict(X)[0]),
    }

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

    return predictions, probas, text_norm


# ======================================
# LOAD DATASET
# ======================================
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
else:
    df = pd.DataFrame(columns=["Sentimen", "text_stemming"])


# ======================================
# ROUTES
# ======================================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/home")
def home_redirect():
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

    return render_template("index.html", predictions=predictions, probas=probas, text_input=text_input)


@app.route("/visualisasi")
def visualisasi():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS

    df_vis = df.copy()
    label_col = None
    for col in df_vis.columns:
        if any(key in col.lower() for key in ["label", "sentimen", "kategori", "class"]):
            label_col = col
            break

    if label_col is None or label_col not in df_vis.columns:
        df_vis = pd.DataFrame(columns=["Sentimen_Binary", "text_stemming"])
    else:
        df_vis["Sentimen_Binary"] = df_vis[label_col].map(
            lambda s: "Positive" if str(s).strip().lower() in {"positive", "positif", "1", "true", "yes"} else "Negative"
        )

    count_pos = (df_vis["Sentimen_Binary"] == "Positive").sum()
    count_neg = (df_vis["Sentimen_Binary"] == "Negative").sum()

    os.makedirs(STATIC_DIR, exist_ok=True)

    def make_wordcloud(label, cmap, fname):
        subset = df_vis[df_vis["Sentimen_Binary"] == label]
        if subset.empty:
            return
        corpus = " ".join(subset["text_stemming"].fillna("").astype(str))
        if not corpus.strip():
            return

        wc = WordCloud(
            width=800,
            height=400,
            colormap=cmap,
            background_color="white",
            stopwords=STOPWORDS,
            collocations=False,
        ).generate(corpus)

        out_path = os.path.join(STATIC_DIR, fname)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(out_path, format="png")
        plt.close()

    make_wordcloud("Positive", "Greens", "wordcloud_positive.png")
    make_wordcloud("Negative", "Reds", "wordcloud_negative.png")

    return render_template("visualisasi.html", pie_data=[count_pos, count_neg], bar_data=[count_pos, count_neg])


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


# ======================================
# RUN LOCAL
# ======================================
if __name__ == "__main__":
    app.run(debug=True)
