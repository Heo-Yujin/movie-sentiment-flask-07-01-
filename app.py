import os
import joblib
import re
from flask import Flask, render_template, request
from konlpy.tag import Okt

app = Flask(__name__)

# jdk저장 위치
okt = Okt(
    jvmpath=r"C:\Users\PC2510\Downloads\microsoft-jdk-11.0.31-windows-x64\jdk-11.0.31+11\bin\server\jvm.dll"
)

def okt_tokenizer(text):
    return okt.morphs(text)

# 현재 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["review"]
    text = re.sub(r'[^ ㄱ-ㅣ가-힣]+', ' ', text)
    x = tfidf.transform([text])
    pred = model.predict(x)[0]

    if pred == 1:
        result = "👍 긍정 리뷰입니다."
    else:
        result = "👎 부정 리뷰입니다."

    return render_template(
        "index.html",
        review=text,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)