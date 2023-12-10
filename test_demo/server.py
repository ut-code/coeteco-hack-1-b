from flask import Flask, request, jsonify, render_template
import os
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# アップロードされたファイルの保存先
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    # 受け取ったリクエストから動画2つと文字列を取得
    video1 = request.files["video1"]
    video2 = request.files["video2"]
    input_text = request.form["input_text"]

    # アップロードされた動画を保存
    video1_filename = secure_filename(video1.filename)
    video2_filename = secure_filename(video2.filename)
    video1_path = os.path.join(app.config["UPLOAD_FOLDER"], video1_filename)
    video2_path = os.path.join(app.config["UPLOAD_FOLDER"], video2_filename)
    video1.save(video1_path)
    video2.save(video2_path)

    # 適当な処理を行う（ここではファイル名を変更）
    processed_video1_path = os.path.join(
        app.config["UPLOAD_FOLDER"], "processed_video1.mp4"
    )
    processed_video2_path = os.path.join(
        app.config["UPLOAD_FOLDER"], "processed_video2.mp4"
    )
    os.rename(video1_path, processed_video1_path)
    os.rename(video2_path, processed_video2_path)

    # 処理結果をMultipart/form-data形式でクライアントに返す
    files = {
        "processed_image1": (
            "processed_image1.jpg",
            open("processed_image1.jpg", "rb"),
        ),
        "processed_image2": (
            "processed_image2.jpg",
            open("processed_image2.jpg", "rb"),
        ),
        "processed_text": ("processed_text.txt", input_text.encode("utf-8")),
    }
    response = requests.post("http://client-endpoint/upload_processed", files=files)

    return jsonify({"message": "Files processed and sent successfully"})


if __name__ == "__main__":
    app.run(debug=True)
