import os
import sys
import random
import time
import datetime
import glob
import re
import shutil
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import datetime
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.svm import SVC


def movie_to_frame_keypoints(output_folder, video_path):
    # YOLOモデルの設定
    model = YOLO(r"./yolov8n-pose.pt")  # お使いのファイルに合わせて変更してください
    # キャプチャの設定
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Display size scale
    scale = 1
    # GPUを使用してYOLOv8による推論を行うための設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # バッチサイズ
    batch_size = 32
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ファイルへの書き込み用のテキストファイルをオープン
    output_txt_file = open(output_folder + "/segmentation_estimation_results.txt", "w")
    output_txt_file_keypoints = open(
        output_folder + "/pose_estimation_results.txt", "w"
    )

    # フレーム処理のためのループ　いったん最終データのみ取る
    frame_count = 0
    while cap.isOpened():
        # バッチサイズ分のフレームを読み込む
        batch_frames = []
        for _ in range(batch_size):
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                batch_frames.append(frame)
            # もし次のフレームがない場合は、最後のフレームをコピーする
            else:
                # 最初のフレームが存在する場合にのみ最後のフレームをコピーする
                if len(batch_frames) > 0:
                    last_frame = batch_frames[-1].copy()
                    # データが不足する場合、最後のフレームで埋める
                    missing_frames = batch_size - len(batch_frames)
                    for _ in range(missing_frames):
                        batch_frames.append(last_frame)
                break  # これを追加してループを抜ける

        if not batch_frames:
            break

        # YOLOv8による推論をバッチで実行
        results_batch = model.track(
            source=batch_frames,
            show=False,
            save=True,
            device=device,
            conf=0.5,
            save_txt=True,
            save_conf=True,
            tracker="bytetrack.yaml",
            classes=[0],
        )

        # バッチ内の各フレームの推論結果を処理
        for i, results in enumerate(results_batch):
            if results is not None and len(results) > 0:
                # 各情報を取得
                boxes = results[0].boxes.xyxy[0, :].cpu().numpy().astype(int)
                names = results[0].names
                orig_img = results[0].orig_img
                orig_shape = results[0].orig_shape
                speed = results[0].speed

                output_folder_frame = f"{output_folder}/frame_{frame_count:04d}"
                frame_filename = f"{output_folder_frame}.jpg"
                cv2.imwrite(frame_filename, orig_img)

                # 推論結果の情報をテキストファイルに書き込む
                output_txt_file.write(f"Frame {frame_count}, Box {i}:\n")
                output_txt_file.write(f"  Boxes: {boxes}\n")
                output_txt_file.write(f"  Names: {names}\n")
                output_txt_file.write(f"  Speed: {speed}\n")

                # マスク情報が存在する場合のみ保存
                if results[0].masks is not None:
                    masks = (
                        results[0].masks[0].mul(255).byte().cpu().numpy()
                    )  # 二値画像としてマスクを取得
                    mask_filename = f"{output_folder_frame}/output_mask_{i:02d}.png"
                    cv2.imwrite(mask_filename, masks)

                # 姿勢推定結果をテキストファイルに書き込む
                if results[0].keypoints is not None:
                    # output_txt_file_keypoints.write(f"{results[0].keypoints.conf},{results[0].keypoints.xyn}\n")
                    # テンソルをCPUに転送してから1次元に変換
                    flattened_tensor_conf = results[0].keypoints.conf.cpu().view(-1)
                    flattened_tensor_xyn = results[0].keypoints.xyn.cpu().view(-1)

                    # テンソルの各要素を1行に書き込む
                    output_txt_file_keypoints.write(
                        ",".join(map(str, flattened_tensor_conf.tolist()))
                    )
                    output_txt_file_keypoints.write(",")
                    output_txt_file_keypoints.write(
                        ",".join(map(str, flattened_tensor_xyn.tolist())) + "\n"
                    )
                frame_count += 1

    # テキストファイルをクローズ
    output_txt_file.close()

    # Release the video capture object
    cap.release()
    df1 = pd.read_csv(
        output_folder + "/pose_estimation_results.txt",
        header=None,
        names=[
            "r0",
            "r1",
            "r2",
            "r3",
            "r4",
            "r5",
            "r6",
            "r7",
            "r8",
            "r9",
            "r10",
            "r11",
            "r12",
            "r13",
            "r14",
            "r15",
            "r16",
            "x0",
            "y0",
            "x1",
            "y1",
            "x2",
            "y2",
            "x3",
            "y3",
            "x4",
            "y4",
            "x5",
            "y5",
            "x6",
            "y6",
            "x7",
            "y7",
            "x8",
            "y8",
            "x9",
            "y9",
            "x10",
            "y10",
            "x11",
            "y11",
            "x12",
            "y12",
            "x13",
            "y13",
            "x14",
            "y14",
            "x15",
            "y15",
            "x16",
            "y16",
        ],
    )
    df1_sub = df1.loc[
        :, ["r5", "r6", "r8", "r10", "x5", "y5", "x6", "y6", "x8", "y8", "x10", "y10"]
    ]

    def calc_i(x1, y1, x2, y2, x3, y3):
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x1, y3 - y1])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            raise ValueError("zero vector")
        # internal product
        return (
            np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))
            / np.pi
            * 180
        )

    df1_sub["migihizi_r"] = df1.apply(
        lambda x: (x["r6"] + x["r8"] + x["r10"]) / 3, axis=1
    )
    df1_sub["migikata_r"] = df1.apply(
        lambda x: (x["r5"] + x["r8"] + x["r6"]) / 3, axis=1
    )
    df1_sub = df1_sub.dropna(subset=["migihizi_r", "migikata_r"])
    df1_sub["migihizi_theta"] = df1.apply(
        lambda x: calc_i(x["x8"], x["y8"], x["x6"], x["y6"], x["x10"], x["y10"]), axis=1
    )
    df1_sub["migikata_theta"] = df1.apply(
        lambda x: calc_i(x["x6"], x["y6"], x["x8"], x["y8"], x["x5"], x["y5"]), axis=1
    )
    df1_sub = df1_sub.dropna(subset=["migihizi_theta", "migikata_theta"])

    theta_migihizi = np.dot(
        np.array(df1_sub["migihizi_theta"]), np.array(df1_sub["migihizi_r"])
    ) / np.sum(np.array(df1_sub["migihizi_r"]))
    theta_migikata = np.dot(
        np.array(df1_sub["migikata_theta"]), np.array(df1_sub["migikata_r"])
    ) / np.sum(np.array(df1_sub["migikata_r"]))

    last_frame_path = output_folder + "/frame_{:04d}.jpg".format(frame_count - 1)

    return df1, theta_migihizi, theta_migikata, last_frame_path


def predict_kata(theta_migikata, data_train_all):
    # ラベル列を含める
    X = data_train_all.loc[:, ["theta_kata"]]
    y = data_train_all["label"]

    # データをトレーニングセットとテストセットに分割
    # SVMモデルを作成
    clf = SVC()

    # モデルをトレーニングセットで学習
    clf.fit(X, y)

    # 任意の新しいデータに対して予測
    new_data = pd.DataFrame([[theta_migikata]], columns=["theta_kata"])
    prediction = clf.predict(new_data)
    return prediction


def predict_hizi(theta_migihizi, data_train_all):
    # ラベル列を含める
    X = data_train_all.loc[:, ["theta_hizi"]]
    y = data_train_all["label"]

    # データをトレーニングセットとテストセットに分割
    # SVMモデルを作成
    clf = SVC()

    # モデルをトレーニングセットで学習
    clf.fit(X, y)

    # 任意の新しいデータに対して予測
    new_data = pd.DataFrame([[theta_migihizi]], columns=["theta_hizi"])
    prediction = clf.predict(new_data)
    return prediction


import io
import time

import api_key
import auth
import requests
from moviepy.editor import VideoFileClip
from openai import OpenAI
from PIL import Image


def pil_to_binary(image: Image.Image):
    with io.BytesIO() as binary_stream:
        image.save(binary_stream, format="JPEG")
        binary_stream.seek(0)
        return binary_stream.read()


def invoke_gpt(messages):
    response = OpenAI(api_key=api_key.API_KEY).chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=500,
    )

    print(response.usage.prompt_tokens)
    print(response.usage.completion_tokens)
    return response.choices[0]


def upload_image_to_imgur(image: Image.Image):
    url = "https://api.imgur.com/3/image"
    client_id = auth.ID
    headers = {"Authorization": f"Client-ID {client_id}"}

    files = {"image": ("image.jpg", pil_to_binary(image))}
    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        print(response.json()["data"]["deletehash"])
        return response.json()
    else:
        print(f"POST request failed with status code {response.status_code}")
        return None


def delete_image_from_imgur(image_hash: str):
    url = f"https://api.imgur.com/3/image/{image_hash}"
    client_id = auth.ID
    headers = {"Authorization": f"Client-ID {client_id}"}

    response = requests.delete(url, headers=headers)

    if response.status_code == 200:
        return True
    else:
        print(f"DELETE request failed with status code {response.status_code}")
        return False


class Processor:
    api_key = api_key.API_KEY

    def __init__(self, movie_file: str, tips: str, kata_score, hizi_score):
        self.movie_file = movie_file
        self.tips = tips
        self.flame_gap = 20
        self.image_hashes: list[str] = []
        self.kata_score = kata_score
        self.hizi_score = hizi_score

    def process(self):
        clip = VideoFileClip(self.movie_file)
        frames = clip.iter_frames(fps=1, dtype="uint8")
        for i, frame in enumerate(frames):
            if i % self.flame_gap == 0:
                print(self.process_image(Image.fromarray(frame)))

    def process_image(self, image: Image.Image):
        converted_image = image.convert("RGB")
        resized_image = self.resize_image(converted_image)
        encoded_image, delete_hash = self.encode_image(resized_image)
        try:
            return self.introduce_image(encoded_image)
        finally:
            delete_image_from_imgur(delete_hash)

    def resize_image(self, image: Image.Image):
        default_height = 400
        default_width = 600
        new_height = int(image.height / image.width * default_width)
        new_width = int(image.height / image.width * default_height)
        if new_height < default_height:
            new_size = (new_height, default_width)
        else:
            new_size = (default_height, new_width)
        return image.resize(new_size)

    def encode_image(self, image: Image.Image):
        response = upload_image_to_imgur(image)
        if response:
            return response["data"]["link"], response["data"]["deletehash"]
        raise Exception(response)

    #     def introduce_image(self, encoded_image: str):
    #         choice = invoke_gpt(
    #             messages=[
    #                 {
    #                     "role": "assistant",
    #                     "content": f"TIPS:\n{self.tips}",
    #                 },
    #                 {
    #                     "role": "system",
    #                     "content": """
    # 画像に対し、次にしたがって小学校の低学年の体育の先生としてわかりやすく回答してください.
    # - 対象: 最も大きく写っている人物
    # - 内容:
    #  1. TIPSのどの段階(STEP数・丸付き数字の番号)か判別する
    #  2. 体の各部位の姿勢のうち、TIPSの該当する番号の記述を満たしているかどうか.該当なしの場合、STEP2の②であるので、その場合、手の位置について細かく述べること.文章の最後に、「合格です」か「もう少しがんばりましょう」を、記述への適合度をもとに記入すること.
    # -注意点：画像が斜め方向からとられている場合は、腕の上げ具合に気を付けること。つまり垂直にみえるものは実際は体操選手のようにうまく上げている場合が多く、角度が垂直に見えないものは必ず不十分であることに注意すること。
    # - 形式: できるだけ簡潔に要約された、「ですます調」の箇条書き
    # """,
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {
    #                             "type": "image_url",
    #                             "image_url": encoded_image,
    #                         },
    #                     ],
    #                 },
    #             ],
    #         )
    #         return choice.message.content

    def introduce_image(self):
        choice = invoke_gpt(
            messages=[
                {
                    "role": "assistant",
                    "content": f"TIPS:\n{self.tips}",
                },
                {
                    "role": "system",
                    "content": f"""
あなたは小学校低学年の体育の先生で、授業内で児童が前転の練習をしていてとくに、前転の最後に体操選手が手を挙げてポーズをとるところを練習していて、あなたはその様子を見ているとします。重要でできるべきポイントは①右肩、右ひじ、右手首が一直線になるようにする。
②体操選手のように右腕をしっかり上にあげる。
このとき、児童の評価は①は{self.hizi_score}(1:上手/0:練習がまだ必要)で、②は{self.kata_score}(1:上手/0:練習がまだ必要)です。これをもとに児童の撮影した動画に対するフィードバックを児童が理解できるようになるべくひらがなでわかりやすく記述してください。
上手であればそれは褒めてあげてください。練習が必要な場合は、どのようにすればよいかを具体的に記述してください。ただし、スコアについては０，１ではなくもっとがんばろうとかごうかくとかにおきかえてふぃーどばっくにはちょくせつかかないでください。
""",
                },
                {"role": "user", "content": "Nothing"},
            ],
        )
        return choice.message.content


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
    video1 = request.files["modelVideo"]
    video2 = request.files["studentVideo"]
    input_text = request.form["tipsText"]

    # アップロードされた動画を保存
    video1_filename = secure_filename(video1.filename)
    video2_filename = secure_filename(video2.filename)
    video1_path = os.path.join(app.config["UPLOAD_FOLDER"], video1_filename)
    video2_path = os.path.join(app.config["UPLOAD_FOLDER"], video2_filename)
    video1.save(video1_path)
    video2.save(video2_path)

    # 処理結果をMultipart/form-data形式でクライアントに返す
    files = {
        "model_video": (
            "model_video.mp4",
            open(video1_path, "rb"),
        ),
        "your_video": (
            "your_video.mp4",
            open(video2_path, "rb"),
        ),
        "gpt_message": ("gpt_message.txt", main("test", video2_path)),
    }
    return jsonify(files)


if __name__ == "__main__":
    app.run(debug=True, port=5000)


def main(output_folder, video_path):
    df1, theta_migihizi, theta_migikata, last_frame_path = movie_to_frame_keypoints(
        output_folder, video_path
    )
    data_train_all = pd.read_csv(
        "data_train_all.csv",
        header=None,
        names=["theta_hizi", "theta_kata", "label"],
        skiprows=1,
    )
    kata_score = predict_kata(theta_migikata, data_train_all)
    hizi_score = predict_hizi(theta_migihizi, data_train_all)
    print("kata:", kata_score, "hizi", hizi_score)
    return (
        Processor(
            "",
            """STEP1: ①右肩、右ひじ、右手首が一直線になるようにしたうえで、体操選手のように右腕がかなりうえにあがっているようにする。""",
            kata_score,
            hizi_score,
        ).process_image(Image.open(last_frame_path)),
        last_frame_path,
    )
