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
        max_tokens=300,
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

    def __init__(self, movie_file: str, tips: str):
        self.movie_file = movie_file
        self.tips = tips
        self.flame_gap = 20
        self.image_hashes: list[str] = []

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

    def encode_image(self, image: Image.Image) -> tuple[str, str]:
        response = upload_image_to_imgur(image)
        if response:
            return response["data"]["link"], response["data"]["deletehash"]
        raise Exception(response)

    def introduce_image(self, encoded_image: str):
        choice = invoke_gpt(
            messages=[
                {
                    "role": "assistant",
                    "content": f"TIPS:\n{self.tips}",
                },
                {
                    "role": "system",
                    "content": """
画像に対し、次にしたがって回答してください.
- 対象: 最も大きく写っている人物
- 内容:
 1. TIPSのどの段階(STEP数・丸付き数字の番号)か判別する
 2. 体の各部位の姿勢のうち、TIPSの該当する番号の記述を満たしているかどうか.該当なしの場合、STEP2の②であるので、その場合、手の位置について細かく述べること.文章の最後に、「合格です」か「もう少しがんばりましょう」を、記述への適合度をもとに記入すること.
- 形式: できるだけ簡潔に要約された、「ですます調」の箇条書き
""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": encoded_image,
                        },
                    ],
                },
            ],
        )
        return choice.message.content


if __name__ == "__main__":
    start = time.time()
    print(
        Processor(
            "",
            """STEP1. 回り方 ①マットに足を揃えて立ち、しゃがんで手をつきます。 POINT手は肩幅程度に開く。 ②準備姿勢に入ります。足を閉じてお尻を頭より高く上げていきましょう。 ③ひじを曲げながら体重を前に移動し、後頭部が床につくようにあごを引きます。後頭部・背中・腰の順番で床につくように身体のラインを丸めて回りましょう。 POINTあごを引いて、しっかりとおへそを覗き込む。 ④回転の途中で、お腹と太ももを離さないように注意しましょう。 ⑤両足でマットを押し込むイメージで床を蹴ることが、美しく前転を行うコツです。 STEP2. 立ち上がり方 ①回転の後半に入ったら、お尻のほうにかかとを引きつけましょう。回転に勢いがつき、上半身を起こしやすくなります。 ②体育座りの姿勢から足の裏をしっかりと床につけ、両手を前に出して立ち上がりましょう。""",
        ).process_image(
            Image.open("/Users/nishiyama.takuki/Downloads/output_frame_06.jpg")
        )
    )
    print(time.time() - start)
