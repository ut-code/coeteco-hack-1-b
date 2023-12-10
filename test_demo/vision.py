# 何を作るか
# - 回答
# -

import base64
from typing import Any

import api_key
from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return f'data:image/jpeg;base64{base64.b64encode(image_file.read()).decode("utf-8")}'


tips = ""

selected_flames: list[Any] = []

client = OpenAI(api_key=api_key.API_KEY)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "この画像を説明してください",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://chmod774.com/wp-content/uploads/2023/08/image.png",
                        "detail": "low",
                    },
                },
            ],
        }
    ],
    max_tokens=1000,
)

print(response.choices[0])
print(response.usage.prompt_tokens)
print(response.usage.completion_tokens)
