from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your_api_key")

stream = client.chat.completions.create(
    model="Marshall-gemma3-270m",
    messages=[
        {"role": "system", "content": "Tentukan apakah feedback dari pengguna ini positif, negatif, atau netral. Hanya jawab dengan satu kata."},
        {"role": "user", "content": "Film merah putih one for all absolute trash"},
    ],
    temperature=0.7,
    max_completion_tokens=512,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
