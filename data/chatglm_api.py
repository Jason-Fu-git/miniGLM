import zhipuai
import json

zhipuai.api_key = "3aa94056a78cd99d8919137da89fc3ca.ZZo0pJu5NJGulxUV"

with open('temp.txt', 'r') as f:
    texts = f.readlines()
    for text in texts:
        print('{"Question": "' + text.strip() + '", "Answer": "', end='')
        text += '请简短回答，不要使用换行符。'
        response = zhipuai.model_api.sse_invoke(
            model="chatglm_pro",
            prompt=[
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=10000,
        )
        for event in response.events():
            if event.event == "add":
                print(event.data, end='')
        print('"}')
