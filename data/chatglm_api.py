import zhipuai
import json

zhipuai.api_key = "3aa94056a78cd99d8919137da89fc3ca.ZZo0pJu5NJGulxUV"

for i in range(100):
    text = "请生成关于《射雕英雄传》的一个问题"

    response = zhipuai.model_api.sse_invoke(
        model="chatglm_pro",
        prompt=[
            {"role": "user", "content": text},
        ],
        temperature=0.5,
        top_p=0.7,
        max_tokens=10000,
    )
    for event in response.events():
        if event.event == "add":
            print(event.data, end='')
    print()
