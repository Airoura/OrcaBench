python main.py \
    --platform openai \
    --base-url https://api.fireworks.ai/inference/v1 \
    --api-key NkUNSHbuSYFvsBZYxq1WhQTIWibG7Q3foDAQHRnYqyALHDRA \
    --model accounts/fireworks/models/llama-v3p1-8b-instruct \
    --max-tokens 1024 \
    --temperature 0.6 \
    --top-p 0.7 \
    --platform-critic openai \
    --base-url-critic http://172.16.64.188:8000/v1 \
    --api-key-critic - \
    --model-critic llama3.1-70b \
    --max-tokens-critic 1024 \
    --temperature-critic 0.01 \
    --top-p-critic 0.7 \
    --convs-per-chunk 10 \
    --qps 20 \
    --qps-critic 20 \
    --max-retry-times 5 \
    --ablation profile