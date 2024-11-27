# 社交大模型评测基准-生成
## 流程
1. 我们一共有 m 个角色，每个角色都有一个模拟的资料，每个资料都有确切的人格特质和得分。
2. 我们为每个角色准备了 n 条真实推文，每条推文都有一个潜在知识，每条推文都与用户简历、人格特质和潜在知识密切相关。
3. 根据上述信息构建请求体。
4. 要求 LLM 根据提示词发布推文。
5. 在收集到 LLM 的生成的推文后，我们根据以下标准对模型的性能进行评估：
    1. 重叠度和多样性
        1. Bleu。
        2. Rouge。
        3. Distinct。
        
    2. 大模型裁判
        1. 简历相关 (+1)。
        2. 人格相关 (+1)。
        3. 潜在知识相关 (+1)。

    3. 大五人格特质一致性
       1. 根据大模型生成的 n 条推文推断角色的人格特质得分。
       2. 比较角色的人格特质得分与真实人格特质得分的一致性。

## Usage
```bash
python main.py \
    --platform zhipuai \
    --base-url https://open.bigmodel.cn/api/paas/v4 \
    --api-key 120985c00120985c00120985c00 \
    --model glm-4-flash \
    --max-tokens 1024 \
    --temperature 0.6 \
    --top-p 0.7 \
    --platform-critic openai \
    --base-url-critic https://api.openai.com/v1 \
    --api-key-critic 120985c00120985c00120985c00 \
    --model-critic GPT-4o \
    --max-tokens-critic 1024 \
    --temperature-critic 0.01 \
    --convs-per-chunk 10 \
    --qps 30 \
    --qps-critic 30 \
    --max-retry-times 5
```

## Reference
要引用此工作，请使用：
```
@misc{huang2024orcaenhancingroleplayingabilities,
      title={Orca: Enhancing Role-Playing Abilities of Large Language Models by Integrating Personality Traits}, 
      author={Yuxuan Huang},
      year={2024},
      eprint={2411.10006},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.10006}, 
}
```

## License
OrcaBench 是在 [Apache-2.0 许可证](https://www.apache.org/licenses/LICENSE-2.0 许可证下发布的，详情请参阅[许可证](https://github.com/Airoura/OrcaBench/blob/main/LICENSE)。