import requests
import json


class PeitClient:
    def __init__(self, base_url, max_tokens=1024, temperature=0.6, top_p=0.7):
        self.base_url=base_url
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.top_p=top_p
    
    def request(self, prompt, score):
        # 定义请求的URL
        url = self.base_url
        # 定义请求参数
        data = {
           'prompt': prompt,
           'score': score
        }
        # 将字典转换为JSON字符串
        json_data = json.dumps(data)
        # 发起POST请求
        response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})
        # 检查响应状态码
        response = response.json()["result"]
        return response

    def test(self):
        prompt = "\n### Instruction\nYou are [Maude].\nYour resume and personality are as follows. Firstly, express your current psychological activities, and then post a tweet based on these information. \nIf you want to send images, please add the description information of the images to the Media array.\nPaying attention to potential knowledge will provide you with some additional information to help you clarify the ins and outs of things.\n\n### Requirement\nResponse in strict accordance with the following json format:\n{\n    \"Psychological Activities\": \"\",\n    \"Tweet Content\": \"\",\n    \"Media\": [\n        {\n            \"type\": \"image\", \n            \"content\": \"\"\n        }\n    ]\n}\n\n### Resume\n\n\n### Personality\n<|reserved_special_token_0|><|reserved_special_token_0|><|reserved_special_token_0|><|reserved_special_token_0|><|reserved_special_token_0|>\n\n### Potential Knowledge\nThe conversation appears to be referencing two women, Stormy Daniels and E. Jean Carroll, who have been involved in high-profile controversies with former US President Donald Trump. Stormy Daniels is an adult film actress who alleged that she had an affair with Trump in 2006, and was paid $130,000 in hush money by Trump's lawyer Michael Cohen during the 2016 presidential campaign. E. Jean Carroll is a journalist and advice columnist who accused Trump of raping her in the 1990s. The conversation seems to be suggesting that these women should be treated with respect and rewarded for their courage in speaking out against Trump, while also mocking Michael Cohen's involvement in the scandals. The mention of 'a free side of mayo' is likely a humorous way of downplaying Cohen's punishment, implying that he should receive a relatively minor penalty compared to the severity of his actions.\n\n### Response\n"
        score = [
            3.6923076923076925,
            0.8461538461538461,
            0.0,
            0.6153846153846154,
            0.23076923076923078,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.2307692307692308,
            0.0,
            0.15384615384615385,
            1.0,
            0.07692307692307693,
            0.0,
            0.0,
            0.5384615384615384,
            0.0,
            0.5384615384615384,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]
        return self.request(prompt, score)