import requests
import json
import re

def receive_gmlp(base_text):
    """
    gMLPの生成結果を受け取る関数
    Args:
        base_text: 初期のテキストデータ
    Returns:
        num_sentence: 生成された文の数
        re_text: 生成された文のリスト
    """
    url_get='http://192.168.11.9:8000'
    url_post='http://192.168.11.9:8000'

    response = requests.get(url_get)
    print(response.text+"\n")

    data = {
        'base_text':base_text
    }

    response = requests.post(url_post,json.dumps(data))
    re_text = response.text[1:-1]
    base_text = base_text + re_text
    re_text=[i for i in re.split("(?<=#)",re_text) if i !=""]
    num_sentence = str(len(re_text))
    print("RESPONSE STATUS CODE: "+str(response.status_code))
    # print("RESPONSE SEQ_SIZE:"+str(len(re_text)))
    # print(re_text)
        
    return num_sentence, re_text

# 使用例
if __name__ == "__main__":
    sample = receive_gmlp("*a1b3d9#")
    print(sample)
    print(type(sample))