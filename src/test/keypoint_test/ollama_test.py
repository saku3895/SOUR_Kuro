from ollama import Client

# 🔥 PCのIPアドレスをここに入れます
pc_ip = 'http://192.168.11.5:11434'

try:
    print("🌐 PCのOllamaに接続テスト中...")
    client = Client(host=pc_ip)
    
    response = client.chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': 'Hello from Raspberry Pi!'}],
    )
    print("\n🎉 接続成功！PCからの返答:")
    print(response.message.content)

except Exception as e:
    print("\n❌ 接続エラーが発生しました。以下の原因が考えられます：")
    print(e)
    print("\n【チェックリスト】")
    print("1. PCとラズパイが同じWi-Fi（ネットワーク）に繋がっていますか？")
    print("2. PC側で 'ollama serve' が動いていますか？")