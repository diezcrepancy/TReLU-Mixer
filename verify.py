import requests

api_key = "f257425fc1ba4ec974c5f9098d7f55224beffa8d"  # Замените на реальный ключ
headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get("https://api.wandb.ai", headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}")