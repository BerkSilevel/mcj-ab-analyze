import json

# JSON key dosyasını oku
with open("service_account.json", "r") as f:
    service_account_json = f.read()

# Escape karakterlerini ekle
escaped = service_account_json.replace("\\", "\\\\").replace('"', '\\"').replace('\n', '\\n')

# TOML formatı olarak bastır
print('GOOGLE_APPLICATION_CREDENTIALS_JSON = """')
print(escaped)
print('"""')
