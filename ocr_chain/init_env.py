import os


def init_api_key():
    with open(".env") as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.split("=")[:2]
            os.environ[key] = value.strip()
