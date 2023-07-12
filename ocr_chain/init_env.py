import os


def init_api_key():
    with open(".env") as f:
        line = f.readlines()[0]
        os.environ['OPENAI_API_KEY'] = line.split("=")[-1].strip()
        os.environ['SERPAPI_API_KEY'] = "c8414793ab4bf091659c484714c1b409b42779c5c535421bdda5beccd05aa776"
