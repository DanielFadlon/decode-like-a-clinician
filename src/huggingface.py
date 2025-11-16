import os
from dotenv import load_dotenv
from huggingface_hub import login


def connect_to_hf():
    load_dotenv()
    access_token = os.getenv('HF_ACCESS_TOKEN')
    login(token=access_token)

