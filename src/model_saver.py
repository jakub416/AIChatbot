from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
import os

def save_model_using_lib(model_name):
    HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HUGGING_FACE_API_KEY)
    model = AutoModelForCausalLM.from_pretrained(model_name, token = HUGGING_FACE_API_KEY)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token = HUGGING_FACE_API_KEY)
    model.save_pretrained(f"D:\AI_Planat\Models\Raw_models\models{model_name}")
    tokenizer.save_pretrained(f"D:\AI_Planat\Tokenizers\Raw_tokenizers\{model_name}")

def save_model(model_name):
    HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
    filenames = []
    for filename in filenames:
        downloaded_model_path = hf_hub_download(model_name, filename=filename, token=HUGGING_FACE_API_KEY)
        print(downloaded_model_path)