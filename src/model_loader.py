import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import threading

generation_control = {"cancel": False}

def cancel_generation():
    generation_control["cancel"] = True

def reset_generation():
    generation_control["cancel"] = False

def quantization(load, type, double_quant, compute_dtype):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=load,
        bnb_4bit_quant_type=type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype
    )
    return nf4_config

_loaded_models = {}
_loaded_tokenizers = {}

def prepare_models():
    models_path = "../../Models/Models_for_testing_app"
    tokenizers_path = "../../Tokenizers/Tokenizers_for_testing_app"

    if not os.path.exists(models_path) or not os.path.exists(tokenizers_path):
        raise RuntimeError("Models or Tokenizers directory does not exist")
    
    for model_name in os.listdir(models_path):
        model_dir = os.path.join(models_path, model_name)
        tokenizer_dir = os.path.join(tokenizers_path, model_name)

        if not os.path.isdir(model_dir) or not os.path.isdir(tokenizer_dir):
            continue

        print(f"🔄 Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

        if "t5" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

        _loaded_models[model_name] = model
        _loaded_tokenizers[model_name] = tokenizer

    print(f"Preloaded models: {list(_loaded_models.keys())}")

def get_model_from_cache(model_name: str):
    if model_name not in _loaded_models or model_name not in _loaded_tokenizers:
        raise ValueError(f"Model '{model_name}' is not loaded in cache. Call prepare_models() first.")
    return _loaded_models[model_name], _loaded_tokenizers[model_name]

def stream_model_response(model, tokenizer, prompt, current_generation={"cancel": False}):
    
    # Encode prompt and create attention mask explicitly
    inputs = tokenizer(prompt, return_tensors="pt")
    attention_mask = inputs["attention_mask"]

    # Create streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,
            streamer=streamer,
        )
    )
    generation_thread.start()

    for new_text in streamer:
        # print("STREAM:", repr(new_text))
        if generation_control["cancel"]:
            break
        yield new_text

