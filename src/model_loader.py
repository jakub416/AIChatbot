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

def stream_model_response(model_path, tokenizer_path, prompt, current_generation = {"cancel": False}):
    assert os.path.exists(model_path), f"Model path does not exist: {model_path}"
    assert os.path.exists(tokenizer_path), f"Tokenizer path does not exist: {tokenizer_path}"

    #quantization WE NEED CUDA!!!
    # nf4_config = quantization(True, "nf4", True, torch.bfloat16)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, local_files_only=True)
    if "t5" in str(model_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    # Encode prompt and create attention mask explicitly
    inputs = tokenizer(prompt, return_tensors="pt")
    attention_mask = inputs["attention_mask"]

    # Create streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            inputs=inputs.input_ids,
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

