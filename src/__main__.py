from .model_saver import save_model, save_model_using_lib

if __name__ == "__main__":
    #model name
    model_id = "google/flan-t5-large"
    save_model_using_lib(model_id)
    # use_model("cpu", "D:\AI_Planat\Models\Raw_models\modelsGensyn\Qwen2.5-0.5B-Instruct", "D:\AI_Planat\Tokenizers\Raw_tokenizers\Gensyn\Qwen2.5-0.5B-Instruct", "When was the Eurepean Union established?")