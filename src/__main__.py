from .model_saver import save_model, save_model_using_lib
from .model_loader import test_model


if __name__ == "__main__":
    #model name
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    save_model_using_lib(model_id)
    # use_model("cpu", "D:\AI_Planat\Models\Raw_models\modelsGensyn\Qwen2.5-0.5B-Instruct", "D:\AI_Planat\Tokenizers\Raw_tokenizers\Gensyn\Qwen2.5-0.5B-Instruct", "When was the Eurepean Union established?")
    prompt = """Look at the following customer reviews and tell me which review was the most negative:

1. The response time is perfect. The keys feel just right and don’t make a loud clack. The lights are cool especially if you like to keep your room dark. Just a great keyboard at a nice price.
2. It suits me well, as described.
3. I love this keyboard. It is really nice to type on. I love the scroll volume control and I like the colourful lighting.
4. worst keyboard i ever used because it is too noisy
5. It is a brilliant, a beautiful keyboard. Plenty of customizations.
"""

# print(test_model("D:/AI_Planat/Models/Raw_models/modelsdeepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "D:/AI_Planat/Tokenizers/Raw_tokenizers/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", prompt))