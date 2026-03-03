import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap

MODEL_NAME = "facebook/bart-large-cnn"

def load_model():
    print("Loading model (first time may take a while)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def summarize_text(tokenizer, model, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def main():
    tokenizer, model = load_model()

    print("\nPaste your article below.")
    print("Press ENTER twice when finished:\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    article = " ".join(lines)

    if not article.strip():
        print("No text provided.")
        return

    print("\nGenerating summary...\n")
    summary = summarize_text(tokenizer, model, article)

    print("===== SUMMARY =====\n")
    print(textwrap.fill(summary, width=100))


if __name__ == "__main__":
    main()