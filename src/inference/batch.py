import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def load_test_data_from_file(filepath):
    test_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                label, sentence = parts
                test_data.append((sentence.strip(), label.strip()))
    return test_data


def load_model_and_tokenizer(model_type, device, full_model_path, lora_adapter_path, base_model_name):
    if model_type == "full":
        model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
    elif model_type == "lora":
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        raise ValueError("Invalid model type.")
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_sentiment(text, model, tokenizer, device):
    labels = ["Negative", "Positive"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    return labels[predicted_class_id]


def run_batch_inference(full_model_path, lora_adapter_path, base_model, test_data_file, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = load_test_data_from_file(test_data_file)
    full_model, full_tokenizer = load_model_and_tokenizer("full", device, full_model_path, lora_adapter_path, base_model)
    lora_model, lora_tokenizer = load_model_and_tokenizer("lora", device, full_model_path, lora_adapter_path, base_model)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["Sentence", "Actual_Sentiment", "Full_Model_Prediction", "LoRA_Model_Prediction"]
        csv_writer.writerow(header)
        for sentence, actual_label in test_data:
            full_pred = predict_sentiment(sentence, full_model, full_tokenizer, device)
            lora_pred = predict_sentiment(sentence, lora_model, lora_tokenizer, device)
            csv_writer.writerow([sentence, actual_label, full_pred, lora_pred])


