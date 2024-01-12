from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def main():
    # пользуемся GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'jbochi/madlad400-3b-mt'
    # Загружаем модель
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Переводим модель в GPU
    model = model.to(device)

    while True:
        # Просим пользователя ввести текст 
        input_text = input("Enter text to translate (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break

        full_input_text = "<2ru>" + input_text

        # токенезируем введенный текст, возвращаем тенсоры
        input_ids = tokenizer.encode(full_input_text, return_tensors="pt").to(device)

        # Генерируем результат
        outputs = model.generate(
            input_ids=input_ids,
            max_length=512,
            num_beams=2,
            no_repeat_ngram_size=2,
            length_penalty=1.2
        )

        # Декодируем тенсоры в текст и выводим
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Translated text (Russian):", translated_text)

if __name__ == "__main__":
    main()