from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
DIR = "nlbb"
model = AutoModelForSeq2SeqLM.from_pretrained(DIR)
tokenizer = AutoTokenizer.from_pretrained(DIR)
model.to_bettertransformer()
model.to('cpu')
text = "he is a good boy"

inputs = tokenizer(text, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("pan_Guru"), max_length=512
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])