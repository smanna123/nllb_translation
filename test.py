from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.bettertransformer import BetterTransformer
DIR = "nlbb"
model = AutoModelForSeq2SeqLM.from_pretrained(DIR)
model = BetterTransformer.transform(model)
print(model)