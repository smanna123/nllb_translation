from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from queue import Queue
import threading
import torch

# Initialize the FastAPI app
app = FastAPI()

# Load the model
DIR = "nlbb_distilled"
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(DIR)
    model.to('cuda')  # Move model to GPU
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Prepare a pool of tokenizers
NUM_TOKENIZERS = 4  # You can adjust the number based on your server's capability
tokenizer_queue = Queue()
for _ in range(NUM_TOKENIZERS):
    try:
        tokenizer = AutoTokenizer.from_pretrained(DIR)
        tokenizer_queue.put(tokenizer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tokenizer: {str(e)}")

# Define the request model
class TranslationRequest(BaseModel):
    text: str
    forced_bos_token: str

@app.post("/translate/")
def translate(request: TranslationRequest):
    tokenizer = tokenizer_queue.get()  # Get a tokenizer from the queue
    try:
        # Tokenize the input text
        #inputs = tokenizer(request.text, return_tensors="pt")
        inputs = tokenizer(request.text, return_tensors="pt").to('cuda')
        bos_token_id = tokenizer.convert_tokens_to_ids(request.forced_bos_token)
        bos_token_id = torch.tensor([bos_token_id], device='cuda')

        # Generate the translated tokens
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=bos_token_id, max_length=512
        )
        # Decode the tokens and return the translated text
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tokenizer_queue.put(tokenizer)  # Return the tokenizer to the queue

    return {"translated_text": translated_text}