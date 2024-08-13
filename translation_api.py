from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize the FastAPI app
app = FastAPI()

# Load the model and tokenizer
DIR = "nlbb"
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(DIR)
    tokenizer = AutoTokenizer.from_pretrained(DIR)
    model.to('cpu')
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model or tokenizer: {str(e)}")


# Define the request model
class TranslationRequest(BaseModel):
    text: str
    forced_bos_token: str


@app.post("/translate/")
def translate(request: TranslationRequest):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.text, return_tensors="pt")
        # Convert the dynamic forced_bos_token to its ID
        bos_token_id = tokenizer.convert_tokens_to_ids(request.forced_bos_token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization error: {str(e)}")

    try:
        # Generate the translated tokens
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=bos_token_id, max_length=512
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model generation error: {str(e)}")

    try:
        # Decode the tokens and return the translated text
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding error: {str(e)}")

    return {"translated_text": translated_text}
