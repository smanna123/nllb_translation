import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_and_tokenizer(checkpoint, directory):
    try:
        # Load environment variables
        load_dotenv()
        access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        if access_token is None:
            logging.error("Hugging Face access token not found.")
            raise ValueError("Hugging Face access token not found in environment variables.")

        # Initialize tokenizer and model
        logging.info(f"Loading the tokenizer and model for checkpoint: {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

        # Save tokenizer and model
        logging.info(f"Saving the tokenizer and model to directory: {directory}")
        tokenizer.save_pretrained(directory)
        model.save_pretrained(directory)

        logging.info("Model and tokenizer have been saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    DIR = "nlbb_distilled"
    CHECKPOINT = "facebook/nllb-200-distilled-1.3B"
    load_model_and_tokenizer(CHECKPOINT, DIR)
