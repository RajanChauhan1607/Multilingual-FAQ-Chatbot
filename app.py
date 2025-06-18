#lets go
# Import libraries
import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import io


# safe translation pipeline loader
def get_translator(model_name):
    try:
        return pipeline(
            "translation",
            model=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f" Failed to load {model_name}: {e}")
        return None


# Load translation pipelines
translator_to_english = {
    "es": get_translator("Helsinki-NLP/opus-mt-es-en"),
    "fr": get_translator("Helsinki-NLP/opus-mt-fr-en"),
    "hi": get_translator("Helsinki-NLP/opus-mt-hi-en")
}

translator_from_english = {
    "es": get_translator("Helsinki-NLP/opus-mt-en-es"),
    "fr": get_translator("Helsinki-NLP/opus-mt-en-fr"),
    "hi": get_translator("Helsinki-NLP/opus-mt-en-hi")
}


# Translation functions
def translate_to_english(text, lang):
    try:
        if lang != "en" and translator_to_english.get(lang):
            translated = translator_to_english[lang](text)[0]['translation_text']
            print(f" {lang} → en: '{text}' → '{translated}'")
            return translated
        return text
    except Exception as e:
        print(f" translate_to_english error: {e}")
        return text


def translate_from_english(text, lang):
    try:
        if lang != "en" and translator_from_english.get(lang):
            translated = translator_from_english[lang](text)[0]['translation_text']
            print(f" en → {lang}: '{text}' → '{translated}'")
            return translated
        return text
    except Exception as e:
        print(f"translate_from_english error: {e}")
        return text


# Load sentence embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Default FAQ dataset
default_faq = [
    ["What are the application deadlines?",
     "The deadline for Fall 2025 is June 30, 2025. For Spring 2026, it's November 15, 2025."],
    ["How do I apply for scholarships?",
     "Submit the scholarship application form along with your admission application by the deadline."],
    ["What documents are required?",
     "You need transcripts, a statement of purpose, two recommendation letters, and a valid ID proof."],
    ["Is there an application fee?",
     "Yes, the application fee is $50 for domestic students and $75 for international students."],
    ["Do you offer online courses?", "Yes, we offer hybrid and fully online programs for select majors."],
    ["Can international students apply?",
     "Yes, international students are welcome and must meet the admission requirements."],
    ["Can I get admission here?",
     "Yes, you can apply if you meet the eligibility criteria listed on our admissions page."],
    ["Is financial aid available?", "Yes, need-based and merit-based financial aid options are available."],
    ["What are the English proficiency requirements?",
     "TOEFL or IELTS scores are required for non-native English speakers."],
    ["Are there on-campus housing options?",
     "Yes, on-campus housing is available and students can apply after admission."]
]

faq_questions = [q for q, a in default_faq]
faq_answers = [a for q, a in default_faq]
faq_embeddings = embedding_model.encode(faq_questions, convert_to_tensor=True)


# File upload support
def load_faq_file(file):
    global faq_questions, faq_answers, faq_embeddings
    try:
        if file is None:
            return " No file uploaded."

        file_bytes = file.read()
        if file.name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return " Upload a CSV or XLSX with 'question' and 'answer' columns."

        if not {'question', 'answer'}.issubset(df.columns):
            return " File must contain 'question' and 'answer' columns."

        faq_questions = df["question"].tolist()
        faq_answers = df["answer"].tolist()
        faq_embeddings = embedding_model.encode(faq_questions, convert_to_tensor=True)
        return f" Loaded {len(faq_questions)} questions from file."
    except Exception as e:
        return f" Error loading file: {str(e)}"


# Chatbot function
def faq_chatbot(user_input, lang, chat_history):
    print(f"User input: {user_input}, Lang: {lang}")
    try:
        # Translate to English if needed
        english_input = translate_to_english(user_input, lang)
        print(f"Translated input: {english_input}")

        # Find most similar question
        input_embedding = embedding_model.encode(english_input, convert_to_tensor=True)
        similarity_scores = util.cos_sim(input_embedding, faq_embeddings)
        best_match_index = similarity_scores.argmax().item()

        # Get and translate answer
        english_answer = faq_answers[best_match_index]
        final_answer = translate_from_english(english_answer, lang)

        # Update chat history
        chat_history.append((user_input, final_answer))
        print(f"Answer: {final_answer}")
        return "", chat_history
    except Exception as e:
        error_msg = f" Error: {str(e)}"
        chat_history.append((user_input, error_msg))
        print(f"Chatbot error: {e}")
        return "", chat_history


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("##  University FAQ Chatbot (Multilingual)")

    with gr.Row():
        file_input = gr.File(label=" Upload FAQ File (CSV/XLSX)", file_types=[".csv", ".xlsx"])
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    file_input.change(fn=load_faq_file, inputs=file_input, outputs=upload_status)

    lang_selector = gr.Dropdown(
        ["en", "es", "fr", "hi"],
        label=" Choose Language",
        value="en"
    )

    chatbot = gr.Chatbot(height=400)
    user_input = gr.Textbox(placeholder="Ask your question here...", label="Your Question")
    submit_btn = gr.Button("Send", variant="primary")

    # Connect the chatbot function
    submit_btn.click(
        fn=faq_chatbot,
        inputs=[user_input, lang_selector, chatbot],
        outputs=[user_input, chatbot]
    )

    # Allow submitting with Enter key
    user_input.submit(
        fn=faq_chatbot,
        inputs=[user_input, lang_selector, chatbot],
        outputs=[user_input, chatbot]
    )

demo.launch()