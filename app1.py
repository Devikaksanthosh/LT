import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Set the title of the app
st.title("🌐 Language Translator")

# Description
st.markdown("""
This application translates text from one language to another using a pre-trained model from Hugging Face.
""")

# Sidebar for additional options
st.sidebar.header("Settings")

# Option to select source and target languages
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko"
}

source_language = st.sidebar.selectbox("Select Source Language", list(languages.keys()))
target_language = st.sidebar.selectbox("Select Target Language", list(languages.keys()))

# Ensure source and target languages are different
if languages[source_language] == languages[target_language]:
    st.warning("Source and target languages must be different.")
    st.stop()

# Determine the appropriate model based on selected languages
def get_model_name(src_lang, tgt_lang):
    return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

# Load translation model and tokenizer
@st.cache_resource
def load_translator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer, framework="pt")
    return translator

# Ensure the model exists for the selected language pair
model_name = get_model_name(languages[source_language], languages[target_language])
translator = load_translator(model_name)

# Text input
st.subheader("Enter the text you want to translate:")
text_input = st.text_area("", height=200)

if st.button("Translate"):
    if text_input.strip() == "":
        st.error("Please enter some text.")
    else:
        with st.spinner("Translating..."):
            try:
                # Translate text
                translation = translator(text_input)
                translated_text = translation[0]['translation_text']

                # Display the translated text
                st.subheader("Translated Text:")
                st.write(translated_text)

                # Provide download link
                st.download_button(
                    label="Download Translation",
                    data=translated_text,
                    file_name="translation.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error during translation: {e}")
else:
    st.info("Enter text and select languages to start translating.")
