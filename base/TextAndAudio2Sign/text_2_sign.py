import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
import nltk
from nltk.stem import WordNetLemmatizer

# ✅ Linux-safe base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_RESPONSE_PATH = os.path.join(BASE_DIR, "prompt_response.csv")
WORDS_META_PATH = os.path.join(BASE_DIR, "300_words_meta.csv")
FAISS_DIR = os.path.join(BASE_DIR, "sign_retrieval_index")
CHROMA_DIR = os.path.join(BASE_DIR, "sign_retrieval_chroma")

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load embedding model
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ✅ Linux-safe CSV load
df = pd.read_csv(PROMPT_RESPONSE_PATH)
df["prompt"] = df["prompt"].astype(str)
df["response"] = df["response"].astype(str)

docs = [
    Document(page_content=str(row["prompt"]), metadata={"response": row["response"]})
    for _, row in df.iterrows()
]

vectordb = FAISS.from_documents(docs, embedding=embedding_model)
vectordb.save_local(FAISS_DIR)

vectorstore_sentence = FAISS.load_local(
    FAISS_DIR,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ Second CSV
df = pd.read_csv(WORDS_META_PATH)

docs = [
    Document(page_content=str(row["SIGN"]), metadata={"response": str(row["ID"])})
    for _, row in df.iterrows()
]

vectordb = Chroma.from_documents(docs, embedding=embedding_model, persist_directory=CHROMA_DIR)
vectordb.persist()

vectorstore_words = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

lemmatizer = WordNetLemmatizer()
auxiliary_verbs = {"am", "is", "are", "was", "were"}

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    cleaned = []
    for word, tag in nltk.pos_tag(words):
        if word in auxiliary_verbs:
            continue
        if tag.startswith("V"):
            lemma = lemmatizer.lemmatize(word, pos="v")
        else:
            lemma = lemmatizer.lemmatize(word)
        cleaned.append(lemma)
    return cleaned

def retrieve_video(user_input: str, similarity_threshold: float = 0.8):
    results = vectorstore_sentence.similarity_search_with_score(user_input, k=1)

    if results:
        doc, score = results[0]
        similarity = 1 - score
        if similarity >= similarity_threshold:
            return {"mode": "sentence", "videos": [doc.metadata.get("response", "")]}

    cleaned_words = preprocess_text(user_input)
    video_ids = []

    for word in cleaned_words:
        word_results = vectorstore_words.similarity_search_with_score(word, k=1)
        if word_results:
            doc, score = word_results[0]
            word_similarity = 1 - score
            video_ids.append(doc.metadata.get("response", "") if word_similarity >= 0.63 else "")
        else:
            video_ids.append("")

    return {"mode": "word-by-word", "videos": video_ids}
