import streamlit as st
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Set page config FIRST
st.set_page_config(
    page_title="Text Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NLTK Setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #2E86C1;
        --secondary-color: #F4F6F6;
    }
    
    .main {background-color: var(--secondary-color);}
    h1 {color: var(--primary-color); border-bottom: 2px solid var(--primary-color);}
    .stButton>button {background-color: var(--primary-color); color: white; border-radius: 5px;}
    .stProgress > div > div > div {background-color: var(--primary-color);}
    .custom-card {padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: white;}
</style>
""", unsafe_allow_html=True)

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    corpus = []
    for sentence in sentences:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', sentence)
        cleaned = cleaned.lower().split()
        cleaned = [lemmatizer.lemmatize(word) for word in cleaned if word not in stop_words]
        corpus.append(' '.join(cleaned))
    return corpus

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>📚 Text Analytics Suite</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-bottom: 30px; color: #666;'>by Mohammed Shoeb</div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Logo", use_container_width=True)
        st.markdown("### Configuration")
        analysis_type = st.radio(
            "Analysis Type:",
            ["Bag of Words", "TF-IDF"],
            index=0
        )
        st.markdown("---")
        st.markdown("**Parameters**")
        max_features = st.slider("Max Features", 50, 500, 100)
        ngram_range = st.selectbox("N-gram Range", [(1,1), (1,2), (1,3)])

    # Main Interface
    with st.container():
        input_text = st.text_area("Input Text:", height=200, placeholder="Enter your text here...")
        
        if st.button("Analyze Text", type="primary"):
            if not input_text.strip():
                st.error("Please input text for analysis")
                return
                
            with st.spinner("Processing..."):
                try:
                    corpus = preprocess_text(input_text)
                    
                    # Feature Extraction
                    if analysis_type == "Bag of Words":
                        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
                    else:
                        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
                        
                    X = vectorizer.fit_transform(corpus)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Display Results
                    with st.expander("🔍 Processed Text", expanded=True):
                        for i, sent in enumerate(corpus):
                            st.markdown(f"**Sentence {i+1}:** {sent}")
                    
                    with st.expander("📊 Feature Matrix", expanded=True):
                        df = pd.DataFrame(X.toarray(), columns=feature_names)
                        st.dataframe(df.style.background_gradient(cmap="Blues"), height=400)
                        
                        if analysis_type == "TF-IDF":
                            st.markdown("---")
                            st.subheader("IDF Values")
                            idf_df = pd.DataFrame({
                                'Feature': feature_names,
                                'IDF Score': vectorizer.idf_
                            }).sort_values('IDF Score', ascending=False)
                            st.dataframe(idf_df, height=300)
                    
                    # Download Section
                    with st.expander("💾 Download Results"):
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Feature Matrix",
                            data=csv,
                            file_name="text_features.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    main()