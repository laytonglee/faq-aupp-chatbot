# ITM 454 FAQ Chatbot for AUPP Website

A comprehensive FAQ chatbot system built using machine learning and natural language processing to provide automated responses to frequently asked questions for the AUPP website.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [API Endpoints](#api-endpoints)

## 🎯 Overview

This project implements an intelligent FAQ chatbot system designed to automatically answer frequently asked questions for the American University of Phnom Penh (AUPP) website. The system uses machine learning techniques including Support Vector Machines (SVM) and semantic text embeddings to understand user queries and provide accurate responses.

## ✨ Features

- **Intelligent Query Understanding**: Uses both TF-IDF vectorization and SBERT semantic embeddings to understand user intent
- **Dual ML Architecture**: Combines SVM classifier with semantic similarity search for robust matching
- **Interactive Web Interface**: Built with Streamlit for easy-to-use chatbot interface
- **Pre-trained Models**: Includes pre-trained SVM pipeline and SBERT embeddings for immediate deployment
- **Semantic Understanding**: SBERT model understands meaning, not just keywords
- **Real-time Responses**: Instant answers to user queries
- **User-Friendly UI**: Clean and intuitive interface for non-technical users
- **JSON-based Dataset**: Easy-to-update FAQ database stored in JSON format

## 🛠 Technologies Used

### Core Technologies
- **Python 3.11.6**: Primary programming language
- **Streamlit**: Web framework for API development
- **Scikit-learn**: Machine learning library for SVM implementation
- **Joblib**: Model serialization and deserialization
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis

### Machine Learning Components
- **TF-IDF Vectorizer**: Text feature extraction
- **Support Vector Machine (SVM)**: Classification algorithm
- **Linear SVC**: Support Vector Classification with linear kernel
- **SBERT (Sentence-BERT)**: Pre-trained transformer model for semantic embeddings
- **all-MiniLM-L6-v2**: Lightweight and efficient sentence embedding model
- **Pipeline**: Streamlined ML workflow

## 📁 Project Structure

```
faq-aupp-chatbot/
├── app.py                              # Main Streamlit application
├── dataset.json                        # FAQ dataset with questions and answers
├── requirements.txt                    # Python dependencies
├── pipeline_svm.joblib                 # Trained SVM pipeline model
├── faq_with_embeddings.joblib          # FAQ data with embeddings
└── README.md                           # Project documentation (this file)
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/laytonglee/faq-aupp-chatbot.git

cd faq-aupp-chatbot
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv final-project-env

# Activate virtual environment
# On Windows:
final-project-env\Scripts\activate

# On macOS/Linux:
source final-project-env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Application Locally

```bash
# Make sure virtual environment is activated
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`

### Using the Chatbot Interface

1. **Ask a Question**: Type your question in the text input box
2. **Submit**: Click the "Ask" button or press Enter
3. **View Response**: The chatbot will display the answer instantly
4. **Ask Another Question**: Clear the input and ask a new question

### Example Questions to Try

- "What programs does AUPP offer?"
- "How do I apply to AUPP?"
- "What are the admission requirements?"
- "Tell me about tuition fees"
- "What is the campus location?"

## 🤖 Model Details

### Machine Learning Pipeline

The chatbot uses a sophisticated dual-approach ML system:

1. **Text Preprocessing**: Cleaning and normalization of input text
2. **Feature Extraction**: 
   - **TF-IDF Vectorization**: Converts text to numerical features for SVM
   - **SBERT Embeddings**: Creates semantic vector representations using all-MiniLM-L6-v2
3. **Classification**: Linear SVM classifier to match questions to answers
4. **Semantic Search**: Uses cosine similarity on embeddings for intelligent matching
5. **Confidence Scoring**: Probability estimates for response confidence

### Training Process

#### Step 1: Train SVM Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', LinearSVC(random_state=42))
])

# Train model
pipeline.fit(questions, answers)

# Save model
joblib.dump(pipeline, 'pipeline_svm.joblib')
print("SVM Pipeline Saved Successfully!")
```

#### Step 2: Generate SBERT Embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import joblib

# Load pre-trained SBERT model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for each question
df_expanded["embedding"] = df_expanded["question"].apply(lambda x: sbert.encode(x))

# Save embeddings for future use
joblib.dump(df_expanded, "faq_with_embeddings.joblib")
print("SBERT FAQ Embeddings Saved Successfully!")
```

### Why Two Approaches?

- **SVM with TF-IDF**: Fast, accurate for exact keyword matching
- **SBERT Embeddings**: Understands semantic meaning, handles paraphrased questions
- **Combined Power**: Best of both worlds for robust FAQ matching

### Model Performance

- **Response Time**: < 100ms average
- **Model Size**: 
  - SVM Pipeline: 300 KB
  - SBERT Embeddings: 4000 KB
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Training Data**: Custom AUPP FAQ dataset
- **Accuracy**: Optimized for FAQ matching with high precision

## 📊 Dataset

### Structure

The `dataset.json` file contains FAQ pairs in the following format:

```json
{
  {
    "category": "Academic Policies",
    "question": "When can I take a leave of absence from AUPP?",
    "answer": "You can request a Leave of Absence (LOA) for life situations, medical conditions, or psychological conditions that significantly impair your ability to function successfully at the University.\n\nThe impairment must be recognized by a physician or the University.\n\nYou need to complete a LOA Form and submit it to the Office of the Registrar with appropriate signatures.",
    "alt_questions": [
      "How do I apply for a leave of absence?",
      "Can I take time off from AUPP?",
      "What are the rules for requesting a LOA?",
      "When am I allowed to take a break from studies?"
    ]
  },
  {
    "category": "Academic Policies",
    "question": "Who approves a leave of absence?",
    "answer": "The authority to grant a Leave of Absence rests with the President.\n\nThe Office of the Registrar investigates your situation and makes a recommendation to the President.\n\nThe President also has the authority to grant permission to return from a LOA.",
    "alt_questions": [
      "Who decides if I can take a leave?",
      "Which office approves my LOA request?",
      "Does the Registrar approve leave of absence?",
      "Who gives permission to return after a LOA?"
    ]
  }
}
```

### Adding New FAQs

1. Open `dataset.json`
2. Add new FAQ entry following the structure above
3. Regenerate embeddings by running:

```python
from sentence_transformers import SentenceTransformer
import pandas as pd
import joblib

# Load SBERT model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Load your updated dataset
df_expanded = pd.read_json("dataset.json")

# Generate new embeddings
df_expanded["embedding"] = df_expanded["question"].apply(lambda x: sbert.encode(x))

# Save updated embeddings
joblib.dump(df_expanded, "faq_with_embeddings.joblib")
print("Embeddings updated successfully!")
```

4. Restart the Streamlit application - the new FAQs will be automatically loaded

## 🌐 API Endpoints

This application uses Streamlit's interactive interface and does not expose REST API endpoints. All interactions are handled through the web interface.

### Application Features

- **Interactive Chat Interface**: Type questions directly in the web UI
- **Real-time Processing**: Instant responses without page reloads
- **Session State Management**: Maintains conversation context
- **Clean UI**: Streamlit's native components for better user experience
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

# For development with auto-reload (default in Streamlit)
# Just save your changes and Streamlit will auto-reload
```

## 🙏 Acknowledgments

- American University of Phnom Penh (AUPP) for project inspiration
- ITM 454 course instructors and teaching assistants
- Open source community for amazing tools and libraries

## 🔮 Future Enhancements

- [ ] Add conversation history display in sidebar
- [ ] Implement multi-language support (Khmer, English)
- [ ] Add confidence score display for answers
- [ ] Create admin panel for FAQ management
- [ ] Add export conversation feature
- [ ] Implement user feedback mechanism
- [ ] Add search functionality for FAQ database
- [ ] Create analytics dashboard for common queries
- [ ] Add voice input support
- [ ] Implement suggested questions feature

**Note**: This is an academic project developed as part of ITM 454 course at AUPP.
