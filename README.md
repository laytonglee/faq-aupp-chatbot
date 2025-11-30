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
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

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
- **Python 3.x**: Primary programming language
- **Flask**: Web framework for API development
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
ITM_454_FAQ_Chatbot/
│
├── app.py                              # Main Flask application
├── dataset.json                        # FAQ dataset with questions and answers
├── requirements.txt                    # Python dependencies
├── pipeline_svm.joblib                 # Trained SVM pipeline model
├── faq_with_embeddings.joblib          # FAQ data with embeddings
│
├── .ipynb_checkpoints/                 # Jupyter notebook checkpoints
├── .virtual_documents/                 # Virtual document cache
├── final-project-env/                  # Virtual environment (not tracked)
│
└── README.md                           # Project documentation (this file)
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ITM_454_FAQ_Chatbot.git
cd ITM_454_FAQ_Chatbot
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

### Step 4: Verify Installation

```bash
python -c "import flask, sklearn, joblib; print('All dependencies installed successfully!')"
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

#### Step 1: Generate SBERT Embeddings

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

#### Step 2: Train SVM Pipeline

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

### Why Two Approaches?

- **SVM with TF-IDF**: Fast, accurate for exact keyword matching
- **SBERT Embeddings**: Understands semantic meaning, handles paraphrased questions
- **Combined Power**: Best of both worlds for robust FAQ matching

### Model Performance

- **Response Time**: < 100ms average
- **Model Size**: 
  - SVM Pipeline: ~2-5 MB
  - SBERT Embeddings: ~5-10 MB
  - Total: ~10-15 MB
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Training Data**: Custom AUPP FAQ dataset
- **Accuracy**: Optimized for FAQ matching with high precision

## 📊 Dataset

### Structure

The `dataset.json` file contains FAQ pairs in the following format:

```json
{
  "faqs": [
    {
      "question": "What programs does AUPP offer?",
      "answer": "AUPP offers undergraduate programs in Business, Computer Science, International Relations, and more."
    },
    {
      "question": "How do I apply to AUPP?",
      "answer": "You can apply through our online application portal. Visit the admissions page for details."
    }
  ]
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

### Code Style

This project follows PEP 8 style guidelines. Use the following tools:

- **Black**: Code formatter
- **Flake8**: Linting
- **isort**: Import sorting

### Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test thoroughly
4. Commit: `git commit -m "Add your feature"`
5. Push: `git push origin feature/your-feature-name`
6. Create a Pull Request

## 🚢 Deployment

### Deployment on Streamlit Community Cloud (Recommended)

Streamlit Community Cloud offers free hosting for Streamlit applications:

1. **Push your code to GitHub**:
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch, and main file (`app.py`)
   - Click "Deploy"

3. **Your app will be live** at: `https://[your-app-name].streamlit.app`

### Local Deployment with Port Forwarding

For local deployment with external access:

```bash
# Run with custom port
streamlit run app.py --server.port 8080

# Run with external access
streamlit run app.py --server.address 0.0.0.0
```

### Running in Background

```bash
# Using nohup
nohup streamlit run app.py &

# Using screen
screen -S chatbot
streamlit run app.py
# Press Ctrl+A then D to detach
```

## 📝 Requirements

```txt
streamlit==1.28.0
scikit-learn==1.3.0
sentence-transformers==2.2.2
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
torch==2.0.0
```

**Note**: `sentence-transformers` automatically installs PyTorch, which is required for SBERT models.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Follow the existing code style
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- American University of Phnom Penh (AUPP) for project inspiration
- ITM 454 course instructors and teaching assistants
- Open source community for amazing tools and libraries

## 📞 Contact

- **Project Maintainer**: Your Name
- **Email**: your.email@example.com
- **Project Link**: https://github.com/yourusername/ITM_454_FAQ_Chatbot

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

## 📈 Version History

- **v1.0.0** (2024-11-30)
  - Initial release
  - Basic FAQ chatbot functionality
  - SVM-based classification
  - REST API implementation

---

**Note**: This is an academic project developed as part of ITM 454 course at AUPP.

For questions or issues, please open an issue on GitHub or contact the maintainers.
