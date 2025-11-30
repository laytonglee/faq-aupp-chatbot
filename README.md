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

- **Intelligent Query Understanding**: Uses TF-IDF vectorization and semantic embeddings to understand user intent
- **Machine Learning Powered**: Implements SVM classifier for accurate question matching
- **RESTful API**: Built with Flask for easy integration with web applications
- **Pre-trained Models**: Includes pre-trained pipeline for immediate deployment
- **Scalable Architecture**: Modular design allows for easy updates and maintenance
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
python app.py
```

The application will start on `http://localhost:5000`

### Making API Requests

#### Using cURL

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What programs does AUPP offer?"}'
```

#### Using Python

```python
import requests

url = "http://localhost:5000/ask"
data = {"question": "What are the admission requirements?"}

response = requests.post(url, json=data)
print(response.json())
```

#### Using JavaScript (Fetch API)

```javascript
fetch('http://localhost:5000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'How do I apply to AUPP?'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🤖 Model Details

### Machine Learning Pipeline

The chatbot uses a sophisticated ML pipeline consisting of:

1. **Text Preprocessing**: Cleaning and normalization of input text
2. **Feature Extraction**: TF-IDF vectorization to convert text to numerical features
3. **Classification**: Linear SVM classifier to match questions to answers
4. **Confidence Scoring**: Probability estimates for response confidence

### Training Process

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
import joblib
joblib.dump(pipeline, 'pipeline_svm.joblib')
```

### Model Performance

- **Accuracy**: Optimized for FAQ matching
- **Response Time**: < 100ms average
- **Model Size**: Approximately 2-5 MB
- **Training Data**: Custom AUPP FAQ dataset

## 📊 Dataset

### Structure

The `dataset.json` file contains FAQ pairs in the following format:

```json
{
  "faqs": [
    {
      "id": 1,
      "question": "What programs does AUPP offer?",
      "answer": "AUPP offers undergraduate programs in Business, Computer Science, International Relations, and more.",
      "category": "academics",
      "keywords": ["programs", "majors", "degrees"]
    }
  ]
}
```

### Adding New FAQs

1. Open `dataset.json`
2. Add new FAQ entry following the structure above
3. Retrain the model:

```bash
python train_model.py  # Create this script if needed
```

## 🌐 API Endpoints

### POST /ask

Ask a question to the chatbot.

**Request:**
```json
{
  "question": "What are the tuition fees?"
}
```

**Response:**
```json
{
  "question": "What are the tuition fees?",
  "answer": "Tuition fees vary by program. Please contact admissions for detailed information.",
  "confidence": 0.95,
  "timestamp": "2024-11-30T10:30:00Z"
}
```

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Run linting
flake8 app.py
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

### Deployment Options

#### Option 1: Heroku

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create new app
heroku create your-app-name

# Deploy
git push heroku main
```

#### Option 2: AWS EC2

1. Launch EC2 instance
2. SSH into instance
3. Clone repository
4. Install dependencies
5. Run with Gunicorn:

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

#### Option 3: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t faq-chatbot .
docker run -p 5000:5000 faq-chatbot
```

## 📝 Requirements

```txt
Flask==2.3.0
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
gunicorn==21.2.0
```

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

- [ ] Add multi-language support
- [ ] Implement conversation history tracking
- [ ] Add admin dashboard for FAQ management
- [ ] Integrate with AUPP website directly
- [ ] Add natural language generation for dynamic responses
- [ ] Implement feedback mechanism for continuous improvement
- [ ] Add authentication and rate limiting
- [ ] Create mobile application interface

## 📈 Version History

- **v1.0.0** (2024-11-30)
  - Initial release
  - Basic FAQ chatbot functionality
  - SVM-based classification
  - REST API implementation

---

**Note**: This is an academic project developed as part of ITM 454 course at AUPP.

For questions or issues, please open an issue on GitHub or contact the maintainers.

