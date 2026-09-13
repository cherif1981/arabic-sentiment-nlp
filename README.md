🇩🇿 Arabic Sentiment NLP

«Arabic & Algerian Dialect Sentiment Analysis using Machine Learning and Transformer Models»

A modular Natural Language Processing (NLP) project for sentiment analysis of Arabic text, with a particular focus on Algerian Arabic (Darija) and social-media-style text.

The project combines traditional Machine Learning approaches with modern Arabic Transformer models and provides both a training/evaluation pipeline and a production-ready inference API.

---

🚀 Project Overview

Sentiment analysis for Arabic is challenging because of:

- Arabic linguistic diversity
- MSA vs. dialectal Arabic
- Algerian Darija
- spelling variations
- missing diacritics
- Arabic/French code-switching
- emojis and informal expressions
- social-media abbreviations
- negation
- sarcasm and implicit sentiment

This project aims to build a reproducible and extensible pipeline capable of classifying Arabic text into:

- 🟢 Positive
- 🔴 Negative
- ⚪ Neutral

The architecture is designed to support experimentation with both classical Machine Learning models and Transformer-based models.

---

🏗️ Architecture

                         Arabic Text
                              │
                              ▼
                    ┌──────────────────┐
                    │ Text Preprocessing│
                    └─────────┬────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             Classical ML          Transformers
                    │                   │
              TF-IDF + LR        AraBERT / MARBERT
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
                     Sentiment Classifier
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
             Positive      Neutral       Negative
                              │
                              ▼
                   ┌────────────────────┐
                   │ Inference / API     │
                   └─────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
              FastAPI               Streamlit

---

📁 Project Structure

arabic-sentiment-nlp/
│
├── api/                    # FastAPI application
│
├── app/                    # Streamlit application
│
├── data/                   # Dataset and processed data
│
├── models/                 # Trained models and artifacts
│
├── notebooks/              # Experiments and exploratory analysis
│
├── reports/                # Evaluation reports and results
│
├── src/                    # Core source code
│   ├── data/               # Data loading and processing
│   ├── preprocessing/      # Arabic text preprocessing
│   ├── features/           # Feature extraction
│   ├── models/             # Model definitions
│   ├── training/           # Training pipelines
│   └── evaluation/         # Evaluation utilities
│
├── tests/                  # Automated tests
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .github/
│   └── workflows/          # CI/CD workflows
│
└── README.md

---

🧠 Models

The project is designed around a progressive modeling strategy.

1. TF-IDF + Logistic Regression

A classical baseline:

Arabic Text
     │
     ▼
Preprocessing
     │
     ▼
TF-IDF
     │
     ▼
Logistic Regression
     │
     ▼
Sentiment

This baseline provides an important reference point for evaluating Transformer models.

---

2. AraBERT

"AraBERT" (https://github.com/aub-mind/arabert) is used as an Arabic Transformer baseline for contextual text representation and sentiment classification.

Input Text
    │
    ▼
AraBERT Tokenizer
    │
    ▼
AraBERT Encoder
    │
    ▼
Classification Head
    │
    ▼
Positive / Neutral / Negative

---

3. MARBERT

MARBERT is particularly relevant for this project because it was designed with Arabic dialectal and social-media text in mind.

The model is evaluated against the classical baseline and other Transformer approaches using the same evaluation protocol.

---

📊 Evaluation

Models should be evaluated using the same test set and evaluation methodology.

Recommended metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Weighted F1
- Confusion Matrix

Results

«Results below should be populated after reproducible experiments.»

Model| Accuracy| Macro F1| Weighted F1
TF-IDF + Logistic Regression| TBD| TBD| TBD
AraBERT| TBD| TBD| TBD
MARBERT| TBD| TBD| TBD

Why Macro F1?

For multi-class sentiment classification, Macro F1 is particularly useful because it gives equal importance to each class.

This is important when the dataset is not perfectly balanced.

---

🔬 Error Analysis

Model evaluation should not rely only on a single metric.

The project also investigates difficult Arabic NLP cases such as:

Dialectal Arabic

"الخدمة هايلة بزاف"

Arabic/French code-switching

"الخدمة كانت super"

Negation

"ماشي مليح"

Emojis

"روعة 😍🔥"

Informal spelling

"روووعة"

Ambiguous or sarcastic expressions

"واش من خدمة هذي 😂"

These examples demonstrate why sentiment analysis in Algerian social-media text is more challenging than simple keyword-based classification.

---

🧹 Arabic Text Preprocessing

The preprocessing pipeline is designed to handle characteristics of Arabic text.

Potential preprocessing operations include:

- Unicode normalization
- Arabic character normalization
- whitespace normalization
- removal of unnecessary characters
- handling repeated characters
- URL handling
- mention handling
- hashtag processing
- emoji preservation
- Arabic/French mixed text
- optional diacritic handling

Important principle

Preprocessing should not destroy sentiment information.

For example, emojis can carry strong sentiment:

"ممتاز 😍🔥"

Therefore, preprocessing decisions are evaluated according to their impact on model performance.

---

⚙️ Installation

Clone the repository

git clone https://github.com/cherif1981/arabic-sentiment-nlp.git
cd arabic-sentiment-nlp

Create a virtual environment

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

Windows

python -m venv .venv
.venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

For development:

pip install -r requirements-dev.txt

---

🧪 Running Tests

Run the test suite with:

pytest

For verbose output:

pytest -v

---

🏋️ Training

The training pipeline is organized to make experiments reproducible.

Typical workflow:

Dataset
   │
   ▼
Data Validation
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Preprocessing
   │
   ▼
Feature Extraction / Tokenization
   │
   ▼
Model Training
   │
   ▼
Evaluation
   │
   ▼
Model Artifact

Training commands depend on the selected model and configuration.

---

🔍 Inference

After training, a model can be used to predict the sentiment of new Arabic text.

Example:

Input:
"الخدمة كانت رائعة بزاف"

Output:
Positive

A production inference interface is provided through the API.

---

🌐 API

The project includes a FastAPI application for model inference.

Example request:

POST /predict

Request:

{
  "text": "الخدمة كانت رائعة"
}

Example response:

{
  "label": "positive",
  "confidence": 0.97
}

«The exact API endpoints and response schema should match the implementation in "api/".»

---

🖥️ Streamlit Application

A Streamlit interface is provided for interactive experimentation.

The application can be used to:

1. Enter Arabic text
2. Submit the text for inference
3. Display the predicted sentiment
4. Display the model confidence
5. Experiment with different Arabic expressions

Run the application with:

streamlit run app/app.py

«Update the path if the Streamlit entry point differs in the current implementation.»

---

🐳 Docker

The project provides Docker support to simplify deployment and reproducibility.

Build the image:

docker build -t arabic-sentiment-nlp .

Run the container:

docker run -p 8000:8000 arabic-sentiment-nlp

If using Docker Compose:

docker compose up --build

---

🔁 Reproducibility

Machine Learning experiments should be reproducible whenever possible.

The project therefore aims to explicitly control:

- random seeds
- dataset splits
- model configuration
- tokenizer configuration
- hyperparameters
- dependency versions
- training configuration
- evaluation methodology

When reporting results, the following information should be recorded:

Model
Dataset
Train/Validation/Test split
Tokenizer
Max sequence length
Batch size
Learning rate
Number of epochs
Random seed
Hardware
Training time
Accuracy
Macro F1
Weighted F1

---

🧪 Experimental Methodology

To ensure a fair comparison between models:

                 Same Dataset
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       TF-IDF      AraBERT      MARBERT
          │           │           │
          └───────────┼───────────┘
                      ▼
                Same Test Set
                      │
                      ▼
             Same Evaluation Metrics

Models should not be compared using different test sets.

---

🔐 Security Considerations

The API should not expose:

- model internals unnecessarily
- filesystem paths
- environment variables
- secrets
- debugging information
- stack traces in production

Production deployment should also consider:

- request validation
- maximum input length
- rate limiting
- logging
- authentication if required
- resource limits

---

📈 Continuous Integration

The project includes automated development workflows.

Recommended CI checks include:

Push / Pull Request
       │
       ▼
Install Dependencies
       │
       ▼
Lint
       │
       ▼
Type Checking
       │
       ▼
Unit Tests
       │
       ▼
Integration Tests
       │
       ▼
Build

---

🎯 Project Goals

The main objectives are:

- Build a robust Arabic sentiment analysis pipeline
- Support Algerian Arabic/Darija
- Compare classical ML with Transformer models
- Establish reproducible experiments
- Provide a production-oriented inference API
- Provide an interactive demonstration
- Perform detailed error analysis
- Create a foundation for further Arabic NLP research

---

🛣️ Roadmap

Phase 1 — Foundation

- [x] Project structure
- [x] Data pipeline
- [x] Preprocessing
- [x] Classical baseline
- [x] Transformer integration
- [x] API
- [x] Streamlit application
- [x] Docker
- [x] Automated tests

Phase 2 — Evaluation

- [ ] Complete benchmark results
- [ ] Confusion matrices
- [ ] Macro/Weighted F1 comparison
- [ ] Error analysis
- [ ] Class imbalance analysis
- [ ] Reproducible experiment configurations

Phase 3 — Arabic/Darija NLP

- [ ] Improved Algerian dialect handling
- [ ] Code-switching analysis
- [ ] Emoji-aware sentiment analysis
- [ ] Negation analysis
- [ ] Sarcasm analysis
- [ ] Dialect-aware evaluation

Phase 4 — Production

- [ ] Model versioning
- [ ] Model registry
- [ ] API authentication
- [ ] Rate limiting
- [ ] Monitoring
- [ ] Performance benchmarking
- [ ] Automated deployment

---

⚠️ Limitations

Arabic sentiment analysis remains challenging because sentiment is often context-dependent.

The model may have difficulty with:

- sarcasm
- irony
- ambiguous expressions
- implicit sentiment
- mixed Arabic/French text
- spelling variations
- rare Algerian expressions
- cultural context
- very short messages

Model performance should therefore be interpreted together with qualitative error analysis.

---

🔮 Future Research

Potential extensions include:

- emotion classification
- sarcasm detection
- offensive-language detection
- hate-speech detection
- dialect identification
- aspect-based sentiment analysis
- multilingual sentiment analysis
- retrieval-augmented NLP
- instruction-tuned Arabic LLMs
- ensemble models
- explainable sentiment classification

A longer-term objective is to develop a broader Arabic/Algerian NLP platform rather than limiting the project to a single classification task.

---

🤝 Contributing

Contributions are welcome.

Suggested workflow:

git checkout -b feature/my-feature

Make your changes, add tests, and submit a pull request.

Please ensure that:

- existing tests pass
- new functionality includes tests
- code follows the project conventions
- documentation is updated when necessary

---

📜 License

See the "LICENSE" file for the license applicable to this project.

---

👨‍💻 Author

Cherif Meddour

GitHub:

"https://github.com/cherif1981" (https://github.com/cherif1981)

Project:

"https://github.com/cherif1981/arabic-sentiment-nlp" (https://github.com/cherif1981/arabic-sentiment-nlp)

---

⭐ Acknowledgments

This project builds upon the broader open-source Arabic NLP ecosystem and publicly available Arabic Transformer research and models.

Special attention is given to models designed for Arabic and dialectal/social-media text, particularly AraBERT and MARBERT.

---

📌 Status

🚧 Active Development

The project is currently evolving from an experimental NLP project toward a reproducible research and production-oriented Arabic NLP system.