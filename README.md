# LLM-Based RAG Research Assistant

An AI-powered research assistant that enables users to ask natural-language questions about research papers and receive context-grounded answers with source references. Built using **Python, Retrieval-Augmented Generation (RAG), FAISS, Sentence Transformers, Google Gemini, and Streamlit**.

## Project Overview

Research papers often contain complex technical concepts and extensive information that can be time-consuming to navigate. This project addresses that challenge by developing a Retrieval-Augmented Generation (RAG) application that allows users to interact with a research paper through natural-language questions.

The system extracts text from a PDF document, divides it into manageable chunks, generates semantic embeddings, and retrieves relevant passages using FAISS vector similarity search. Google Gemini then uses the retrieved context to generate informative answers.

An interactive **Streamlit web interface** provides a convenient way to submit questions, view generated answers, and inspect the retrieved source passages.

## Key Features

* **PDF Document Processing:** Extracts research paper text using PyMuPDF.
* **Intelligent Text Chunking:** Splits documents into overlapping chunks to preserve contextual information.
* **Semantic Embeddings:** Uses Sentence Transformers to convert text into numerical vector representations.
* **Vector Similarity Search:** Implements FAISS for efficient retrieval of semantically relevant passages.
* **Generative AI Integration:** Uses the Google Gemini API to generate context-grounded responses.
* **Source References:** Displays relevant passages, PDF page numbers, and similarity scores for verification.
* **Interactive Streamlit Interface:** Allows users to ask questions and explore answers through a web browser.
* **API Key Management:** Uses environment variables to avoid hardcoding API credentials in application code.

## Interactive Web Interface

The application includes a user-friendly web interface developed with **Streamlit**, making the RAG pipeline accessible without requiring users to interact directly with the underlying Python code.

### Interface Capabilities

* Enter questions about the research paper in natural language.
* Generate answers based on retrieved document passages.
* View supporting source passages and their PDF page numbers.
* Inspect similarity scores to understand which passages were retrieved.
* Explore technical concepts from the research paper through a simple question-and-answer workflow.

### How to Use the Application

1. Launch the application using Streamlit.
2. Open the local URL displayed in the terminal.
3. Enter a question about the research paper.
4. Click **Get Answer**.
5. Review the generated response.
6. Expand **View retrieved sources** to inspect the supporting passages and page references.

### Application Architecture

The application follows a retrieval-augmented generation pipeline:

**User Question → Streamlit Interface → Sentence Transformer Embeddings → FAISS Vector Search → Retrieved Context → Google Gemini → Answer and Source Display**

The retrieval stage identifies relevant passages from the research paper before the language model generates an answer. This approach helps ground responses in the supplied document, although generated answers and citations should still be verified.

## Technology Stack

| Technology               | Purpose                               |
| ------------------------ | ------------------------------------- |
| Python                   | Core application development          |
| Jupyter Notebook         | Experimentation and development       |
| PyMuPDF                  | PDF text extraction                   |
| LangChain Text Splitters | Text chunking and preprocessing       |
| Sentence Transformers    | Semantic embedding generation         |
| FAISS                    | Vector indexing and similarity search |
| Google Gemini API        | Context-grounded answer generation    |
| Streamlit                | Interactive web application           |
| NumPy                    | Numerical operations on embeddings    |

## Project Workflow

### 1. Document Loading

Load the research paper PDF and extract text from its pages using PyMuPDF.

### 2. Text Chunking

Split the extracted text into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`. This helps the retriever locate relevant sections without processing the entire paper for every question.

### 3. Embedding Generation

Generate semantic embeddings for the text chunks using the `sentence-transformers/all-MiniLM-L6-v2` model.

### 4. FAISS Indexing

Normalize the embeddings and store them in a FAISS index to enable efficient similarity-based retrieval.

### 5. Question Embedding

Convert the user's question into a vector using the same Sentence Transformer model.

### 6. Context Retrieval

Search the FAISS index and retrieve the most relevant document chunks based on vector similarity.

### 7. Answer Generation

Pass the retrieved passages and the user's question to Google Gemini to generate an answer grounded in the available document context.

### 8. Streamlit Presentation

Display the generated answer in the web interface, along with expandable source passages, page numbers, and similarity scores.

## Project Structure

```text
Research-Assistant/
├── LLM RAG Research Assistant.ipynb
├── app.py
├── Research Paper.pdf
├── research_paper.index
├── requirements.txt
├── .gitignore
├── README.md
└── screenshots/
    └── rag-assistant.png
```

## Installation and Setup

### Prerequisites

* Python 3.10 or a compatible Python version
* A Google Gemini API key
* Git
* Access to the research paper PDF

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the placeholders with your GitHub username and repository name.

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the packages used by your application, such as:

```text
streamlit
numpy
faiss-cpu
PyMuPDF
sentence-transformers
langchain-text-splitters
google-genai
```

### 3. Configure the Gemini API Key

Set your Gemini API key as an environment variable.

**Windows — Command Prompt or Anaconda Prompt:**

```bat
set GEMINI_API_KEY=YOUR_ACTUAL_API_KEY
```

Replace the placeholder with your actual key locally. Never upload your real key or a command containing it to GitHub.

The application reads the key using:

```python
import os

api_key = os.getenv("GEMINI_API_KEY")
```

### 4. Run the Streamlit Application

Navigate to the project directory and execute:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal to interact with the application.

## Example Questions

The application can be used to explore concepts in the selected research paper, including:

* What is the main objective of the paper?
* Why does the Transformer use attention instead of recurrent neural networks?
* What is self-attention?
* How does multi-head attention work?
* What is the purpose of positional encoding?
* What are the main contributions of the research?
* Which sections discuss the Transformer architecture?

The quality of the response depends on the relevance of the retrieved passages and the language model's generation.

## Evaluation and Limitations

The application can be evaluated by manually testing questions and checking:

* Whether the retrieved passages are relevant to each question.
* Whether the generated answers are supported by the research paper.
* Whether the displayed page references point to the correct source material.
* How the system handles questions whose answers are absent from the document.

This project is an initial implementation focused on a single research paper. Retrieval quality, document formatting, and language-model behavior may affect results. Generated answers are not guaranteed to be completely accurate, so source verification is recommended.

## Future Enhancements

* Support for multiple research papers and document collections.
* Hybrid search combining semantic and keyword retrieval.
* Reranking to improve the relevance of retrieved passages.
* Conversation history and follow-up question support.
* Automated evaluation of retrieval quality, answer correctness, and faithfulness.
* Improved handling of tables, equations, and complex PDF layouts.
* Deployment to a cloud platform for broader accessibility.

## Learning Outcomes

This project provided hands-on experience with:

* Building a Retrieval-Augmented Generation pipeline.
* Extracting and preprocessing text from PDF documents.
* Creating semantic embeddings using Sentence Transformers.
* Implementing vector similarity search with FAISS.
* Integrating a large language model through the Google Gemini API.
* Developing an interactive AI application using Streamlit.
* Managing API credentials through environment variables.
* Connecting document retrieval with context-grounded answer generation.

