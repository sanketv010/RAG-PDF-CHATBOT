# RAG-PDF-CHATBOT

The RAG-PDF Chatbot leverages the power of modern language models and vector embeddings to provide accurate and contextually relevant responses to user queries. The system retrieves relevant information from PDFs and generates responses using a language model.

## Features
- **Interactive User Interface**: Built with Streamlit for ease of use.
- **Advanced Embeddings**: Uses OllamaEmbeddings for generating vector embeddings.
- **Efficient Retrieval**: Utilizes Pinecone for fast and accurate vector search.
- **Powerful Language Model**: Employs ChatGroq's Inference API for generating responses.
- **Context Management**: Maintains conversation context using ConversationBufferWindowMemory.

## Demo Video

Check out the demo of the RAG-PDF Chatbot in action:

https://github.com/sanketv010/RAG-PDF-CHATBOT/assets/113123926/82af9529-a1b1-4a32-9c1a-33fe2f8c0a0d


  
## Installation
To install and run the RAG-PDF Chatbot, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/sanketv010/RAG-PDF-CHATBOT.git
    cd RAG-PDF-CHATBOT
    ```

2. **Create and Activate a Virtual Environment**:
    ```bash
    conda create -p environment_name python=3.10 - y
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**:
    Create a `.env` file in the root directory of the project and add the following environment variables:
    ```
    PINECONE_API_KEY=your_pinecone_api_key
    GROQ_API_KEY=your_groq_api_key
    ```

5. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## Usage
1. Open the web interface in your browser (usually at `http://localhost:8000`).
2. Enter your query in the text input field.
3. The chatbot will retrieve relevant information from the PDFs and generate a response based on the context.

---

Thank you for checking out the RAG-PDF Chatbot! If you have any questions or feedback, feel free to open an issue or reach out.  
