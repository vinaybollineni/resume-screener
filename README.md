# AI-Powered Resume Screener

An intelligent resume screening system that uses natural language processing to match job descriptions with the most suitable candidates from a pool of resumes. Built with LangChain, Ollama, and FAISS for efficient similarity search.

## Features

- **Multi-format Support**: Processes both PDF and DOC/DOCX resume formats
- **Semantic Search**: Uses embeddings to understand the context and meaning of resumes and job descriptions
- **Local Processing**: All processing happens locally, ensuring data privacy
- **Customizable**: Easily adjust the number of candidates to return
- **Efficient**: Uses FAISS for fast similarity search even with large numbers of resumes

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- Required Python packages (install via `pip install -r requirements.txt`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/resume-screener.git
   cd resume-screener
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running and has the required models:
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

## Usage

1. **Add Resumes**: Place your resume files (PDF or DOC/DOCX) in the `resumes/` directory

2. **Create Vector Store**: Process the resumes to create a searchable vector store:
   ```bash
   python ingest.py
   ```

3. **Run the Screener**: Start the resume screening agent:
   ```bash
   python agent.py
   ```

4. **Enter Job Description**: When prompted, paste the job description you want to match against the resumes.

## Project Structure

```
resume-screener/
├── agent.py           # Main agent for processing job descriptions
├── ingest.py          # Script to process resumes and create vector store
├── requirements.txt   # Python dependencies
├── local_store/       # Directory for storing the FAISS index
└── resumes/           # Directory containing resume files (PDF/DOC/DOCX)
```

## How It Works

1. **Ingestion Phase** (`ingest.py`):
   - Reads all resumes from the `resumes/` directory
   - Converts them to text and generates embeddings using Ollama's nomic-embed-text model
   - Stores the embeddings in a FAISS vector store for efficient similarity search

2. **Search Phase** (`agent.py`):
   - Takes a job description as input
   - Generates embeddings for the job description
   - Uses FAISS to find the most similar resumes based on semantic similarity
   - Returns the top matching candidates

## Customization

- Adjust the number of candidates returned by modifying the `k` parameter in the `find_candidates` function in `agent.py`
- Change the LLM model by modifying the model name in the `load_models` function
- Add more file formats by extending the `ingest.py` script

## Requirements

- langchain-ollama
- langchain
- faiss-cpu (or faiss-gpu for GPU acceleration)
- PyPDF2
- python-docx
- python-dotenv

## License



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [LangChain](https://www.langchain.com/) for the LLM framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
