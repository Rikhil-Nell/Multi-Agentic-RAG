# Multi-Agentic RAG System

## Overview

The **Multi-Agentic RAG System** is a prototype chatbot application that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate and context-aware responses. It integrates multiple tools and APIs to process user queries, retrieve relevant documentation, and provide dictionary definitions when needed. The system is designed to handle PDF uploads, process them into chunks, and store them for efficient retrieval during conversations.

## Features

- **PDF Upload and Processing**: Upload up to 3 PDF files, which are processed into chunks for retrieval.
- **Retrieval-Augmented Generation (RAG)**: Retrieve relevant documentation chunks based on user queries.
- **Dictionary Lookup**: Fetch word definitions using the Merriam-Webster API.
- **Streamlit Interface**: User-friendly web interface for chatting and managing uploads.
- **Terminal Chat Support**: Interact with the bot directly via the terminal.

## Setup Instructions

### Prerequisites

- Python 3.13 or higher
- Git
- [uv](https://github.com/ultraviolet-ai/uv) (optional but recommended for dependency management)

### Clone the Repository

```bash
git clone https://github.com/Rikhil-Nell/Multi-Agentic-RAG.git
cd Multi-Agentic-RAG
```

### Installation

#### Option 1: Using uv (Preferred)

Install uv if not already installed:

```bash
pip install uv
```

Sync dependencies:

```bash
uv sync
```

Activate the virtual environment:

```bash
.\.venv\Scripts\activate
```

#### Option 2: Using pip and Virtual Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### Streamlit Web Interface

To launch the Streamlit interface:

```bash
streamlit run app.py
```

### Terminal Chat

To interact with the bot via the terminal:

```bash
python main.py
```

## Environment Variables

The application requires the following environment variables to be set in a `.env` file:

```bash
GROQ_API_KEY=<your_groq_api_key>
MW_API_KEY=<your_merriam_webster_api_key>
OPENAI_API_KEY=<your_openai_api_key>
SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>
```

Ensure the `.env` file is placed in the root directory of the project.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details
