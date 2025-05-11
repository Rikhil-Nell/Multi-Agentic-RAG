# --- CODE FLOW OVERVIEW ---
# This script processes documents (initially designed for markdown, with a PDF utility included)
# by breaking them into manageable chunks and enriching them with AI-generated metadata.
# The processed chunks, including their content, summaries, titles, and embeddings, are then
# stored in a Supabase database.
#
# The main workflow is orchestrated by `process_and_store_document()`:
# 1. Document text is split into chunks using `chunk_text()`.
# 2. For each chunk, `process_chunk()` is called (concurrently):
#    a. `get_title_and_summary()`: An LLM (Groq Llama 3.3 70B) generates a title and summary.
#    b. `get_embedding()`: OpenAI's embedding model generates a vector embedding for the chunk's content.
#    c. A `ProcessedChunk` dataclass instance is created.
# 3. Each `ProcessedChunk` is then inserted into the Supabase 'documents' table
#    using `insert_chunk()` (concurrently).
#
# Client initializations for OpenAI, Groq, and Supabase are handled globally.
# Configuration is managed via a `Settings` class (details not shown here but assumed).

# --- IMPORTS ---
import io
import asyncio
import json
from dataclasses import dataclass
from typing import List, Dict, Any 

import pdfplumber
from groq import AsyncGroq
from openai import AsyncOpenAI
from supabase import create_client, Client

from settings import Settings

# --- SETTINGS & CLIENT INITIALIZATION ---
settings = Settings()

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
groq_client = AsyncGroq(api_key=settings.groq_api_key)
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)

# --- DATA STRUCTURES ---
@dataclass
class ProcessedChunk:
    """
    Represents a processed chunk of a document, ready for storage.
    """
    filename: str
    chunk_number: int
    title: str
    summary: str
    content: str
    embedding: List[float]

# --- CORE FUNCTIONS ---

async def extract_text_from_pdf(file_object: io.BytesIO) -> str:
    """
    Extracts all text from a given PDF file.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A string containing all extracted text from the PDF, with pages separated by double newlines.
    """
    text = ""
    try:
        with pdfplumber.open(file_object) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    text += "\n\n"
    except Exception as e:
        print(f"Error extracting text from PDF {file_object}: {e}")
        return ""
    return text.strip()

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Splits a given text into smaller chunks, attempting to respect paragraph and code block boundaries.

    Args:
        text: The input string to be chunked.
        chunk_size: The target maximum size for each chunk.

    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        current_slice = text[start:end]
        potential_end = end

        # Try to find a code block boundary (```) within the current slice
        # Heuristic: Only consider if the code block marker is past 30% of the chunk size
        # to avoid very small initial chunks if a code block starts early.
        code_block_index = current_slice.rfind('```')
        if code_block_index != -1 and code_block_index > chunk_size * 0.3:
            potential_end = start + code_block_index + 3 # Include the ``` marker
        else:
            # If no suitable code block, try to break at a paragraph end (\n\n)
            paragraph_break_index = current_slice.rfind('\n\n')
            if paragraph_break_index != -1 and paragraph_break_index > chunk_size * 0.3:
                potential_end = start + paragraph_break_index + 2 # Include the \n\n
            else:
                # If no paragraph break, try to break at a sentence end (. )
                sentence_break_index = current_slice.rfind('. ')
                if sentence_break_index != -1 and sentence_break_index > chunk_size * 0.3:
                    potential_end = start + sentence_break_index + 1 # Include the period

        # Extract chunk and clean it up
        chunk_content = text[start:potential_end].strip()
        if chunk_content: # Ensure the chunk is not empty
            chunks.append(chunk_content)

        # Move start position for the next chunk
        # Ensure start always moves forward to prevent infinite loops on unprocessed text
        start = max(start + 1, potential_end)


    return chunks


async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generates an embedding vector for the given text using OpenAI's API.

    Args:
        text: The text to embed.
        model: The OpenAI embedding model to use.

    Returns:
        A list of floats representing the embedding vector.
        Returns a zero vector of the expected dimension (1536 for text-embedding-3-small) on error.
    """
    try:
        response = await openai_client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 1536


async def get_title_and_summary(chunk_content: str, filename: str, model: str = "llama-3.3-70b-versatile") -> Dict[str, str]: # Corrected model name based on usage
    """
    Extracts a title and summary from a given text chunk using an LLM.
    (Uses Groq with Llama 3.1 70B model as per client setup and call)

    Args:
        chunk_content: The text chunk to process.
        filename: The name of the source file, for context.
        model: The Groq model to use for generation.

    Returns:
        A dictionary with 'title' and 'summary' keys.
        Returns error messages in title and summary on failure.
    """
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
Return a JSON object with 'title' and 'summary' keys.
For the title: If this seems like the start of a document, extract its main title. If it's a middle chunk, derive a concise, descriptive title relevant to this chunk's content.
For the summary: Create a concise summary (1-3 sentences) of the main points in this chunk.
Keep both title and summary informative and directly related to the provided content."""

    context_length = 1500
    content_for_llm = chunk_content[:context_length]
    if len(chunk_content) > context_length:
        content_for_llm += "..."

    try:
        response = await groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Source Filename: {filename}\n\nChunk Content:\n{content_for_llm}"}
            ],
            response_format={"type": "json_object"}
        )

        extracted_data = json.loads(response.choices[0].message.content)
        return {
            "title": extracted_data.get("title", "Title not found"),
            "summary": extracted_data.get("summary", "Summary not found")
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"LLM raw response: {response.choices[0].message.content if response and response.choices else 'No response'}")
        return {"title": "Error: Invalid JSON response", "summary": "Error: Could not parse summary from LLM"}
    except Exception as e:
        print(f"Error getting title and summary using model {model}: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}


async def process_chunk(chunk_content: str, chunk_number: int, filename: str) -> ProcessedChunk:
    """
    Processes a single chunk of text to extract metadata and generate an embedding.

    Args:
        chunk_content: The actual text content of the chunk.
        chunk_number: The sequential number of this chunk within the document.
        filename: The name of the source file.

    Returns:
        A ProcessedChunk object containing the original content and all extracted metadata.
    """
    print(f"Processing chunk {chunk_number} for {filename}...")

    extracted_meta = await get_title_and_summary(chunk_content, filename)

    embedding = await get_embedding(chunk_content)

    return ProcessedChunk(
        filename=filename,
        chunk_number=chunk_number,
        title=extracted_meta['title'],
        summary=extracted_meta['summary'],
        content=chunk_content,
        embedding=embedding
    )

async def insert_chunk(db_client: Client, chunk_data: ProcessedChunk) -> Any:
    """
    Inserts a processed chunk into the Supabase 'documents' table.

    Args:
        db_client: The Supabase client instance.
        chunk_data: The ProcessedChunk object to insert.

    Returns:
        The result of the Supabase insert operation, or None on error.
    """
    try:
        
        data_to_insert = {
            "filename": chunk_data.filename,
            "chunk_number": chunk_data.chunk_number,
            "title": chunk_data.title,
            "summary": chunk_data.summary,
            "content": chunk_data.content,
            "embedding": chunk_data.embedding
        }

        result = await asyncio.to_thread(
            db_client.table("documents").insert(data_to_insert).execute
        )
        # result = db_client.table("documents").insert(data_to_insert).execute() # Original sync call
        print(f"Successfully inserted chunk {chunk_data.chunk_number} for {chunk_data.filename}")
        return result
    except Exception as e:
        print(f"Error inserting chunk {chunk_data.chunk_number} for {chunk_data.filename}: {e}")
        return None

# --- MAIN ORCHESTRATION ---

async def process_and_store_document(file: Any) -> None:
    """
    Processes a given document (as a string) by chunking, extracting metadata,
    generating embeddings, and storing everything in Supabase.

    Args:
        file: the files ot be processed and stored
    
    Returns:
        None
    """

    filename = file.name
    document_content = await extract_text_from_pdf(file)

    print(f"Starting processing for document: {filename}")

    text_chunks = chunk_text(text=document_content)
    if not text_chunks:
        print(f"No chunks were generated for {filename}. Skipping further processing.")
        return

    print(f"Document {filename} split into {len(text_chunks)} chunks.")

    processing_tasks = [
        process_chunk(chunk_content=chunk, chunk_number=i, filename=filename)
        for i, chunk in enumerate(text_chunks)
    ]
    processed_chunks_list = await asyncio.gather(*processing_tasks, return_exceptions=True)

    valid_processed_chunks = [chunk for chunk in processed_chunks_list if isinstance(chunk, ProcessedChunk)]
    for i, result in enumerate(processed_chunks_list):
        if isinstance(result, Exception):
            print(f"Error processing chunk {i} for {filename}: {result}")


    if not valid_processed_chunks:
        print(f"No chunks were successfully processed for {filename}. Aborting storage.")
        return

    print(f"Successfully processed {len(valid_processed_chunks)} chunks for {filename}.")

    insertion_tasks = [
        insert_chunk(db_client=supabase,chunk_data=p_chunk) # Pass the global supabase client
        for p_chunk in valid_processed_chunks
    ]
    await asyncio.gather(*insertion_tasks, return_exceptions=True)

    print(f"Finished processing and storing document: {filename}")

if __name__ == "__main__":
    res = asyncio.run(get_embedding("hello"))
    print(res[:10])