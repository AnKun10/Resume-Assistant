import os
import json
import logging
from typing import List, Tuple
from datetime import datetime
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.output_parsers.base import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from llama_index.core import Document

from config import DOMAINS, METADATA_MODEL
from utils import extract_doc
from chatbot import load_llm


class CustomMetadataParser(BaseOutputParser):
    def parse(self, text):
        """
        Parse text that contains schema information and extract the example data.
        """
        try:
            print(text)
            # Try to parse the text as JSON
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            # Handle cases where output isn't valid JSON
            raise ValueError(f"Failed to parse JSON: {e}")
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {e}")


def get_metadata_chain(model_name: str = METADATA_MODEL):
    # Get LLM model
    metadata_model = load_llm(model_name, temperature=0.0)

    # Use the custom parser
    metadata_parser = CustomMetadataParser()

    # Create a simplified prompt that just asks for the information
    metadata_template = """
    Extract information from this resume:
    {resume}

    IMPORTANT FORMATTING INSTRUCTIONS:
    1. Return ONLY valid JSON without any comments or explanations
    2. Do not include explanatory comments in the JSON
    3. Do not wrap the JSON in markdown code blocks
    4. The JSON must have the EXACT following structure with these exact field names:
      - yob: The person's year of birth as a string (default value: "")
      - domain: The person's main working domain and must be in {domains} (default value: "")
      - education: Array of education entries with institution, degree, gpa, and dates (default value: [])
      - experience: Array of work experiences with company, title, dates (default value: [])
      - skills: Array of skill strings (default value: [])
      - languages: Array of language strings (default value: [string of input text's language (e.g. "English")])

    Example of CORRECTLY formatted response:
    {{
      "yob": "1990",
      "domain": "Engineering",
      "education": [
        {{
          "institution": "Example University",
          "degree": "B.S. Computer Science",
          "gpa": "3.8/4.0",
          "dates": "2010-2014"
        }}
      ],
      "experience": [
        {{
          "company": "Example Corp",
          "title": "Software Engineer",
          "dates": "2014-2018",
        }}
      ],
      "skills": ["Python", "JavaScript"],
      "languages": ["English"]
    }}

    DO NOT include the schema information in your response, only the data.
    """

    metadata_prompt = PromptTemplate(
        template=metadata_template,
        input_variables=["resume"],
        partial_variables={"domains": DOMAINS},
    )

    metadata_chain = metadata_prompt | metadata_model | metadata_parser
    return metadata_chain


def extract_metadata_content(resume_path: str, metadata_model: str) -> Tuple[List[Document], List[Document]]:
    # Chain for extracting metadata
    metadata_chain = get_metadata_chain(metadata_model)

    # Set up logging
    log_filename = f"resume_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Load docs
    documents = []
    unprocessed_documents = []  # contain only file names

    resumes = os.listdir(resume_path)
    logger.info(f"Starting extraction for {len(resumes)} resumes")

    loop = tqdm(resumes, desc="Extracting Metadata", position=0, leave=False)
    for i, resume in enumerate(loop):
        try:
            cur_resume_path = os.path.join(resume_path, resume)
            logger.info(f"Processing [{i + 1}/{len(resumes)}]: {resume}")

            doc_content = extract_doc(cur_resume_path)
            logger.info(f"Extracted content from {resume}, length: {len(doc_content)} chars")

            # Log before invoking the potentially memory-intensive operation
            logger.info(f"Starting metadata extraction for {resume}")

            metadata = metadata_chain.invoke({
                "resume": doc_content,
            })
            metadata["file_name"] = resume

            logger.info(f"Successfully extracted metadata for {resume}")
            documents.append(Document(text=doc_content, metadata=metadata))

            # Periodically log memory usage (optional)
            if (i + 1) % 5 == 0:
                import psutil
                mem_info = psutil.Process(os.getpid()).memory_info()
                logger.info(f"Memory usage after {i + 1} documents: {mem_info.rss / (1024 * 1024):.2f} MB")

        except Exception as e:
            unprocessed_documents.append(resume)
            logger.error(f"Error processing {resume}: {str(e)}")
            # Continue with the next file rather than crashing
            continue

    logger.info(f"Completed processing {len(documents)} out of {len(resumes)} resumes")
    return documents, unprocessed_documents
