import os
import pickle
import logging
import shutil
from typing import List, Tuple
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from llama_index.core import Document, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core import Settings
import faiss


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_files(source_dir, target_dir):
    """Copy files from source directory to target directory"""
    os.makedirs(target_dir, exist_ok=True)
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        shutil.copy(source_file, target_file)


def extract_doc(data_path: str) -> str:
    """Extract text content from a PDF document"""
    doc = " ".join([page.page_content for page in PyPDFLoader(data_path).load()])
    return doc


def save_documents_list(documents: List[Document], pkl_path: str):
    """Save a list of documents to a pickle file"""
    with open(pkl_path, 'wb') as f:
        pickle.dump(documents, f)


def load_documents_list(pkl_path: str) -> List[Document]:
    """Load a list of documents from a pickle file"""
    with open(pkl_path, 'rb') as f:
        documents = pickle.load(f)
    return documents


def initialize_index(docs: List[Document], model_name: str, model_dim: int) -> VectorStoreIndex:
    """Initialize a vector index from a list of documents"""
    # Configure LlamaIndex to use HuggingFace embedding model
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.embed_model = embed_model

    # Create a semantic splitter that uses embeddings for more contextual splitting
    text_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )

    # Process documents into semantically coherent nodes
    all_nodes = text_splitter.get_nodes_from_documents(docs)
    print(f"Split {len(docs)} documents into {len(all_nodes)} total nodes")

    # Create FAISS vector store with appropriate dimensions
    faiss_index = faiss.IndexFlatL2(model_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index with the nodes
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context
    )

    print(f"Created initial index with {len(all_nodes)} semantically split nodes")
    return index


def add_docs_to_index(docs: List[Document], index: VectorStoreIndex) -> VectorStoreIndex:
    """Add documents to an existing index"""
    # Retrieve the embedding model from settings for consistency
    embed_model = Settings.embed_model

    # Use the same semantic splitter for consistency
    text_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )

    # Process new documents into semantically coherent nodes
    new_nodes = text_splitter.get_nodes_from_documents(docs)
    print(f"Split {len(docs)} new documents into {len(new_nodes)} nodes")

    # Insert the new nodes into the existing index
    for node in new_nodes:
        index.insert(node)

    print(f"Added {len(new_nodes)} semantically split nodes to the index")
    return index


def save_index(index: VectorStoreIndex, index_path: str):
    """Save an index to disk"""
    index.storage_context.persist(persist_dir=index_path)
    print(f"FAISS Index saved to {index_path}")


def load_index(index_path: str) -> VectorStoreIndex:
    """Load an index from disk"""
    faiss_vector_store = FaissVectorStore.from_persist_dir(index_path)
    storage_context = StorageContext.from_defaults(
        vector_store=faiss_vector_store,
        persist_dir=index_path
    )
    index = load_index_from_storage(storage_context)
    print(f"FAISS Index loaded from {index_path}")
    return index


def show_processed_document(document: Document, index: VectorStoreIndex):
    """Display a processed document and its nodes"""
    from retriever import get_document_nodes

    # Get document nodes
    print(f"File Name: {document.metadata['file_name']}")
    doc_nodes = get_document_nodes(index, document.doc_id)

    print("=" * 80)
    # Print each node with its metadata
    for i, node in enumerate(doc_nodes):
        print(f"Node {i + 1}/{len(doc_nodes)}")
        print(f"Node ID: {node.node_id}")
        print(f"Text ({len(node.text)} chars):")
        print("-" * 40)
        print(node.text)
        print("-" * 40)
        print(f"Metadata: {node.metadata}")
        print("=" * 80)
