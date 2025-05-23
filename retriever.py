from typing import List
from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain_core.tools import Tool
from langchain_core.prompts.chat import ChatPromptTemplate
from transformers import pipeline

from llama_index.core import Document
from config import RAG_K_THRESHOLD, QUERY_CLASSIFIER_MODEL


def get_document_nodes(index, doc_id: str) -> List[Document]:
    """Get all nodes from a document by its ID"""
    # Get all nodes from the index
    all_nodes = index.docstore.docs.values()

    # Filter nodes that belong to the specified document
    doc_nodes = [node for node in all_nodes if node.ref_doc_id == doc_id]

    if not doc_nodes:
        print(f"No nodes found for document ID: {doc_id}")
        return None

    print(f"Found {len(doc_nodes)} nodes for document ID: {doc_id}")
    return doc_nodes


class ResumeID(BaseModel):
    """List of applicant IDs to retrieve resumes for"""
    id_list: List[str] = Field(description="List of applicant IDs to retrieve resumes for")


class JobDescription(BaseModel):
    """Description of a job to retrieve similar resumes for"""
    job_description: str = Field(description="Description of a job to retrieve similar resumes for")


class Retriever():
    def __init__(self, index):
        self.index = index

    def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
        """Implement reciprocal rank fusion for re-ranking results"""
        fused_scores = {}
        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                fused_scores[doc] += 1 / (rank + k)
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        return reranked_results

    def __retrieve_docs_id__(self, question: str, k=50):
        """Retrieve document IDs based on a query"""
        retriever = self.index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(question)
        docs_score = {str(res.node.ref_doc_id): res.score for res in results}
        return docs_score

    def retrieve_id_and_rerank(self, subquestion_list: list):
        """Retrieve and rerank documents based on multiple subqueries"""
        document_rank_list = []
        for subquestion in subquestion_list:
            document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
        reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
        return reranked_documents


class ResumeRetriever(Retriever):
    def __init__(self, index):
        super(ResumeRetriever, self).__init__(index)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in talent acquisition."),
            ("user", "{input}")
        ])
        self.metadata = {
            "query_type": "no retrieve resumes",
            "extracted_input": "",
            "subqueries_list": [],
            "retrieved_docs_with_scores": []
        }
        self.query_classifier = pipeline("zero-shot-classification", model=QUERY_CLASSIFIER_MODEL)

    def classify_query(self, query: str, labels: List[str]):
        """Classify the query type"""
        result = self.query_classifier(query, labels)
        return result["labels"][0]  # Return the highest probability label

    def retrieve_docs(self, question: str, llm):
        """Retrieve documents based on the question type"""

        @tool(args_schema=ResumeID)
        def retrieve_applicant_id(id_list: list):
            """Retrieve resumes for applicants in the id_list"""
            retrieved_resumes = []

            for id in id_list:
                try:
                    resume_nodes = get_document_nodes(self.index, id)
                    file_name = resume_nodes[0].metadata["file_name"]
                    resume_with_id = "Applicant ID: " + id + " | File Name: " + file_name + "\n" + ' '.join(
                        [node.text for node in resume_nodes])
                    retrieved_resumes.append(resume_with_id)
                except:
                    return []
            return retrieved_resumes

        @tool(args_schema=JobDescription)
        def retrieve_applicant_jd(job_description: str):
            """Retrieve similar resumes given a job description"""
            # Generate subqueries for RAG Fusion approach
            subqueries_list = [job_description]
            subqueries_list += llm.generate_subquestions(question)

            self.metadata["subqueries_list"] = subqueries_list
            retrieved_ids = self.retrieve_id_and_rerank(subqueries_list)
            self.metadata["retrieved_docs_with_scores"] = retrieved_ids

            # Retrieve documents with the IDs
            retrieved_resumes = []
            for doc_id in list(retrieved_ids.keys())[:RAG_K_THRESHOLD]:
                try:
                    resume_nodes = get_document_nodes(self.index, doc_id)
                    file_name = resume_nodes[0].metadata["file_name"]
                    resume_with_id = "Applicant ID: " + doc_id + " | File Name: " + file_name + "\n" + ' '.join(
                        [node.text for node in resume_nodes])
                    retrieved_resumes.append(resume_with_id)
                except Exception as e:
                    print(f"Error retrieving document {doc_id}: {str(e)}")

            return retrieved_resumes

        def router(response: str):
            """Route the query to the appropriate retrieval function based on classification"""
            try:
                # Parse the XML response from the LLM to extract tool name and input
                import re
                tool_match = re.search(r'<tool>(.*?)</tool>', response)
                input_match = re.search(r'<tool_input>(.*?)</tool_input>', response, re.DOTALL)
                
                if tool_match and input_match:
                    tool_name = tool_match.group(1).strip()
                    tool_input = input_match.group(1).strip()
                    
                    # Update metadata
                    self.metadata["query_type"] = "retrieve resumes"
                    self.metadata["extracted_input"] = tool_input
                    
                    # Map tools
                    toolbox = {
                        "retrieve_applicant_id": retrieve_applicant_id,
                        "retrieve_applicant_jd": retrieve_applicant_jd
                    }
                    
                    # Execute tool if it exists in the toolbox
                    if tool_name in toolbox:
                        return toolbox[tool_name](tool_input), tool_name
                    else:
                        return [], "no retrieve resumes"
                else:
                    # No tool specified in the response
                    return [], "no retrieve resumes"
            except Exception as e:
                print(f"Error in router: {str(e)}")
                return [], "no retrieve resumes"

        # Determine query type using the classifier
        query_types_list = ["retrieve resumes", "no retrieve resumes"]
        query_type = self.classify_query(question, query_types_list)
        self.metadata["query_type"] = query_type
        
        # If classified as no retrieval, return empty results
        if query_type == "no retrieve resumes":
            return [], query_type
        
        # Otherwise, invoke LLM to determine which retrieval tool to use
        messages = self.prompt.format_messages(input=question)
        response = llm.llm.invoke(messages).content
        
        # Use router to process the response and execute the appropriate tool
        results, tool_used = router(response)
        return results, tool_used
