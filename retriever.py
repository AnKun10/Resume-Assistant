import xml.etree.ElementTree as ET
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import tool
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
            ("system", """You are an expert in talent acquisition. Respond with an XML string in the following format:
            <response>
                <type>final_answer|tool_call</type>
                <tool_name>retrieve_applicant_id|retrieve_applicant_jd|null</tool_name>
                <tool_input>...</tool_input>
                <output>...</output>
            </response>

            IMPORTANT INSTRUCTIONS:
            1. For queries that ask for retrieve or find suitable resumes based on the given job descriptions, job requirements or job postings use:
               - <type>tool_call</type>
               - <tool_name>retrieve_applicant_jd</tool_name>
               - <tool_input>[FULL JOB DESCRIPTION]</tool_input>
            
            2. For queries that specifically mention applicant IDs or resume IDs, use:
               - <type>tool_call</type>
               - <tool_name>retrieve_applicant_id</tool_name>
               - <tool_input>[LIST OF IDS]</tool_input>
            
            3. For queries that don't require retrieve resume, use:
               - <type>final_answer</type>
               - <tool_name>null</tool_name>
               - <output>[YOUR ANSWER]</output>
            
            4. If you don't know the answer, just say that in the <output> field.
            
            5. NEVER respond without using this exact XML format."""),
            ("user", "{input}")
        ])
        self.metadata = {
            "query_type": "no_retrieve",
            "extracted_input": "",
            "subqueries_list": [],
            "retrieved_docs_with_scores": []
        }
        # Using this when llm cannot handle query classification (Implement later)
        self.query_classifier = pipeline("zero-shot-classification", model=QUERY_CLASSIFIER_MODEL)

    def classify_query(self, query: str, labels: List[str]):
        """Classify the query type"""
        result = self.query_classifier(query, labels)
        return result
    
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
            try:
                # Parse XML response
                root = ET.fromstring(response.strip())
                response_type = root.find("type").text
                tool_name = root.find("tool_name").text
                tool_input = root.find("tool_input").text
                output = root.find("output").text

                if response_type == "final_answer":
                    return output or ""

                if response_type == "tool_call":
                    # Update metadata
                    self.metadata["query_type"] = tool_name
                    self.metadata["extracted_input"] = tool_input

                    # Map tools
                    toolbox = {
                        "retrieve_applicant_id": retrieve_applicant_id,
                        "retrieve_applicant_jd": retrieve_applicant_jd
                    }

                    if tool_name not in toolbox:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    # Execute tool
                    return toolbox[tool_name].run(tool_input)

                raise ValueError("Invalid response type")

            except ET.ParseError:
                # Treat invalid XML as final answer
                return response
            except Exception as e:
                return f"Error: {str(e)}"

        # Invoke LLM
        messages = self.prompt.format_messages(input=question)
        response = llm.llm.invoke(messages).content
        return router(response)
