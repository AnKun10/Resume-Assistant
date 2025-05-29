import torch
import streamlit as st
import numpy as np
import os
from config import RAG_K_THRESHOLD, LORA_PATHS
from peft import PeftConfig, PeftModel
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils.logging import set_verbosity_info, set_verbosity_error
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from huggingface_hub import login

# Set verbosity for more detailed error messages
set_verbosity_info()


def load_llm(path: str, temperature=0.1, max_new_tokens=2048, fine_tune=False):
    """Load a Hugging Face language model with quantization"""
    # Login to Hugging Face Hub with API token if available
    if "api_key" in st.session_state and st.session_state["api_key"]:
        login(token=st.session_state["api_key"], write_permission=False)
        os.environ["HUGGINGFACE_TOKEN"] = st.session_state["api_key"]
    
    model, tokenizer = None, None
    if fine_tune:
        try:
            path_map = LORA_PATHS[path]
            config = PeftConfig.from_pretrained(path_map)

            # Configure 4-bit quantization for better compatibility
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Load base model with 4-bit quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                token=st.session_state.get("api_key", None)
            )

            # Load LoRA adapter on top of base model
            model = PeftModel.from_pretrained(base_model, path_map)
            model = model.merge_and_unload()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(path_map, token=st.session_state.get("api_key", None))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            # Try alternative loading method without quantization
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    token=st.session_state.get("api_key", None)
                )
                model = PeftModel.from_pretrained(base_model, path_map)
                model = model.merge_and_unload()
                tokenizer = AutoTokenizer.from_pretrained(path_map, token=st.session_state.get("api_key", None))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print("Loaded fine-tuned model without quantization")
            except Exception as fallback_error:
                print(f"Fallback loading for fine-tuned model also failed: {fallback_error}")
                raise RuntimeError(f"Failed to load fine-tuned model: {str(e)}. Fallback also failed: {str(fallback_error)}")

    else:
        # Quantization setup
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                token=st.session_state.get("api_key", None)
            )
        except Exception as e:
            print(f"Error loading model with quantization: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=torch.float16,
                token=st.session_state.get("api_key", None)
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, token=st.session_state.get("api_key", None))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Pipeline configuration
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    # Explicitly set the model_id to ensure it's not None
    llm_model = ChatHuggingFace(llm=llm, model_id=path)
    return llm_model


def render(document_list: list, retriever_metadata: dict, time_elapsed: float):
    retriever_message = st.expander(f"Verbosity")
    message_map = {
        "retrieve_applicant_jd": "**A job description is detected**. The system defaults to using RAG.",
        "retrieve_applicant_id": "**Applicant IDs are provided**. The system defaults to using exact ID retrieval.",
        "no_retrieve": "**No retrieval is required for this task**. The system will utilize chat history to answer."
    }

    with retriever_message:
        st.markdown(f"Total time elapsed: {np.round(time_elapsed, 3)} seconds")
        st.markdown(f"{message_map[retriever_metadata['query_type']]}")

        if retriever_metadata["query_type"] == "retrieve_applicant_jd":
            st.markdown(f"Returning top {RAG_K_THRESHOLD} most similar resumes.")

            button_columns = st.columns([float(1 / RAG_K_THRESHOLD) for _ in range(RAG_K_THRESHOLD)], gap="small")
            for index, document in enumerate(document_list[:RAG_K_THRESHOLD]):
                with button_columns[index], st.popover(f"Resume {index + 1}"):
                    st.markdown(document)

            st.markdown(f"**Extracted query**:\n`{retriever_metadata['extracted_input']}`\n")
            st.markdown(f"**Generated questions**:\n`{retriever_metadata['subqueries_list']}`")
            st.markdown(f"**Document re-ranking scores**:\n`{retriever_metadata['retrieved_docs_with_scores']}`")

        elif retriever_metadata["query_type"] == "retrieve_applicant_id":
            st.markdown(f"Using the ID to retrieve.")

            button_columns = st.columns([float(1 / RAG_K_THRESHOLD) for _ in range(RAG_K_THRESHOLD)], gap="small")
            for index, document in enumerate(document_list[:RAG_K_THRESHOLD]):
                with button_columns[index], st.popover(f"Resume {index + 1}"):
                    st.markdown(document)

            st.markdown(f"**Extracted query**:\n`{retriever_metadata['extracted_input']}`\n")


class ChatBot():
    def __init__(self, path: str, fine_tune: bool = False):
        """Initialize the chatbot with a language model"""
        self.llm = load_llm(
            path=path,
            temperature=0.1,
            fine_tune=fine_tune
        )

    def generate_subquestions(self, question: str):
        """Generate subqueries from a job description to improve retrieval"""
        system_message = SystemMessage(content="""
            You are an expert in talent acquisition. Separate this job description into 3-4 more focused aspects for efficient resume retrieval.
            Make sure every single relevant aspect of the query is covered in at least one query. You may choose to remove irrelevant information that doesn't contribute to finding resumes such as the expected salary of the job, the ID of the job, the duration of the contract, etc.
            Only use the information provided in the initial query. Do not make up any requirements of your own.
            Put each result in one line, separated by a linebreak.
        """)

        user_message = HumanMessage(content=f"""
            Generate 3 to 4 sub-queries based on this initial job description:
            {question}
        """)

        oneshot_example = HumanMessage(content="""
            Generate 3 to 4 sub-queries based on this initial job description:

            Wordpress Developer
            We are looking to hire a skilled WordPress Developer to design and implement attractive and functional websites and Portals for our Business and Clients. You will be responsible for both back-end and front-end development including the implementation of WordPress themes and plugins as well as site integration and security updates.
            To ensure success as a WordPress Developer, you should have in-depth knowledge of front-end programming languages, a good eye for aesthetics, and strong content management skills. Ultimately, a top-class WordPress Developer can create attractive, user-friendly websites that perfectly meet the design and functionality specifications of the client.
            WordPress Developer Responsibilities:
            Meeting with clients to discuss website design and function.
            Designing and building the website front-end.
            Creating the website architecture.
            Designing and managing the website back-end including database and server integration.
            Generating WordPress themes and plugins.
            Conducting website performance tests.
            Troubleshooting content issues.
            Conducting WordPress training with the client.
            Monitoring the performance of the live website.
            WordPress Developer Requirements:
            Bachelors degree in Computer Science or a similar field.
            Proven work experience as a WordPress Developer.
            Knowledge of front-end technologies including CSS3, JavaScript, HTML5, and jQuery.
            Knowledge of code versioning tools including Git, Mercurial, and SVN.
            Experience working with debugging tools such as Chrome Inspector and Firebug.
            Good understanding of website architecture and aesthetics.
            Ability to project manage.
            Good communication skills.
            Contract length: 12 months
            Expected Start Date: 9/11/2020
            Job Types: Full-time, Contract
            Salary: 12,004.00 - 38,614.00 per month
            Schedule:
            Flexible shift
            Experience:
            Wordpress: 3 years (Required)
            web designing: 2 years (Required)
            total work: 3 years (Required)
            Education:
            Bachelor's (Preferred)
            Work Remotely:
            Yes
        """)

        oneshot_response = AIMessage(content="""
            1. WordPress Developer Skills:
              - WordPress, front-end technologies (CSS3, JavaScript, HTML5, jQuery), debugging tools (Chrome Inspector, Firebug), code versioning tools (Git, Mercurial, SVN).
              - Required experience: 3 years in WordPress, 2 years in web designing.

            2. WordPress Developer Responsibilities:
              - Meeting with clients for website design discussions.
              - Designing website front-end and architecture.
              - Managing website back-end including database and server integration.
              - Generating WordPress themes and plugins.
              - Conducting website performance tests and troubleshooting content issues.
              - Conducting WordPress training with clients and monitoring live website performance.

            3. WordPress Developer Other Requirements:
              - Education requirement: Bachelor's degree in Computer Science or similar field.
              - Proven work experience as a WordPress Developer.
              - Good understanding of website architecture and aesthetics.
              - Ability to project manage and strong communication skills.

            4. Skills and Qualifications:
              - Degree in Computer Science or related field.
              - Proven experience in WordPress development.
              - Proficiency in front-end technologies and debugging tools.
              - Familiarity with code versioning tools.
              - Strong communication and project management abilities.
        """)

        response = self.llm.invoke([system_message, oneshot_example, oneshot_response, user_message])
        result = response.content.split("\n\n")
        return result

    def generate_message_stream(self, question: str, docs: list, history: list, prompt_cls: str):
        """Generate a streaming response to a user query with context from retrieved documents"""
        context = "\n\n".join(doc for doc in docs)

        if prompt_cls == "retrieve_applicant_jd":
            system_message = SystemMessage(content="""
              You are an expert in talent acquisition that helps determine the best candidate among multiple suitable resumes.
              Use the following pieces of context to determine the best resume given a job description.
              You should provide some detailed explanations for the best resume choice.
              Because there can be applicants with similar names, use the applicant ID to refer to resumes in your response.
              If you don't know the answer, just say that you don't know, do not try to make up an answer.
            """)

            user_message = HumanMessage(content=f"""
              Chat history: {history}
              Context: {context}
              Question: {question}
            """)
        elif prompt_cls == "retrieve_applicant_id":
            system_message = SystemMessage(content="""
              You are an expert in talent acquisition that helps analyze any resumes.
              Use the following pieces of context to analyze the given resumes.
              You should provide some detailed explanations for any questions related to those resumes.
              If there are no questions about those resumes and the user only ask for retrival, you can just summarize the given resumes.
              Because there can be applicants with similar names, use the applicant ID to refer to resumes in your response.
              If you don't know the answer, just say that you don't know, do not try to make up an answer.
            """)

            user_message = HumanMessage(content=f"""
              Chat history: {history}
              Context: {context}
              Question: {question}
            """)
        else:
            system_message = SystemMessage(content="""
              You are an expert in talent acquisition that can answer any job-related concepts, requirements or questions.
              You may use the following pieces of context to answer your question.
              You can only use chat history if the question mention about information in the chat history.
              In that case, do not mention in your response that you are provided with a chat history.
              If you don't know the answer, just say that you don't know, do not try to make up an answer.
            """)

            user_message = HumanMessage(content=f"""
              Chat history: {history}
              Context: {context}
              Question: {question}
            """)

        stream = self.llm.stream([system_message, user_message])
        return stream
