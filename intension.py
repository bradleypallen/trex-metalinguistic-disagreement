import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain.output_parsers import RegexParser
from langchain_core.prompts import PromptTemplate

class Intension:
    """Represents a zero-shot chain-of-thought implementing an intension for triples."""

    PROMPT_TEMPLATE = """
Is the following knowledge graph triple true or false? 
Subject: {s_label}
Predicate: {p_label} ({p_definition})
Object: {o_label}

Determine the truth value of this triple in 
a hypothetical world where the following is true:
{world}

Let's think step by step. Provide a rationale for 
your decision, then based on that rationale,
provide an answer of 1 if true, otherwise 
provide an answer of 0.
###
Rationale: {{rationale}}
Answer: {{answer}}
"""

    PROMPT = PromptTemplate(input_variables=["s_label", "p_label", "p_definition", "o_label", "world"], template=PROMPT_TEMPLATE)

    OUTPUT_PARSER = RegexParser(
        regex=r"(?is).*Rationale:\**\s*(.*?)Answer:\**\s*(0|1)",
        output_keys=["rationale", "answer"],
        default_output_key="rationale"
    )
    
    def __init__(self, model="gpt-4-0125-preview", temperature=0.1):
        """
        Initializes a classification procedure for a concept, given a unique identifier, a term, and a definition.
        
        Parameters:
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model = model
        self.temperature = temperature
        self.llm = self._llm(model, temperature)
        self.chain = LLMChain(llm=self.llm, prompt=self.PROMPT, output_parser=self.OUTPUT_PARSER)

    def _llm(self, model, temperature=0.1):
        if model in [ "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4o-2024-05-13" ]:
            return ChatOpenAI(model_name=model, temperature=temperature)
        elif model in [ "claude-3-opus-20240229", "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307" ]:
            return ChatAnthropic(
                temperature=temperature, 
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], 
                model_name=model
            )
        elif model in [ "gemini-1.0-pro" ]:
            return ChatGooglePalm(
                temperature=temperature, 
                google_api_key=os.environ["GOOGLE_API_KEY"], 
                model=model
            )
        elif model in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "mistralai/Mixtral-8x7B-Instruct-v0.1", 
            "mistralai/Mistral-7B-Instruct-v0.3", 
            "google/gemma-2-9b-it",
            "google/gemma-7b-it", 
            "google/gemma-2b-it",
            "meta-llama/Meta-Llama-3-70B-Instruct", 
            "microsoft/Phi-3-mini-128k-instruct",
            ]:
            return HuggingFaceEndpoint(
                repo_id=model, 
                temperature=temperature, 
                timeout=300,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
            )
        else:
            raise Exception(f'Model {model} not supported')
