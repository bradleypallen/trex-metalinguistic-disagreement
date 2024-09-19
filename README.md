# trex-metalinguistic-disagreement
 Experiments using the T-REx alignment dataset in the detection of metalinguistic disagreements between knowledge graphs and knowledge-enhanced LLMs

 [Bradley P. Allen](https://orcid.org/0000-0003-0216-3930) and [Paul T. Groth](https://orcid.org/0000-0003-0183-6910)   
 [INtelligent Data Engineering Lab](https://indelab.org/)  
 University of Amsterdam, Amsterdam, The Netherlands

## Overview
Knowledge-enhanced large language models (LLMs) are neurosymbolic systems that aim to reduce hallucination by grounding their outputs in various knowledge sources. Evaluating these models for tasks like fact extraction typically involves comparing their output to a knowledge graph (KG) using accuracy metrics. These evaluations often assume that discrepancies between a KG and an LLM represent factual disagreements. However, human discourse frequently features *metalinguistic disagreement*, where parties differ not on facts but on the meaning of the language used to express them. Given the sophistication of natural language processing and generation using LLMs, we ask: do metalinguistic disagreements occur between knowledge graphs and knowledge-enhanced LLMs? In this work, we apply a formal framework based on intensional semantics to represent meaning using knowledge-enhanced LLMs to identify instances of both factual and metalinguistic disagreement between Wikidata and a knowledge-enhanced LLM using [TREx, a large-scale dataset aligning Wikipedia abstracts with Wikidata triples](https://hadyelsahar.github.io/t-rex/).
Our findings suggest that addressing disagreements through *metalinguistic negotiation* may become an important task in neurosymbolic knowledge engineering. 

There are two principal contributions of this work: first, the definition of an approach to the representation of meaning using a knowledge-enhanced LLM; and second, the use of that approach to provide experimental evidence that metalinguistic disagreement does occur between knowledge graphs and knowledge-enhanced LLMs. This repository contains the data and code involved in our evaluation of this approach.

## License
MIT.

## Requirements
- Python 3.11 or higher.
- OPENAI_API_KEY and HUGGINGFACE_API_TOKEN environment variables set to your respective OpenAI and Hugging Face API keys.

## Installation
    $ git clone https://github.com/bradleypallen/trex-metalinguistic-disagreement.git
    $ cd trex-metalinguistic-disagreement
    $ python -m venv env
    $ source env/bin/activate
    $ pip install -r requirements.txt

## Software and data artifacts in this repository

This repository contains code and data for a set of experiments investigating the concept of *metalinguistic disagreement* in the conjoined use of knowledge graphs and LLMs.

### Code

#### Python classes

- [intension.py](https://github.com/bradleypallen/trex-metalinguistic-disagreement/blob/main/intension.py): A zero-shot chain-of-thought classifier for computing the truth value of a knowledge graph triple based on a first-order intensional semantics.
- [md-detector.py](https://github.com/bradleypallen/trex-metalinguistic-disagreement/blob/main/md_detector.py): A zero-shot chain-of-thought classifier for determining if a truth value assignment of false to a triple is due to metalinguistic (as opposed to factual) disagreement.

#### Notebooks

- [trex_experiment_generator.ipynb](https://github.com/bradleypallen/trex-metalinguistic-disagreement/blob/main/trex_experiment_generator.ipynb): Generating the experimental dataset from T-REx.
- [trex_experiments.ipynb](https://github.com/bradleypallen/trex-metalinguistic-disagreement/blob/main/trex_experiments.ipynb): Assigning truth values aligned triples using a zero-shot chain-of-thought.
- [md_detection.ipynb](https://github.com/bradleypallen/trex-metalinguistic-disagreement/blob/main/md_detection.ipynb): Classifying instances of metalinguistic disagreement for "false negative" errors.
- [trex_performance_statistics.ipynb](https://github.com/bradleypallen/trex-metalinguistic-disagreement/blob/main/trex_performance_statistics.ipynb): Performance statistics and error analysis.

### Data

- [data/trex_sample_triples.json](https://github.com/bradleypallen/trex-metalinguistic-disagreement/blob/main/data/trex_sample_triples.json): The sample generated from the T-REx sample file.
- [experiments](https://github.com/bradleypallen/trex-metalinguistic-disagreement/tree/main/experiments): A directory containing the truth value assignments for each evaluated LLM.
