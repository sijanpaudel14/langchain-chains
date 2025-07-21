# LangChain Chains Examples

This repository demonstrates several types of chains using LangChain, including conditional, parallel, sequential, and simple chains. The examples use Google Gemini models and show how to build flexible, modular workflows for AI-powered tasks.

## Prerequisites

- Python 3.8+
- `langchain-core`, `langchain-google-genai`, `python-dotenv`, `pydantic`
- Google Generative AI API key (set in `.env`)

## Files Overview

### 1. `conditional_chain.py`

- **Purpose:** Classifies sentiment of feedback and generates a professional, empathetic response based on the sentiment.
- **Key Concepts:**
  - Uses a Pydantic model for output parsing.
  - Branches logic using `RunnableBranch` to select response chain based on sentiment.
  - Demonstrates prompt engineering for both classification and response generation.
  - Shows how to combine chains and visualize the chain graph.

### 2. `parallel_chain.py`

- **Purpose:** (Assumed) Runs multiple chains in parallel for tasks like multi-model inference or batch processing.
- **Key Concepts:**
  - Demonstrates how to execute chains concurrently.
  - Useful for speeding up workflows that do not depend on sequential results.

### 3. `sequential_chain.py`

- **Purpose:** (Assumed) Runs chains in a defined sequence, passing outputs from one to the next.
- **Key Concepts:**
  - Shows how to build multi-step workflows.
  - Useful for tasks like data extraction, transformation, and final output generation.

### 4. `simple_chain.py`

- **Purpose:** (Assumed) Demonstrates a basic chain for single-step inference or transformation.
- **Key Concepts:**
  - Good starting point for understanding LangChain's chaining mechanism.

## How to Run

1. Clone the repository.
2. Install dependencies:
   ```fish
   pip install langchain-core langchain-google-genai python-dotenv pydantic
   ```
3. Set up your `.env` file with your Google Generative AI API key.
4. Run any example:
   ```fish
   python conditional_chain.py
   ```

## What You Learn

- How to use LangChain's chaining API for building modular AI workflows.
- How to parse and branch outputs using Pydantic and custom logic.
- How to design prompts for both classification and response generation.
- How to visualize and debug chain graphs.
- How to run chains in parallel and sequentially for different use cases.

## References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Google Generative AI](https://ai.google.dev/)
- [Pydantic](https://docs.pydantic.dev/)

---

Feel free to explore each file for more details and experiment with different feedback inputs and chain configurations!
