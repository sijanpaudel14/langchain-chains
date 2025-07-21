import time

start_time = time.time()

# 👉 Your code starts here


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()

# Define models
model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model3 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define prompt templates
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n\n{text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short Q&A pairs from the following text in this format:\nQ: question\nA: answer\n\nText:\n{text}",
    input_variables=['text']
)

# This one will use final merged output
prompt3 = PromptTemplate(
    template="Merge the provided notes and questions into a single document.First provide only notes and then question answers \n\nNotes:\n{notes}\n\nQuestions:\n{questions}",
    input_variables=['notes', 'questions']
)

parser = StrOutputParser()

# Parallel processing for notes and questions
parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'questions': prompt2 | model2 | parser
    }
)

# Merge chain
merged_chain = prompt3 | model3 | parser

# Complete flow: parallel -> merge
chain = parallel_chain | merged_chain

# Sample input
text = """Artificial Intelligence (AI) is a transformative field of computer science that focuses on creating machines capable of performing tasks that traditionally require human intelligence. These tasks include problem-solving, decision-making, understanding natural language, recognizing patterns and images, and even exhibiting creativity. AI is built on a foundation of mathematics, statistics, and logic, often enhanced through the use of machine learning algorithms that allow systems to learn from data and improve their performance over time without being explicitly programmed. The field of AI encompasses a wide range of sub-disciplines, such as natural language processing (NLP), computer vision, robotics, and neural networks. In recent years, advancements in computational power and the availability of large datasets have significantly accelerated the development of AI technologies, leading to practical applications in nearly every sector, including healthcare, finance, education, transportation, and entertainment. For example, AI is used to power voice assistants like Siri and Alexa, detect diseases from medical scans with higher accuracy, automate trading in stock markets, and personalize content on streaming platforms. Despite its potential, AI also raises critical ethical concerns such as bias in algorithms, data privacy, job displacement due to automation, and the need for transparency and accountability in decision-making systems. To address these challenges, researchers, policymakers, and technologists are increasingly working together to ensure the development of responsible and human-aligned AI. Ultimately, AI holds the promise of greatly augmenting human capabilities, solving complex global problems, and reshaping the way we live and work in the digital age."""

# Invoke the chain
result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()

# 👈 Your code ends here

end_time = time.time()
print(f"\n⏱️  Time taken: {end_time - start_time:.4f} seconds")