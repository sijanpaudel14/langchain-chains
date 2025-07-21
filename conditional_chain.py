from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

os.system('clear')
load_dotenv()

# Define models
model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model3 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define output parsers
parser = StrOutputParser()

class SentimentResponse(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="Give the sentiment classification of the feedback.")

parser2 = PydanticOutputParser(pydantic_object=SentimentResponse)

# Define prompt for sentiment classification
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback as positive, negative, or neutral:\n\n"
        "{feedback}\n{format_instruction}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# This will return a dict: { "sentiment": "positive" }
classifier_chain = prompt1 | model1 | parser2

# Prompts for response based on sentiment
prompt_positive = PromptTemplate(
    template="Write an appropriate Professional and Empathetic response to the positive feedback minimum of 5 lines: {feedback}",
    input_variables=['feedback']
)
prompt_negative = PromptTemplate(
    template="Write an appropriate Professional and Empathetic response to the negative feedback minimum of 5 lines: {feedback}",
    input_variables=['feedback']
)

# Response chains
response_chain_positive = prompt_positive | model2 | parser
response_chain_negative = prompt_negative | model3 | parser

# Conditional branching chain
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'] == 'positive', response_chain_positive),
    (lambda x: x['sentiment'] == 'negative', response_chain_negative),
    RunnableLambda(lambda x: f"Sentiment is {x['sentiment']}, no response needed.")
)

# Combine full chain
chain = classifier_chain | (lambda output: {'sentiment': output.sentiment, 'feedback': feedback_text}) | branch_chain

# Example feedback
feedback_text = "This is a terrible phone. "

# Invoke chain
result = chain.invoke({'feedback': feedback_text})
print(result)

chain.get_graph().print_ascii()