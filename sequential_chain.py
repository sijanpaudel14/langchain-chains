from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


prompt1 = PromptTemplate(
    template="Generate a detailed report on the topic: {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following report: {report}',
    input_variables=['report']
)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({
    'topic': 'Climate Change'
})
print(result)

chain.get_graph().print_ascii()
