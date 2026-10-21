import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"))

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
output_parser = StrOutputParser()

chain = (
    {"topic": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

while True:
    topic = input("Enter a topic for a joke (or an empty line to quit): ")
    if not topic:
        break
    print(chain.invoke(topic))
# print(chain.invoke("ice cream"))