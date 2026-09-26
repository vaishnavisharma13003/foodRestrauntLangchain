from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

response = llm.invoke("Suggest a fancy Italian restaurant name.")

print(response.content)