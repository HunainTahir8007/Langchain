from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder


chat_template = ChatPromptTemplate([
    ("system": "You are a helpful customer support "),
    MessagesPlaceholder(variable_name = "chat_history"),    
    ("human": {query})
]
)