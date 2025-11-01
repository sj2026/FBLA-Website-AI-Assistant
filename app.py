from pydantic import BaseModel 
from typing import List
from fastapi import FastAPI

from Chatbot import Chatbot

class Content(BaseModel):
    content: str
    
class Document(BaseModel):
    contents :list[Content] = []

class Question(BaseModel):
    question : str

chatbot = Chatbot()
app = FastAPI()

@app.post("/loadDocuments")
def load_document(contents : List[Content]):
    """
    Create a new instance of the chatbot and
    loads the given JSON into chatbot memory

    Args:
        contents: List of JSON to load

    """
    documents = []
    for content in contents:
        documents.append({'content' : content.content})
    #print(documents)
    chatbot.createChatbot(documents)
    return "Success"


@app.post("/answerQuestion")
def answer_question(question: Question):
    """
    Invokes the chatbot with the given question
    
    Args:
        question: Question for the chatbot
    Returns:
            response from chatbot.
    """
    answer = chatbot.answerQuestion(question.question)
    return answer
