
import time


from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage


from LLaMa import LLaMa

    
class Chatbot():
    retrieval_chain : None
    
    def createChatbot(self, documents):

        """
        Create a new instance of the LLM and
        loads the given documents into the memory

        Args:
            contents: List of JSON documents to load

        """

        texts = [doc["content"] for doc in documents]

        retriever = FAISS.from_texts(
            texts,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        ).as_retriever(k=5)


        faq_template = """
        You are a chat agent for my E-Commerce Company. As a chat agent, it is your duty to help the human with their inquiry and make them a happy customer.

        Help them, using the following context:
        <context>
        {context}
        </context>
        """

        faq_prompt = ChatPromptTemplate.from_messages([
            ("system", faq_template),
            MessagesPlaceholder("messages")
        ])


        document_chain = create_stuff_documents_chain(LLaMa(), faq_prompt)

        def parse_retriever_input(params):
            return params["messages"][-1].content

        self.retrieval_chain = RunnablePassthrough.assign(
            context=parse_retriever_input | retriever
        ).assign(answer=document_chain)

    def answerQuestion(self, question):  
        """
        Invokes the LLM with the given question
        
        Args:
            question: Question for the LLM
        Returns:
                response from LLM.
        """
        start_time = time.perf_counter()
        response = self.retrieval_chain.invoke({
            "messages": [
                HumanMessage(question)
            ]
        })
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"*** Response received from model : Elapsed time: {elapsed_time:.6f} seconds")
        return response
    


