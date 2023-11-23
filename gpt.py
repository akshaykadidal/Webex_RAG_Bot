from webex_bot.models.command import Command
from webex_bot.models.response import Response
import openai
import os
import ast  # for converting embeddings saved as strings back to arrays
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = os.getenv('OPENAI_KEY')



class gpt(Command):
    messages = []

    df = pd.read_csv("cookie_embedding.csv",converters={"embedding": lambda x: x.strip("[]").split(", ")})
    df['embedding'][0] = [float(i) for i in df['embedding'][0]]
    df['embedding'][1] = [float(i) for i in df['embedding'][1]]
    df = df.drop(columns=['Unnamed: 0'])


    messages.append({"role":"system", "content":"You are a assistant that helps peopleh  their privacy implementation questions"})
    def __init__(self):
        super().__init__()
    
    # search function
    def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 10
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def num_tokens(text: str, model: str = GPT_MODEL) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = gpt.strings_ranked_by_relatedness(query, df)
        introduction = 'Use the below articles on Cookie implemetnation for websites. If the answer cannot be found in the articles, write "I am sorry, I do not have an answer for you. Pease raise a ticket with DPP team.."'
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            next_article = f'\n\n article section:\n"""\n{string}\n"""'
            if (
                gpt.num_tokens(message + next_article + question, model=model)
                > token_budget
            ):
                break
            else:
                message += next_article
        return message + question
    

    def execute(self, query, attachement_actions, activity, df: pd.DataFrame = df, token_budget: int = 4096 - 500, print_message: bool = False):
        #openai.api_key = os.getenv('OPENAI_KEY')
        
        
        message = gpt.query_message(query, df, model=GPT_MODEL,  token_budget=token_budget)
        if print_message:
            print(message)
        self.messages = [
            {"role": "system", "content": "You answer questions about cookie implementation or assesment."},
            {"role": "user", "content": message},]
    
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0) # this is the degree of randomness of the model's output
        print(message)    
#        self.messages.append({"role": "assistant", "content": message}) # for context retention not needed
        return (response.choices[0].message["content"])
        #return(messages)
