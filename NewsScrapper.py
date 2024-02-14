# %%
#Importing essential libraries 
from newsdataapi import NewsDataApiClient
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

from time import sleep

# %%
# Creating a class called 'NewsExtractor' which helps to extract latest news based on search query.

class NewsExtractor:
    '''
    A class for extracting the latest news based on a search query.
    '''

    def __init__(self, NEWSDATA_KEY:str, search_query:str=None):
        '''
        Initializes the NewsExtractor instance.
        
        Args:
            NEWSDATA_KEY (str): The API key for the NewsDataApiClient.
            search_query (str, optional): The variable to be searched. Defaults to None.
        '''
        self.NEWSDATA_KEY = NEWSDATA_KEY
        self.search_query = search_query
    
    def get_news(self, query: str)->list:
        '''
        Retrieves the latest news based on a search query.
        
        Args:
            query (str): The variable to be searched.
        
        Returns:
            list: A list of strings containing the latest news.
        '''
        import random  # Importing the random module for later use
        api = NewsDataApiClient(apikey=self.NEWSDATA_KEY)  # Initializing the NewsDataApiClient with the provided API key
        
        if self.search_query:
            # If a search query is provided, retrieve the news based on the query, country (India), and language (English)
            response = api.news_api(q=self.search_query, country="in", language='en')
        else:
            # If no search query is provided, retrieve the general news for the country (India) and language (English)
            response = api.news_api(country="in", language='en')

        # Extract the content of each news item from the response and store it in the 'news' list
        news = [content['content'] for content in response['results']]

        # Return a randomly selected news item from the 'news' list
        return random.choice(news)


# %%
# Now, Creating a function for Summarization tool using OpenAI

def news_summarizer(NEWSDATA_KEY:str, OPENAI_KEY:str, limit:str="175 characters", total_news:int=5, sleep_seconds:int=20, search_query:str=None):
    '''
    Function that retrieves the latest news and generates short summaries for the news using an AI model.
    
    Args:
        NEWSDATA_KEY (str): The API key for the NewsDataApiClient.
        OPENAI_KEY (str): The API key for the ChatOpenAI model.
        limit (str): The maximum character limit for each news summary. Defaults to "175 characters".
        total_news (int): The number of news articles to summarize. Defaults to 5.
        sleep_seconds (int): The number of seconds to wait between each news retrieval and summarization. Defaults to 20.
        search_query (str, optional): The variable to be searched. Defaults to None.
    
    Returns:
        list: A list of responses containing the generated news summaries.
    '''
    
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0613', openai_api_key=OPENAI_KEY)  # Initializing the ChatOpenAI model with the provided parameters
    
    NE = NewsExtractor(NEWSDATA_KEY, search_query)  # Creating an instance of the NewsExtractor class
    
    newssummary = Tool(
        name='newssummary',
        func=NE.get_news,
        description='get latest news as a list of strings',
    )  # Initializing the newssummary Tool to retrieve the latest news
    
    tools = [newssummary]  # Creating a list of tools containing the newssummary Tool
    
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)  # Initializing the agent with the provided tools and ChatOpenAI model
    
    responses = []  # Initializing an empty list to store the generated news summaries
    
    prompt = f"""
    Your task is to get the latest news and then generate a short summary for this news \
    Summarize the news 
    in at most {limit}. Try to be as short as possible. Never ever cross the {limit} limit for summarized news.
    """  # The prompt for the agent to generate the news summaries
    
    for _ in range(total_news):
        response = agent.run(prompt)  # Running the agent to generate the news summary
        responses.append(response)  # Appending the generated news summary to the responses list
        sleep(sleep_seconds)  # Sleeping for the specified number of seconds before each news retrieval and summarization
    
    return responses  # Returning the list of generated news summaries

# %%
# Getting a response, Inserting Query.
response = news_summarizer(NEWSDATA_KEY = 'pub_3815512c4e6d027a90e388a0eeee9b30c2854',OPENAI_KEY = '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', limit="175 characters", total_news = 1,sleep_seconds = 20,search_query = "business")

# %%
###response

# %%



