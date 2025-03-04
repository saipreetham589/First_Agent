from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def tech(device:str, year:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that performs a web search to only find list of upcoming device release dates and model names.
    Args:
        device: A string representing a device type mobile/laptop/watch.
        year: An integer representing the requested year.
    """
    try:
        search_query = f"list of {device} releasing in {year}"
        search_tool = DuckDuckGoSearchTool()
        results = search_tool(search_query)

        if results:
            return f"Here are some {device} expected to be released in {year}:\n{results}"
        else:
            return f"No information found for {device} releasing in {year}."
        
    except Exception as e:
        return f"Error gatering '{device}' data: {str(e)}"

@tool
def university(uni:str, domain:str, year:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that performs a web search to find univerity ranking based on USNews
    Args:
        uni: A string representing a name of university in USA.
        domain: A string representing a name of course/major.
        year: An integer representing the requested year.
    """
    try:
        search_query = f"{uni} ranking in {year} from USNews for {domain}"
        search_tool = DuckDuckGoSearchTool()
        results = search_tool(search_query)

        if results:
            return f"{uni} ranking for {domain} in {year}:\n{results}"
        else:
            return f"No information found for {domain} in {year}."
        
    except Exception as e:
        return f"Error gatering '{domain}' data: {str(e)}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, tech, university, image_generation_tool], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()