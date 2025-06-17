import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

    
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
parser= PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

#functions = []
#llm_with_tools = llm.bind_functions(functions=functions)

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools

)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query":query})
#response = llm.invoke("What dis the meaning of life?")

#print(raw_response)

try:
    structured_response = parser.parse(raw_response["output"])
    print(structured_response)
except Exception as e:
    print("error parsing response", e, "Raw response - ", raw_response)

#print(structured_response.topic)


'''
try:
    raw_output_string = raw_response.get("output")

    # Remove the '```json\n' prefix and '\n```' suffix
    if raw_output_string.startswith("```json\n") and raw_output_string.endswith("\n```"):
        json_string = raw_output_string[len("```json\n"):-len("\n```")]
    else:
        # Handle cases where the output might not be wrapped as expected
        # For now, let's assume it's always wrapped or throw an error
        raise ValueError("Raw output string is not in the expected '```json\\n...\\n```' format.")

    # Now, parse the extracted JSON string
    # Assuming 'parser' is a JSON parser (e.g., json.loads)
    structured_response = json.loads(json_string) # Or parser.parse(json_string) if parser is a custom JSON parser
    print(structured_response)

except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
'''