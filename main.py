import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
load_dotenv()

# defining pydantic model for output parser
class Headings(BaseModel):
    title: str
class Outline(BaseModel):
    title: str
    headings: list[Headings]

# Get extract title and headings
def formatter(Outline: Outline):
    title = f"{Outline.title}"
    Headings = ""
    for heading in Outline.headings:
        Headings += f"{heading.title}\n"
    return title, Headings    
# Save to markdown
def save_to_markdown(text: str):
    with open("article.md", "w") as f:
        f.write(text)
# defining output parser
parser = PydanticOutputParser(pydantic_object=Outline)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv('API_KEY'),
    # system_instruction="You are an expert copywriter.",
)

brainstorm_outline_prompt = PromptTemplate(
    template="I want you to brainstorm about {topic}. Come up with a detailed outline of the topic.\nThe title should trigger positive SEO for the topic. The headings and sub-headings should reflect the title.\nreturn the outline in the following format: {format}",
    input_variables=["topic"],
    partial_variables={"format": parser.get_format_instructions()}
)

article_generation_prompt = PromptTemplate(
    template="I want you to generate an elaborate article about {title}. Keep the tone authoritative and interesting. These are the headings for the article:\n{headings}\nReturn the article in the markdown format.",
    input_variables=["title", "headings"]
)
brainstorm_chain = ( brainstorm_outline_prompt
    | llm
    | parser)

article_chain = (article_generation_prompt
    | llm
    )


if __name__ == '__main__':
    brainstorm = brainstorm_chain.invoke({"topic": "artificial intelligence"})
    title, headings = formatter(brainstorm)
    article = article_chain.invoke({"title": title, "headings": headings})
    print(article.content)
    save_to_markdown(article.content)
    