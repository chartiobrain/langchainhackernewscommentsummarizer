import requests
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain import Prompt, LLMChain
import requests
import os
import time

os.environ["OPENAI_API_KEY"] = ""




llm = OpenAI(temperature=0)

final_result = []

def get_comments(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    comment_rows = soup.select("tr.athing.comtr, tr.athing.comtrtd, tr.athing.comtrtd table.comment-tree tr.athing.comtr")
    comments = []

    for comment_row in comment_rows:
        comment_div = comment_row.find("div", class_="comment")
        comment_span = comment_div.find("span", class_="commtext")

        if comment_span:
            comments.append(comment_span.get_text())

    return comments






base_url = "https://news.ycombinator.com"
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

title_rows = soup.select("tr.athing")


comment_links = soup.select(".subtext a[href*='item?id=']")


items = []

for title_row in title_rows:
    title = title_row.select_one(".titleline > a")
    subtext = title_row.find_next_sibling("tr").select_one(".subtext")
    if subtext:
        comment_link = subtext.select_one("a[href*='item?id=']")
        if comment_link:
            items.append({"title": title, "comment_link": comment_link})


for index, item in enumerate(items):
    title = item["title"]
    comment_link = item["comment_link"]
    print(comment_link)

    

    # article_url = title['href']
    #article_content = get_page_content(article_url)

    comments_url = f"{base_url}/{comment_link['href']}"
   
    comments = get_comments(comments_url)
    time.sleep(3)

    text = f"Title {index + 1}: {title.text}\n"
    for i, comment in enumerate(comments):
        text += f"Comment {i + 1}: {comment}\n"

   



    _prompt = """Write a detailed summary of the the main points in the comment thread, highlighting in particular any technnologies or github projects mentioned:


{text}


CONCISE SUMMARY:"""
    prompt = Prompt(template=_prompt, input_variables=["text"])

    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 0);
    mp_chain = MapReduceChain.from_params(llm, prompt, textSplitter)

    result = mp_chain.run(text)
    print(result)
    final_result.append(f"Title: {title.text}\nLink to Comments: {base_url}/{comment_link['href']}\nSummary: {result}\n")


    

print("\n".join(final_result))




"""
TODO
1. grab any mention of "author here" and find their user name, link to it

"""