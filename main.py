
BOT_AVATAR_LETTER = "A"
USER_AVATAR_LETTER= "M"
CHAT_MAX_WIDTH = "800px"
PROJECT_ID = "development-411716"
LOCATION = "europe-west4" 
REGION = LOCATION
MODEL = "gemini-1.5-flash"

import requests
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Literal
import base64

from bs4 import BeautifulSoup, SoupStrainer

from langchain_community.document_loaders import RecursiveUrlLoader

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    SafetySetting,
    Part,
)  

import mesop as me
import mesop.labs as mel

model_name = "gemini-1.5-flash"  

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


db = Chroma(persist_directory="/chroma/chroma",embedding_function=embeddings)


def search_vectordb(db: object, query: str, k: int) -> list:
    search_kwargs = {"k": k}
    retriever= db.as_retriever(search_kwargs=search_kwargs)
    results = retriever.invoke(query)
    return results

@me.stateclass
class State:
  input: str
  topic: str 
  in_progress: bool
  example_queries: list[str]
  example_query: str
  output: str
  context: list[str]
  debug: str

def on_load(e: me.LoadEvent):
  me.set_theme_mode("system")


@me.page(
  security_policy=me.SecurityPolicy(
   allowed_iframe_parents=["https://google.github.io"]
  ),
  title="ITS A DISASTER!",
  path="/",
  on_load=on_load,
)

def page():
  state = me.state(State)
  with me.box(
    style=me.Style(
      background=me.theme_var("surface-container-lowest"),
      display="flex",
      height="100%",
      width="100%",
    )
  ):
    with me.box(
      style=me.Style(
      background=me.theme_var("surface-container-low"),
      display="flex",
      flex_direction="column",
      height="100%",
      width="100%",
      )
    ):
      chat_pane()
      chat_input()

  
def chat_pane():
  state = me.state(State)
  context = search_vectordb(db,"Disaster Recover, High Availability, Ransomware, Reliability, DR Plan, DR Test, Backup, Restore, DR",15)
  docs = []
  state.context = docs
  for doc in context:
    content= doc.page_content
    docs.append(content)
  system_instructions =  f"""Answer the question with the given context. You are a Solutions Architect specilizaed in Google Cloud.
  If the information is not available in the context say that the Google Cloud Architecture Center provides no guidance but answer the query anyway
  Alwasy give the soure URl! Do not make up information.
  This is the 'context': {docs}
  Example: "What is disaster recovery?" 
  Context: "Service-interrupting events can happen at any time. Your network could have an outage, your latest application push might introduce a critical bug, or you might have to contend with a natural disaster. When things go awry, it's important to have a robust, targeted, and well-tested DR plan."
  Answer: "Diaster recovery is the recovery of your service due to an outage, like a network outage, your application that pushed a critical bug. For more information see https://cloud.google.com/architecture/dr-scenarios-planning-guide
  """
  model = GenerativeModel(model_name,generation_config=GenerationConfig(max_output_tokens=8192, temperature=0.85, top_p=0.95,candidate_count=1),safety_settings=safety_settings,system_instruction=system_instructions)
  chat_session = model.start_chat()
  if state.output:
    full_input = state.output
    chat_session.send_message(full_input,stream=False)
    state.output = ""
  with me.box(
    style=me.Style(
      display = "flex",
      flex_grow=1,
      overflow_y="auto",
      flex_direction= "column", 
      justify_items= "flex-start"
      )
  ):
    if not len(chat_session.history) == 0:
      for message in chat_session.history:
        role = message.role
        if role == "user":
          text = message.text.split('\n\n')
          user_message(text[0])
        else:
          text = message.parts[0].text
          bot_message(str(text))


    #if state.in_progress:
      #with me.box(key="scroll-to", style=me.Style(height=250)):
      #  pass
    
def user_message(text):
  with me.box(
    style=me.Style(
      display="flex",
      gap=15,
      justify_content="start",
      margin=me.Margin.all(20),
    )
  ):
    with me.box(
      style=me.Style(
        background=me.theme_var("surface-container-low"),
        border_radius=10,
        color=me.theme_var("on-surface-variant"),
        padding=me.Padding.symmetric(vertical=0, horizontal=10),
        width="66%",
      )
    ):
      me.markdown(
        text,
        style=me.Style(
            color=me.theme_var("on-surface"),
            text_align= "end"
          ),
        )
    text_avatar(
      background=me.theme_var("secondary"),
      color=me.theme_var("on-secondary"),
      label=USER_AVATAR_LETTER,
    )



def bot_message(text):
  with me.box(style=me.Style(display="flex", gap=15, margin=me.Margin.all(20))):
    text_avatar(
      background=me.theme_var("primary"),
      color=me.theme_var("on-primary"),
      label=BOT_AVATAR_LETTER,
    )

    # Bot message response
    with me.box(style=me.Style(display="flex", flex_direction="column")):
      me.markdown(
        text,
        style=me.Style(color=me.theme_var("on-surface")),
      )

def chat_input():
  state = me.state(State)
  with me.box(
      style=me.Style(
      display= "flex",
       width="100%",
      )
    ):
      with me.box(style=me.Style(flex_grow=1)):
        me.native_textarea(
          autosize=True,
          key="chat_input",
          min_rows=4,
          on_blur=on_chat_input,
          shortcuts={
            me.Shortcut(shift=True, key="Enter"): on_submit_chat_msg,
            },
          placeholder="Enter your prompt",
          style=me.Style(
            width="100%",
            background=me.theme_var("surface-container"),
            border=me.Border.all(
            me.BorderSide(style="none"),
            ),
            color=me.theme_var("on-surface-variant"),
            outline="none",
            overflow_y="auto",
            padding=me.Padding(top=16, left=16),
            ),
          value=state.input
        )        
      with me.content_button(
        disabled=state.in_progress,
        on_click=on_click_submit_chat_msg,
        type="icon",
      ):
          me.icon("send")


@me.component
def text_avatar(*, label: str, background: str, color: str):
  me.text(
    label,
    style=me.Style(
      background=background,
      border_radius="50%",
      color=color,
      font_size=20,
      height=40,
      line_height="1",
      margin=me.Margin(top=16),
      padding=me.Padding(top=10),
      text_align="center",
      width="40px",
    ),
  )


@me.component
def icon_button(
  *,
  icon: str,
  tooltip: str,
  key: str = "",
  is_selected: bool = False,
  on_click: Callable | None = None,
):
  selected_style = me.Style(
    background=me.theme_var("surface-container-low"),
    color=me.theme_var("on-surface-variant"),
  )
  with me.tooltip(message=tooltip):
    with me.content_button(
      type="icon",
      key=key,
      on_click=on_click,
      style=selected_style if is_selected else None,
    ):
      me.icon(icon)

def on_chat_input(e: me.InputBlurEvent):
  """Capture chat text input on blur."""
  state = me.state(State)
  state.input = e.value

def on_submit_chat_msg(e: me.TextareaShortcutEvent):
  state = me.state(State)
  state.input = e.value
  yield
  yield from _submit_chat_msg()

def on_click_submit_chat_msg(e: me.ClickEvent):
  yield from _submit_chat_msg()


def _submit_chat_msg():
  state = me.state(State)
  if state.in_progress or not state.input:
    return
  input = state.input


  
  # Clear the text input.
  state.input = ""
  yield

  start_time = time.time()
  state.output = input
  state.in_progress = False
  me.focus_component(key="chat_input")
  yield