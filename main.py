
BOT_AVATAR_LETTER = "A"
USER_AVATAR_LETTER= "M"
CHAT_MAX_WIDTH = "800px"
MEDICAL_ROLE = ("Nurse","Doctor")
PROJECT_ID = "development-411716"
LOCATION = "europe-west4" 
REGION = LOCATION
MODEL = "gemini-1.5-flash"
EMPTY_CHAT_MESSAGE = "Please select a Health Topic and Medical Role"

import requests
import os
#from dataclasses import asdict, dataclass
#from typing import Callable, Literal
#import base64

#from bs4 import BeautifulSoup, SoupStrainer

from langchain_google_vertexai import VertexAIEmbeddings
#from langchain_community.document_loaders import WebBaseLoader
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

model_name = "gemini-2.0-flash"  

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

path = "./chroma_db"

def load_embeddings(path):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    if os.path.exists(path):
        db = Chroma(persist_directory=path,embedding_function=embeddings)
    return db

def search_vectordb(db: object, query: str, k: int) -> list:
    search_kwargs = {"k": k}
    retriever= db.as_retriever(search_kwargs=search_kwargs)
    results = retriever.invoke(query)
    return results

def simple_generate(prompt: str, candidate_count: int = 1):
    model = GenerativeModel(model_name)
    generation_config = GenerationConfig(max_output_tokens=8192, temperature=0.8, top_p=0.95,candidate_count=candidate_count)
    responses = model.generate_content(
      [prompt],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=False,
    )
    return responses.candidates

db = load_embeddings(path)

@me.stateclass
class State:
  input: str
  topic: str 
  medical_role: str 
  in_progress: bool
  topic_context_list: list[str]
  example_queries: list[str]
  example_query: str
  session : bool
  output: str
  context: str
  topic_html: str

def on_load(e: me.LoadEvent):
  me.set_theme_mode("dark")


@me.page(
  security_policy=me.SecurityPolicy(
   allowed_iframe_parents=["https://google.github.io"]
  ),
  title="WHO Assistant",
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
      me.text(EMPTY_CHAT_MESSAGE)
      topic_selector_box()
      role_selector_box()
      if state.topic and state.medical_role:
        example_selector_box()
        overview_box()
    if state.example_query:
      with me.box(
        style=me.Style(
        background=me.theme_var("surface-container-low"),
        display="flex",
        flex_direction="column",
        height="100%",
        width="100%",
        justify_items= "end"
        )
      ):
        if state.example_query:
          chat_pane()
          chat_input()
      with me.box(
        style=me.Style(
        background=me.theme_var("surface-container-low"),
        display="flex",
        flex_direction="row",
        height="100%",
        width="30%",
        )
      ):
        me.text("Placeholder")



def topic_selector_box():
  state = me.state(State)
  options = []
  topics = []
  for item in db.get()["metadatas"]:
    if item["type"] == "topic":
      topic_title = item["title"].strip()
      topics.append(topic_title)
      option = me.SelectOption(label=topic_title, value=topic_title)
      options.append(option)
  me.select(
    label="Health Topic",
    options=options,
    on_selection_change=on_selection_change_topic,
    style=me.Style(
      display= "flex",
      flex_basis= "auto",
      width="100%"
    ),
    multiple=False,
    appearance="fill",
    value=state.topic,
    )                                                                                       
  
def role_selector_box():
  state = me.state(State)
  me.select(
    label="Role",
    options = [me.SelectOption(label=MEDICAL_ROLE[0], value=MEDICAL_ROLE[0]),
            me.SelectOption(label=MEDICAL_ROLE[1], value=MEDICAL_ROLE[1])],
    on_selection_change=on_selection_change_role,
    style=me.Style(
      display= "flex",
      flex_basis= "auto",
      width="100%"
    ),
      multiple=False,
      appearance="outline",
      value=state.medical_role,
    )
    
def example_selector_box():
  state = me.state(State)
  options = []
  queries = simple_generate(f"create short one sentence LLM query on {state.topic} using {state.topic_context_list} specificially for someone with {state.medical_role}",3)
  for query in queries:
    option = me.SelectOption(label=query.text.strip(), value=query.text.strip())
    options.append(option)
  me.select(
    label="Example Queries",
    options=options,
    on_selection_change=on_selection_change_example,
    style=me.Style(
    display= "flex",
    flex_basis= "auto",
    width="100%"
    ),
    multiple=False,
    appearance="outline",
    value=state.example_query,
    )
  
def overview_box():
  state = me.state(State)
  me.html(
    html = state.topic_html,
    style=me.Style(
      display= "flex",
      flex_basis= "auto",
      width="100%",
      overflow_y="auto",
    ),
    mode="sanitized"
  )
  
def chat_pane():
  state = me.state(State)
  system_instructions = """
<PERSONA_AND_GOAL>
    -You are a helpful assistant knowledgeable about healthcare and specifically about the documentation as provided by the World Health Organization as Health Topics
    -You are also able to translate from and to all the languages known to you. 
    -You do not make up any information
</PERSONA_AND_GOAL>

<INSTRUCTIONS>
    - The prompt will have a question about {topic} by a {role}
    - Use the the <CONTEXT> to answer the question and if you cannot say "There is no such information withing the WHO Health Topics"
    - Always mention the source of the article as a full url but mention each url only once
    - Use the chat history to determine if the user wants information on a sub-topic. 
<INSTRUCTIONS>

<CONTEXT>
{topic_context}
</CONTEXT>

<CONSTRAINTS>
    - You cannot make up information
    - You cannot answer non healthcare related questions
</CONSTRAINTS>

<OUTPUT_FORMAT>
 - If the prompt is another language answer in that language
 - If the user asks for a translation, give the translation
/OUTPUT_FORMAT>
  """.format(topic= state.topic,role= state.medical_role,topic_context= str(state.topic_context_list))
  model = GenerativeModel(model_name,generation_config=GenerationConfig(max_output_tokens=8192, temperature=1, top_p=0.95,candidate_count=1),safety_settings=safety_settings,system_instruction=system_instructions)
  chat_session = model.start_chat()
  # function 
  if state.output:
    chat_session.send_message(state.output,stream=False)
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
      justify_content="end",
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


# Event Handlers
def on_selection_change_topic(e: me.SelectSelectionChangeEvent):
  state = me.state(State)
  state.topic = e.value
  topic_context_list = []
  topic_context =  search_vectordb(db,state.topic,5)
  for result in topic_context:
    if str(result.metadata["title"]).strip() == state.topic and result.metadata["type"] == "topic":
     state.topic_html = result.metadata["html"]
    topic_context_list.append(result.page_content)
  state.topic_context_list =  topic_context_list
  state.example_query = ""

def on_selection_change_role(e: me.SelectSelectionChangeEvent):
  state = me.state(State)
  state.medical_role = e.value
  medical_role =  state.medical_role
  state.example_query = ""

def on_selection_change_example(e: me.SelectSelectionChangeEvent):
  state = me.state(State)
  state.example_query  = e.value
  state.input = state.example_query
  me.focus_component(key="chat_input")

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
  context = search_vectordb(db,input,1)
  state.context = context[0].page_content
  #full_input = f"{input}\n\n\{context[0].page_content}"
  full_input = f"{input}"
  output= full_input
  state.output = output
  state.in_progress = False
  me.focus_component(key="chat_input")
  yield
