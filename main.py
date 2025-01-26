
BOT_AVATAR_LETTER = "A"
USER_AVATAR_LETTER= "M"
CHAT_MAX_WIDTH = "800px"
PROJECT_ID = "development-411716"
LOCATION = "europe-west4" 
REGION = LOCATION
EMPTY_CHAT_MESSAGE = "Please select a Health Topic and Medical Role"
# Set to your data store location
VERTEX_AI_SEARCH_LOCATION = "eu"  # @param {type:"string"}
# Set to your search app ID
VERTEX_AI_SEARCH_APP_ID = "freebird_1737574726005"  # @param {type:"string"}
# Set to your data store ID
VERTEX_AI_SEARCH_DATASTORE_ID = "freebird_1737574778860_gcs_store"  # @param {type:"string"}

import requests

project_id= requests.get("http://metadata/computeMetadata/v1/project/project-id", headers={'Metadata-Flavor': 'Google'}).text
full_zone_string = requests.get("http://metadata/computeMetadata/v1/instance/zone", headers={'Metadata-Flavor': 'Google'}).text
zone_name = full_zone_string.split("/")[3]
region = zone_name[:-2]

REGION = region
PROJECT_ID = project_id
location = 'us-central1'

import requests
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Literal
import base64

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    SafetySetting,
    Part,
    Tool
)  

import mesop as me
import mesop.labs as mel

model_name = "gemini-1.5-flash"  

tools = [
    Tool.from_retrieval(
        retrieval=generative_models.grounding.Retrieval(
            source=generative_models.grounding.VertexAISearch(
                datastore=VERTEX_AI_SEARCH_DATASTORE_ID,
                project=PROJECT_ID,
                location=VERTEX_AI_SEARCH_LOCATION,
            ),
            disable_attribution=False,
        )
    ),
]


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

vertexai.init(project=PROJECT_ID, location=LOCATION)
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
generation_config = GenerationConfig(max_output_tokens=8192, temperature=0.8, top_p=0.95)

system_instructions = """
<PERSONA_AND_GOAL>
      You are a helpful assistant knowledgeable about PanAm airplanes
</PERSONA_AND_GOAL>

<INSTRUCTIONS>
- Always refer to the tool and Ground your answers in it
- Understand the retrieved snippet by the tool and only use that information to help users
- For supporting references, you can provide the Grounding tool snippets verbatim, and any other info like page number
- For Information not available in the tool, mention you dont have access to the information.
- Output "answer" should be I dont know when the user question is irrelevant or outside the <CONTEXT>
- Leave "reference_snippet" as null if you are not sure about the page and text snippet
<INSTRUCTIONS>

<CONTEXT>
  Grounding tool finds most relevant snippets from the Pan An operatin manuals data store.
  Use the information provided by the tool as your knowledge base.
</CONTEXT>

<CONSTRAINTS>
- ONLY use information available from the Grounding tool
</CONSTRAINTS>
"""


@me.stateclass
class State:
  input: str
  in_progress: bool
  session : bool

def on_load(e: me.LoadEvent):
  me.set_theme_mode("dark")

@me.page(
  security_policy=me.SecurityPolicy(
   allowed_iframe_parents=["https://google.github.io"]
  ),
  title=page_title,
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
        chat_pane()
        chat_input()
      with me.box(
        style=me.Style(
        background=me.theme_var("surface-container-low"),
        display="flex",
        flex_direction="column",
        height="100%",
        width="100%",
        )
      ):
        me.text("Placeholder")


  
def chat_pane():
  state = me.state(State)
  model = GenerativeModel(model_name,generation_config=GenerationConfig(max_output_tokens=8192, temperature=1, top_p=0.95,candidate_count=1),safety_settings=safety_settings,system_instruction=system_instructions, tools=[tools])
  chat_session = model.start_chat()
  if state.input:
    chat_session.send_message(state.output,stream=False)
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


# Event Handlers
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
  state.in_progress = False
  me.focus_component(key="chat_input")
  yield