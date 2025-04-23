
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
page_title = "AIrplane"

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
    Tool,
    FunctionCall,
    FunctionDeclaration
)
import vertexai.generative_models as generative_models

import mesop as me
import mesop.labs as mel
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

model_name = "gemini-1.5-pro"  

def better_search(project_id: str, location: str, engine_id: str, search_query: str, instruction: str):
    """
    Search through documents with flight manuals to use as context when answering questions.
    
    
    :param str instruction: the instruction for vertex ai search to find the right information 
    :param str search_query: the user query or question, often the prompt
    :param str engine_id: the type of search, unstructured 0, structured 1  or website 3. 
    :param str location: the id of the vertex ai search AI datatstore, eu or global
    :param str project_id: the project_id of the goolge project
    """
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
            max_extractive_answer_count=5
        ),
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=False
        ),
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=10,
            include_citations=False,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=False,
            model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
            preamble=instruction
        ),
        model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
            version="preview",
            ),
        ),
    )
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.SearchServiceClient(client_options=client_options)
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=100,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )
    response = client.search(request)

    return response

better_search_func = generative_models.FunctionDeclaration.from_func(better_search)

tools = [Tool(
    function_declarations=[better_search_func],
)]


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
generation_config = GenerationConfig(max_output_tokens=8192, temperature=1, top_p=0.95)

system_instructions = """
<PERSONA_AND_GOAL>
      You are a helpful assistant knowledgeable about PanAm airplanes. You can use function calls to reach that goal.
</PERSONA_AND_GOAL>
<FUNCTION_CALL_PARAMETERS>
    :param str search_query: the user query or question, often the prompt. Feel free to improve it
    :param str engine_id: '0'
    :param str location: 'global'
    :param str project_id: {PROJECT_ID}
</FUNCTION_CALL_PARAMETERS>
""".format(PROJECT_ID = PROJECT_ID)


@me.stateclass
class State:
  input: str
  in_progress: bool
  chat_session : any
  output: str
  debug: str


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
    key = "page_main",
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
        me.text(state.debug)


  
def chat_pane():
  state = me.state(State)
  if not state.chat_session:
    model = GenerativeModel(model_name,generation_config=GenerationConfig(max_output_tokens=8192, temperature=0.9, top_p=0.95,candidate_count=1),safety_settings=safety_settings,system_instruction=system_instructions, tools=tools)
    state.chat_session = model.start_chat()


  #if state.output:
  #  answer = chat_session.send_message(state.output,stream=False)
  #  state_debug = answer
  with me.box(
    key= "chat_pane_box",
    style=me.Style(
      display = "flex",
      flex_grow=1,
      overflow_y="auto",
      flex_direction= "column", 
      justify_content= "flex-start",
      )
  ):
    for message in state.chat_session.history:
        role = message.role
        if role == "user":
          user_message(message.text)
        elif role == "model":
          if not message.parts[0].function_call:
            
          #if message.parts[0] == "function_call":
            text = message.parts[0].text
            bot_message(str(text))



    #if state.in_progress:
      #with me.box(key="scroll-to", style=me.Style(height=250)):
      #  pass
    
def user_message(text):
  me.log(message=text)
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
                text_align="end",
            ),

        )
    text_avatar(
      background=me.theme_var("secondary"),
      color=me.theme_var("on-secondary"),
      label=USER_AVATAR_LETTER,
    )



def bot_message(text):
  me.log(message=text)
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

  answer = state.chat_session.send_message(state.input,stream=False)


  state.output = input
  state.in_progress = False
  me.focus_component(key="chat_input")
  yield