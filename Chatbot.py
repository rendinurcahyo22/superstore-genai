import streamlit as st
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair, GroundingSource, ChatMessage, ChatSession
from google.cloud import bigquery
from datetime import datetime
import json

# Set the background colors
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Light gray background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
    }
    .st-bw {
        background-color: #eeeeee; /* White background for widgets */
    }
    .st-cq {
        background-color: #cccccc; /* Gray background for chat input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
    }

    .st-cx {
        background-color: white; /* White background for chat messages */
    }
    .sidebar .block-container {
        background-color: #f0f0f0; /* Light gray background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    .top-right-image-container {
        position: fixed;
        top: 30px;
        right: 0;
        padding: 20px;
        background-color: white; /* White background for image container */
        border-radius: 0 0 0 10px; /* Add rounded corners to bottom left */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("👨‍💻 Metrodata-KUP Bot")

# # Top right corner image container
# st.markdown(
#     "<div class='top-right-image-container'>"
#     "<img src='https://imgur.com/sxSdMX2.png' width='60'>"
#     "</div>",
#     unsafe_allow_html=True
# )

# Create functions to open each social media app
def open_app(app_name):
    st.experimental_set_query_params(page=app_name)

##################################################################################################
# Function LLM GenAI Studio
def initialize_chat_bot():
    vertexai.init(project="trial-genai", location="us-central1")
    chat_model = ChatModel.from_pretrained("chat-bison")
    chat = chat_model.start_chat(
      context="""
      Kamu adalah Virtual Assistant yang berdedikasi untuk memberikan informasi terkait penjualan Superstore. Jawablah pertanyaan sesuai bahasa penanya.
      Apabila konteks yang didapatkan berbeda bahasa, translate menjadi seperti bahasa penanya.
      
      Konteks knowledge kamu adalah data dari Bigquery dengan skema sebagai berikut:
      - Row ID => Unique ID for each row.
      - Order ID => Unique Order ID for each Customer.
      - Order Date => Order Date of the product.
      - Ship Date => Shipping Date of the Product.
      - Ship Mode=> Shipping Mode specified by the Customer.
      - Customer ID => Unique ID to identify each Customer.
      - Customer Name => Name of the Customer.
      - Segment => The segment where the Customer belongs.
      - Country => Country of residence of the Customer.
      - City => City of residence of of the Customer.
      - State => State of residence of the Customer.
      - Postal Code => Postal Code of every Customer.
      - Region => Region where the Customer belong.
      - Product ID => Unique ID of the Product.
      - Category => Category of the product ordered.
      - Sub-Category => Sub-Category of the product ordered.
      - Product Name => Name of the Product
      - Sales => Sales of the Product.
      - Quantity => Quantity of the Product.
      - Discount => Discount provided.
      - Profit => Profit/Loss incurred.
    
      Misi yang kamu jalani:
      - Membantu mereka dengan memberikan informasi berdasarkan konteks yang diberikan dan terdapat dalam basis pengetahuan kamu. Basis pengetahuan Anda adalah satu-satunya sumber informasi.
      - Tidak menjawab di luar konteks. Katakan kamu tidak bisa menjawabnya dan arahkan penanya untuk bertanya yang berhubungan dengan kebijakan di Metrodata.
      - Buatlah jawaban berupa summary dari konteks yang kamu dapatkan.
      
      Catatan penting:
      - Ingat untuk tidak menjawab diluar konteks.
      - Pastikan kamu menjawab dengan bahasa yang sama dengan bahasa penanya.
      - Ingat sebelum mengirim jawaban, selalu cek kembali apakah jawaban sudah dapat menjawab pertanyaan sesuai dengan misi yang diberikan.
      """,
        max_output_tokens = 1024,
        temperature = 0,
        top_p = 1,
        top_k = 40
    )
    return chat

# Initialize model parameters
def initialize_parameter_bot(grounding_use_web = False):
    grounding_value = ''
    if grounding_use_web:
        grounding_value = GroundingSource.WebSearch()
    else:
        grounding_value = GroundingSource.VertexAISearch(data_store_id="superstore_1697388477640", location="global", project="trial-genai")

    parameters = {
      "candidate_count": 1,
      "grounding_source": grounding_value,
      "max_output_tokens": 1024,
      "temperature": 0,
      "top_p": 1
      }

    return parameters
chat = initialize_chat_bot()
parameters = initialize_parameter_bot()
##################################################################################################

# Initialize the session_state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Halo, aku Virtual Assistant yang akan membantumu memberikan informasi tentang penjualan di Superstore'"}]

# Display existing chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner('Preparing'):
        response = chat.send_message("""{}""".format(prompt), **parameters)

    #st.write(msg)

    st.session_state.messages.append({"role": "assistant", "content": response.text})
    st.chat_message("assistant").write(response.text)
