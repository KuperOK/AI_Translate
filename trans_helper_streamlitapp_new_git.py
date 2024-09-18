import streamlit as st
import openai
import io
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from stqdm import stqdm
import re


def check_openai_api_key(api_key):
    
    """
    Validates the provided OpenAI API key by attempting to list the available models.

    Args:
        api_key (str): The OpenAI API key to be validated.

    Returns:
        bool: True if the API key is valid and the API request succeeds, False otherwise.
    """
    
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

def process_text_file(file, num_parts):
    """
    Splits the content of a text file into a specified number of approximately equal parts.
    Args:
        file (io.TextIOWrapper): A file-like object containing text data to be processed.
        num_parts (int): The number of parts to split the text into.
    Returns:
        List[str]: A list of strings, each representing a part of the original text file.    
    Notes:
        - The function reads the content of the file, splits it into roughly equal parts based on the specified number of parts.
        - The text is split at line breaks, and any leftover lines are distributed among the parts.
        - Each part is returned as a single string within a list.
    """

    text = file.read().decode("utf-8")        
    texts_list = text.splitlines()
    avg_length = len(texts_list) // num_parts
    remainder = len(texts_list) % num_parts    
    result = []
    start = 0    
    for i in range(num_parts):
        end = start + avg_length + (1 if i < remainder else 0)
        result.append(texts_list[start:end])
        start = end
    splited_file_to_list_of_strings = ['\n'.join(part) for part in result]    
    return splited_file_to_list_of_strings
    
def get_gpt_response(chat, split_texts, rules):
    """
    Processes a list of text segments through a GPT model and concatenates the responses.

    This function iterates through each text segment in `split_texts`, formats the segment into a prompt 
    according to predefined rules, sends it to the provided GPT model instance via the `chat` object, 
    and collects the model's responses. The responses are concatenated into a single string with each 
    response separated by a newline. Additionally, the time taken for the translation process is displayed.

    Args:
        chat (object): An instance of a chat interface with the GPT model. It should have an `invoke` method 
                       that takes formatted messages and returns a response object with a `content` attribute.
        split_texts (list of str): A list where each element is a text segment to be processed by the GPT model.

    Returns:
        str: A single string containing all translated text segments concatenated, with each segment's response 
              separated by a newline.

    Example:
        >>> chat = SomeChatInterface()
        >>> texts = ["Hello, how are you?", "I am fine, thank you!"]
        >>> translated_text = get_gpt_response(chat, texts)
        >>> print(translated_text)
        "Hi, how are you?\nI'm good, thanks!"
    """    
    translated = ''
    s = datetime.now()
    for part in stqdm(split_texts):        
        customer_text = f"{part}"
        customer_messages = prompt_template.format_messages(                    
                    rules=rules,
                    text=customer_text)
        response = (chat.invoke(customer_messages)).content        
        response = response.replace("```\n", "")
        response = response.replace("\n```", "")
        translated += response + '\n'         
    e = datetime.now()
    st.write(f'Translation time: {e-s}')
    return translated
    

template_string = """You experienced translator. \
I want you to act as a translator and processor of text fragments of a special structure.
Please translate from German to language which specified in parenthesis in each line
of the text that is delimited by triple backticks. Follow these rules:\n {rules}. \
\ntext: \n```
{text}
```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

def main():
    
    customer_text = """
    1. **Read** the text fragment that I will give you.
    2. **Find** in each line of text the language code in brackets (`(fr)` - French, `(it)` - Italian).
    3. **Translate** the text written in German after the language code  `(fr)` or  `(it)` to the appropriate language (`(fr)` - French, `(it)` - Italian).
    4. **Try** translate every word, don't miss any word which that should be translated from Gуrman to Franch `(fr)` or Italian `(it)`
    5. **Save** all the text before language code and language code without changes.
    6. **Save** formatting and special characters (tags, hyperlinks) in the translated text.
    7. **Return** the processed text.
    **Example 1:**
    **Input text:** `view.elements.card_n50.press_review.header_plural=(fr){0} von {1} Pressestimmen / Test-Siege`
    **Output text:** `view.elements.card_n50.press_review.header_plural=(fr){0} sur {1} avis de presse / Victoires aux tests`
    **Example 2:**
    ** Input text:** `view.elements.footer.note.11=(it) <sup>**</sup> Diese Meinung entstammt unserer Kundenbefragung, die wir seit 2010 continously als Instrument für Qualitäts-Management und Produktverbesserung überführen . Wir befragen hierzu all Direktkunden 21 Tage nach Kauf per E-Mail zu deren Satfriedenheit, Erfahrungen und Verbesserungsvorschlägen mit der Lieferung sowie den orderedten Produkten.`
    ** Output text:** `view.elements.footer.note.11=(it) <sup>**</sup> Questo parere proviene dal nostro sondaggio clienti, che conduciamo continuamente dal 2010 come strumento per la gestione della qualità e il miglioramento dei prodotti. Intervistiamo tutti i clienti diretti 21 giorni dopo l'acquisto via e-mail sulla loro soddisfazione, esperienze e suggerimenti di miglioramento relativi alla consegna e ai prodotti ordinati.`
    
    **Expecting:**
    * Accurate official translation according to the context.
    * Preserving formatting and special characters.
    * Answer in the form of a processed text fragment without adding any comments of your own.
    **Now give me the processed text using the input snippet I'll give you.**"""

    st.title("Special Translate AI Helper")
    

    llm_model = st.selectbox(
                "Select LLM",
                ("gpt-4o-mini-2024-07-18", "chatgpt-4o-latest")
        )
    num_parts = st.slider('Num parts to split file', min_value=1, max_value=20, value=7, step=1)

    st.session_state.customer_rules = st.text_area("Enter prompt (main rules) for translation", height=550, value=customer_text)
    
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    openai.api_key = os.getenv('OPENAI_API_KEY')        
    
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
            
    uploaded_file = st.file_uploader("Upload a text file", type="txt")

    if uploaded_file and st.button("Process File") and st.session_state.customer_rules:
        print('###')
        print(st.session_state.customer_rules)

        st.session_state.translated_text = None
        
        st.write('File successfully uploaded')
        
        with st.spinner(text="Splitting in progress..."):
            processed_text = process_text_file(uploaded_file, num_parts=num_parts)
        st.success("Splitting done!")
                        
        chat = ChatOpenAI(model_name=llm_model, temperature=0)
        
        if st.session_state.translated_text is None:
            with st.spinner(text="AI in progress ..."):
                st.session_state.translated_text = get_gpt_response(chat, processed_text, rules=st.session_state.customer_rules)
                
            buffer = io.BytesIO()
            buffer.write(st.session_state.translated_text.encode("utf-8"))
            buffer.seek(0)

            pattern = r'\((fr|it)\)'
            match = re.search(pattern, st.session_state.translated_text)
            lang = match.group(1)

            filename = lang + '_output_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'    
            st.download_button(
                label="Download File",
                data=buffer,
                file_name=filename,
                mime="text/plain"
            )



    # if st.session_state.api_key and customer_rules:
    #     print(customer_rules)        
    #     if uploaded_file:
    #         st.write('File Seccessfully upload')
            
    #         if st.session_state.translated_text is not None:
    #             st.session_state.translated_text = None


    #         with st.spinner(text="Splitting in progress..."):
    #             processed_text = process_text_file(uploaded_file, num_parts=num_parts)
    #         st.success("Splitting done!")
                            
    #         chat = ChatOpenAI(model_name=llm_model, temperature=0)
    #         if st.session_state.translated_text is None:
    #             with st.spinner(text="AI in progress ..."):
    #                 st.session_state.translated_text = get_gpt_response(chat, processed_text, rules=customer_rules)
    #                 # print(st.session_state.translated_text[:1000])
    #                 # print()
    #                 # print(st.session_state.translated_text[-1000:])
          
    #             buffer = io.BytesIO()
    #             buffer.write(st.session_state.translated_text.encode("utf-8"))
    #             buffer.seek(0)

    #             pattern = r'\((fr|it)\)'
    #             match = re.search(pattern, st.session_state.translated_text)
    #             lang = match.group(1)

    #             filename = lang + '_output_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'    
    #             st.download_button(
    #                         label="Download File",
    #                         data=buffer,
    #                         file_name=filename,
    #                         mime="text/plain"
    #                     )

    #     # else:
    #     #     st.error("Invalid API key. Please try again.")
    #     #     st.session_state.api_key = ""
    #     #     return

if __name__ == "__main__":
    main()
