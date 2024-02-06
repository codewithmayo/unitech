import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
import numpy as np
import openai
import requests
from flask import Flask, render_template, request, jsonify
from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOADED_FILES_FILE'] = 'uploaded_files.json'

openai.api_key = os.getenv("OPENAI_API_KEY")

# Placeholder data storage for uploaded files
if os.path.exists(app.config['UPLOADED_FILES_FILE']):
    with open(app.config['UPLOADED_FILES_FILE'], 'r') as file:
        uploaded_files = json.load(file)
else:
    uploaded_files = []

print(f"The uploaded files are: {uploaded_files}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form.get('message')
    print(user_message)
    bot_response = get_answer(user_message)

    return jsonify({'user_message': user_message, 'bot_response': bot_response})



@app.route('/see_docs')
def see_docs():
    if os.path.exists(app.config['UPLOADED_FILES_FILE']):
        with open(app.config['UPLOADED_FILES_FILE'], 'r') as file:
            uploaded_files = json.load(file)
        return jsonify({'uploaded_files': uploaded_files})
    else:
        return jsonify({'uploaded_files': []})



@app.route('/delete_file/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.remove(file_path)

        # Update the list of uploaded files
        if filename in uploaded_files:
            uploaded_files.remove(filename)
            # Save updated uploaded files list to file
            with open(app.config['UPLOADED_FILES_FILE'], 'w') as file:
                json.dump(uploaded_files, file)

        # Update the knowledge base
        knowledge_base_path = 'knowledge_base.json'
        with open(knowledge_base_path, 'r', encoding='utf-8') as file:
            knowledge_base = json.load(file)

        # Remove entries related to the deleted file
        updated_knowledge_base = [entry for entry in knowledge_base if entry['pdf_name'] != filename]

        # Save the updated knowledge base back to the JSON file
        with open(knowledge_base_path, 'w', encoding='utf-8') as file:
            json.dump(updated_knowledge_base, file, indent=4, ensure_ascii=False)

        return jsonify({'success': True, 'message': f'File {filename} deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})






@app.route('/api_key', methods=['POST'])
def api_key():
    api_key = request.form.get('api_key')

    return jsonify({'success': True, 'message': 'API key saved successfully'})


def get_context(inputPrompt, top_k):
    search_term_vector = get_embedding(inputPrompt, engine='text-embedding-ada-002')

    with open("knowledge_base.json", encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for item in data:
            item['embeddings'] = np.array(item['embeddings'])

        for item in data:
            item['similarities'] = cosine_similarity(item['embeddings'], search_term_vector)

        sorted_data = sorted(data, key=lambda x: x['similarities'], reverse=True)
        context = ''
        for i in sorted_data[:top_k]:
            context += i['chunk'] + '\n'
    return context



def get_answer(user_input):


    context = get_context(user_input,3)

    prompt = "context:\n\n{}.\n\n Answer the following user query according to above given context:\nuser_input: {}".format(context,user_input)

    myMessages = []
    myMessages.append({"role": "system", "content": "You are UNITECH Chat bot. You are a helpful assistant who can also answer general questions, provide, rewrite and fix things like emails and other similar stuff. You cannot perform physical tasks"})
    

    myMessages.append({"role": "user", "content": "context:\n\n{}.\n\n Answer the following user query according to above given context:\nuser_input: {}".format(context,user_input)})

    
    response = openai.ChatCompletion.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        messages=myMessages,
        max_tokens=None,
        stream=False
    )
    
    return response['choices'][0]['message']['content']


def process_chunk(chunk_text):
    embd = get_embedding(chunk_text, engine='text-embedding-ada-002')
    return embd


def generate_json_with_embeddings(data):
    with ThreadPoolExecutor() as executor:
        futures = []
        n = 1
        for ind, i in enumerate(data):
            future = executor.submit(process_chunk, i["chunk"])
            futures.append((ind, future))
            n += 1
            if n >= 100:
                print("embedding 100 done.", flush=True)
                n = 1
                time.sleep(14)
        for ind, future in futures:
            data[ind]["embeddings"] = future.result()
    return data


def extract_pdf_content(pdf_path):
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_name = os.path.basename(pdf_path)
    content_chunks = []

    for page_num, page in enumerate(pdf_reader.pages, 1):
        content = page.extract_text()

        _content = content.split('\n')
        half_page = len(_content) // 2
        chunk = ''
        for i in range(half_page):
            chunk += _content[i] + '\n'
        page_chunk = {
            "chunk_id": str(uuid.uuid4()),
            "chunk": chunk,
            "page_num": page_num,
            "pdf_name": pdf_name,
        }
        content_chunks.append(page_chunk)

        chunk = ''
        for j in range(half_page, len(_content)):
            chunk += _content[j] + '\n'
        page_chunk = {
            "chunk_id": str(uuid.uuid4()),
            "chunk": chunk,
            "page_num": page_num,
            "pdf_name": pdf_name
        }
        content_chunks.append(page_chunk)

    pdf_file.close()

    try:
        pdf_file.close()
    except:
        pass
    return content_chunks




@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded file
        # Update the list of uploaded files
        if file.filename not in uploaded_files:
            uploaded_files.append(file.filename)
            # Save updated uploaded files list to file
            with open(app.config['UPLOADED_FILES_FILE'], 'w') as file:
                json.dump(uploaded_files, file)
       
        data = extract_pdf_content(file_path)
        data = generate_json_with_embeddings(data)

        # Update knowledge base
        knowledge_base = 'knowledge_base.json'
        with open(knowledge_base, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)

        new_data = existing_data + data

        with open(knowledge_base, 'w', encoding='utf-8') as file:
            json.dump(new_data, file, indent=4, ensure_ascii=False)

        

        return 'File uploaded and processed successfully'


if __name__ == '__main__':
    app.run(debug=True)





