import os
from typing import Dict, List

import pandas as pd
import PyPDF2
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_perplexity import ChatPerplexity
from pydantic import SecretStr
from PyPDF2.errors import FileNotDecryptedError


# Step 1: Read all PDF files from the source directory
def read_pdfs_from_directory(directory: str) -> Dict[str, str]:
    """
    Reads all PDF files in the specified directory and returns a dictionary
    mapping filenames to their extracted text content.
    """
    pdf_contents = {}
    for file_name in os.listdir(directory):
        if file_name.lower().endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    document_text = []
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            document_text.append(page_text)
                    pdf_contents[file_name] = '\n'.join(document_text)
            except FileNotDecryptedError:
                print(f'Skipping encrypted PDF: {file_path}')
            except Exception as e:
                print(f'Error reading {file_path}: {e}')
    return pdf_contents


# Step 2: Use LLM to understand the context of each document
def get_document_context(llm, doc_text: str) -> str:
    prompt = f'Understand the document provided below, what information does it provide, what type of information does it provide etc:\n\n{doc_text}'
    # prompt = f'Summarize the main context and purpose of the following document:\n\n{doc_text[:2000]}'
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Step 3: Use LLM to suggest important features to extract
def get_important_features(llm, doc_context: str) -> List[str]:
    prompt = f'Given the following document context, list the most important features (as a Python list of strings) that should be extracted for analysis.\n\nContext: {doc_context}'
    response = llm.invoke([HumanMessage(content=prompt)])
    # Try to parse the response as a Python list
    try:
        features = eval(response.content)
        if isinstance(features, list):
            return [str(f) for f in features]
    except Exception:
        pass
    # Fallback: split by lines
    return [line.strip('- ') for line in response.content.splitlines() if line.strip()]


# Step 4: Extract features from documents and build a DataFrame
def extract_features_from_text(llm, doc_text: str, features: List[str]) -> Dict[str, str]:
    feature_dict = {}
    for feature in features:
        prompt = f"Extract the following feature from the document. If not found, return 'N/A'.\nFeature: {feature}\nDocument: {doc_text[:2000]}"
        response = llm.invoke([HumanMessage(content=prompt)])
        feature_dict[feature] = response.content.strip()
    return feature_dict


def main():
    load_dotenv()
    pplx_api_key = os.getenv('PPLX_API_KEY')
    if not pplx_api_key:
        raise ValueError('PPLX_API_KEY environment variable not set.')

    # Perplexity AI API endpoint and model name
    # llm = ChatPerplexity(
    #     model='sonar',  # Replace with your desired Perplexity model.
    #     api_key=SecretStr(pplx_api_key),  # Set the Perplexity API key.
    #     temperature=0.7,  # Optional: Adjust the temperature for creativity
    #     timeout=60,  # Set a timeout in seconds (adjust as needed)
    # )

    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    source_dir = 'documents'
    pdf_texts = read_pdfs_from_directory(source_dir)
    all_features = set()
    doc_contexts = {}
    doc_features = {}
    # Step 2 & 3: Get context and features for each doc
    for fname, text in pdf_texts.items():
        context = get_document_context(llm, text)
        doc_contexts[fname] = context
        features = get_important_features(llm, context)
        doc_features[fname] = features
        all_features.update(features)
    all_features = list(all_features)
    # Step 4: Extract features for each doc
    rows = []
    for fname, text in pdf_texts.items():
        features = doc_features[fname]
        feature_values = extract_features_from_text(llm, text, features)
        row = {f: feature_values.get(f, 'N/A') for f in all_features}
        row['filename'] = fname
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv('extracted_features.csv', index=False)
    print('Feature extraction complete. Results saved to extracted_features.csv')


if __name__ == '__main__':
    main()
