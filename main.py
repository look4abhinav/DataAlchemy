import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_perplexity import ChatPerplexity
from PyPDF2 import PdfReader
from rich.console import Console
from rich.progress import Progress

console = Console()


# Step 1: Read all PDF files from the source directory using LLMs
def read_pdfs_with_llm(llm, directory: str) -> dict[str, str]:
    console.log(f'[bold blue]Reading PDF files from directory: {directory}[/bold blue]')
    pdf_texts = {}
    with Progress() as progress:
        task = progress.add_task(
            '[bold blue]Reading PDF files...[/bold blue]',
            total=len(os.listdir(directory)),
        )
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                path = os.path.join(directory, filename)
                reader = PdfReader(path)
                pdf_content = ''
                for page in reader.pages:
                    pdf_content += page.extract_text()
                prompt = f"""
                    You are an expert in extracting information from PDF files. Your task is to extract the text content 
                    from the following PDF file. Additionally, if the PDF contains any images, provide a brief description 
                    of each image. If there are no images, do not include an image description in your response. 
                    Ensure that you only share the extracted text and image descriptions, and nothing else.

                    Example 1:
                    Text: 'This is an example text from the PDF document.'
                    Image Description: 'An image of a bar chart showing sales data over time.'

                    Example 2:
                    Text: 'Another example text from a different PDF document.'

                    Now, process the following PDF content:

                    {pdf_content}
                """
                response = llm.invoke([HumanMessage(content=prompt)])
                pdf_texts[filename] = response.content.strip()
            progress.advance(task)
    console.log('[bold blue]Finished reading PDF files.[/bold blue]')
    return pdf_texts


def get_document_context(llm, doc_text: str) -> str:
    console.log('[bold blue]Extracting document context...[/bold blue]')
    prompt = f"""
        You are an expert in analyzing and summarizing documents. Your task is to summarize the main context, purpose, 
        and type of the following document. Additionally, identify the kind of details the document is related to 
        (e.g., financial report, legal document, research paper, etc.). Ensure that your response is concise and 
        provides only the extracted features as a summary.

        Example 1:
        Context: 'This document is a financial report summarizing the quarterly performance of a company, including 
        revenue, expenses, and profit margins.'
        Type: 'Financial Report'
        Details: 'Revenue, Expenses, Profit Margins'

        Example 2:
        Context: 'This document is a research paper discussing the effects of climate change on marine biodiversity, 
        including case studies and statistical analysis.'
        Type: 'Research Paper'
        Details: 'Climate Change, Marine Biodiversity, Case Studies, Statistical Analysis'

        Now, analyze and summarize the following document:

        {doc_text}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def get_important_features(llm, doc_context: str) -> list[str]:
    console.log(
        '[bold blue]Identifying important features from document context...[/bold blue]'
    )
    prompt = f"""
        You are an expert in document analysis and feature extraction. Your task is to identify the most important 
        features to extract from a document based on its context. Focus on understanding the document's type, purpose, 
        and the kind of details it contains. Ensure that your response is concise and provides only the extracted 
        features as a Python list of strings. Do not include any additional text or explanations.

        Example 1:
        Context: 'This document is a financial report summarizing the quarterly performance of a company, including 
        revenue, expenses, and profit margins.'
        Suggested Features: ['Revenue', 'Expenses', 'Profit Margins']

        Example 2:
        Context: 'This document is a research paper discussing the effects of climate change on marine biodiversity, 
        including case studies and statistical analysis.'
        Suggested Features: ['Climate Change', 'Marine Biodiversity', 'Case Studies', 'Statistical Analysis']

        Example 3:
        Context: 'This document is a legal contract outlining the terms and conditions of a business agreement, 
        including parties involved, obligations, and penalties.'
        Suggested Features: ['Parties Involved', 'Obligations', 'Penalties']

        Example 4:
        Context: 'This document is a user manual for a software application, providing instructions on installation, 
        usage, and troubleshooting.'
        Suggested Features: ['Installation Instructions', 'Usage Guidelines', 'Troubleshooting Steps']

        Example 5:
        Context: 'This document is a medical report detailing the diagnosis, treatment plan, and follow-up schedule 
        for a patient.'
        Suggested Features: ['Diagnosis', 'Treatment Plan', 'Follow-Up Schedule']

        Now, based on the following document context, suggest the most important features to extract as a Python list 
        of strings:

        Context: {doc_context}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        features = eval(response.content)
        if isinstance(features, list):
            return [str(f) for f in features]
    except Exception:
        pass
    return [line.strip('- ') for line in response.content.splitlines() if line.strip()]


def extract_features_from_text(
    llm, doc_text: str, features: list[str]
) -> dict[str, str]:
    console.log(
        '[bold blue]Extracting specified features from document text...[/bold blue]'
    )
    feature_dict = {}
    for feature in features:
        prompt = f"""
            You are an expert in extracting specific information from documents. Your task is to extract the value of a given feature 
            from the document. If the feature is not present, return 'N/A'. Ensure that your response is in JSON format, where the 
            feature name is the key and the extracted value is the value. Do not include any additional text or explanations.

            Example 1:
            Feature: 'Revenue'
            Document: 'The company reported a revenue of $5 million in the last quarter.'
            Response: {{ "Revenue": "$5 million" }}

            Example 2:
            Feature: 'Climate Change'
            Document: 'This research paper discusses the impact of climate change on marine biodiversity.'
            Response: {{ "Climate Change": "Impact on marine biodiversity" }}

            Example 3:
            Feature: 'Parties Involved'
            Document: 'This contract is between Company A and Company B for the supply of raw materials.'
            Response: {{ "Parties Involved": "Company A, Company B" }}

            Now, extract the following feature from the document. If not found, return 'N/A'.
            Feature: {feature}
            Document: {doc_text}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            extracted_data = eval(response.content)
            if isinstance(extracted_data, dict) and feature in extracted_data:
                feature_dict[feature] = extracted_data[feature]
            else:
                feature_dict[feature] = 'N/A'
        except Exception:
            feature_dict[feature] = 'N/A'
    console.log('[bold green]Feature extraction completed successfully.[/bold green]')
    return feature_dict


# --- Main Pipeline ---
def main():
    """Main pipeline for processing PDFs and extracting features."""

    # Step 1: Load environment variables
    console.log('[bold blue]Loading environment variables...[/bold blue]')
    load_dotenv()
    pplx_api_key = os.getenv('PPLX_API_KEY')
    if not pplx_api_key:
        console.log(
            '[bold red]Error: PPLX_API_KEY environment variable not set.[/bold red]'
        )
        raise ValueError('PPLX_API_KEY environment variable not set.')
    console.log('[bold green]Environment variables loaded successfully.[/bold green]')

    # Step 2: Initialize the LLM
    console.log('[bold blue]Initializing the LLM...[/bold blue]')
    llm = ChatPerplexity(
        model='sonar',  # Replace with your desired Perplexity model.
        temperature=0.9,  # Optional: Adjust the temperature for creativity
        timeout=60,  # Set a timeout in seconds (adjust as needed)
    )
    console.log('[bold green]LLM initialized successfully.[/bold green]')

    # Step 3: Read PDFs from the source directory
    source_directory = 'documents'  # Update with your directory path
    pdf_texts = read_pdfs_with_llm(llm, source_directory)

    # Step 4: Process each document to extract context and features
    console.log(
        '[bold blue]Processing documents to extract context and features...[/bold blue]'
    )
    doc_contexts = {}
    doc_features = {}
    all_features = set()

    with Progress() as progress:
        task = progress.add_task(
            '[bold blue]Processing documents...', total=len(pdf_texts)
        )

        for feature_name, text in pdf_texts.items():
            # Step 1: Extract document context
            progress.console.log(
                '[bold blue]Step 1: Extracting document context...[/bold blue]'
            )
            doc_contexts[feature_name] = get_document_context(llm, text)

            # Step 2: Extract important features based on context
            progress.console.log(
                '[bold blue]Step 2: Extracting important features based on context...[/bold blue]'
            )
            features = get_important_features(llm, text)
            doc_features[feature_name] = features
            all_features.update(features)

            progress.advance(task)

    # Convert all_features to a sorted list for consistent ordering

    # Step _: Combine similar features
    # all_features = combine_similar_features(llm, all_features)

    # Step 5: Extract feature values and build a DataFrame
    console.log(
        '[bold blue]Extracting feature values and building a DataFrame...[/bold blue]'
    )
    rows = []
    with Progress() as progress:
        task = progress.add_task(
            '[bold blue]Extracting features...', total=len(pdf_texts)
        )
        for feature_name, text in pdf_texts.items():
            feature_values = extract_features_from_text(
                llm, text, doc_features[feature_name]
            )
            row = {
                feature: feature_values.get(feature, 'N/A') for feature in all_features
            }
            row['filename'] = feature_name
            rows.append(row)
            progress.advance(task)

    df = pd.DataFrame(rows)
    # Remove columns that are completely empty
    df = df.dropna(axis=1, how='all')

    # Step 6: Save results to a CSV file
    output_file = 'extracted_features.csv'
    df.to_csv(output_file, index=False)
    console.log(
        f'[bold green]Feature extraction complete. Results saved to {output_file}[/bold green]'
    )


if __name__ == '__main__':
    main()
