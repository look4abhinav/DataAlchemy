import io
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pdf2image import convert_from_path
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console

import prompts.model_prompts as mp


def setup_agent() -> Agent:
    """Sets up the model to be used and returns a Pydantic Agent

    Returns:
        Agent: Pydantic Agent
    """

    load_dotenv()

    model = OpenAIChatModel(
        model_name="sonar",
        provider=OpenAIProvider(base_url="https://api.perplexity.ai", api_key=os.getenv("PPLX_API_KEY", "")),
        settings={"temperature": 0.1},
    )

    agent = Agent(model, system_prompt=mp.SYSTEM_PROMPT)
    return agent


def read_documents(file_path: Path) -> dict:
    documents: dict = {}

    if not file_path.exists():
        console.log(f"[bold red] Source directory {file_path.resolve()} is mising.")

    files = sorted(file_path.glob("*.pdf"))

    for file in files:
        file_contents = []

        for page in convert_from_path(file, dpi=300):
            image_bytes = io.BytesIO()
            page.save(image_bytes, format="PNG")

            image = BinaryContent(image_bytes.getvalue(), media_type="image/png")
            file_contents.append(image)

        documents[file.name] = file_contents

    return documents


if __name__ == "__main__":
    console = Console()
    agent = setup_agent()
    test_files_path = Path.cwd() / "artifacts" / "test_docs"
    documents = read_documents(test_files_path)

    features = []
    data = []
    output_file_path = Path.cwd() / "artifacts" / "extracted_features.csv"

    for i, (document, content) in enumerate(documents.items(), start=1):
        console.log(f"[bold blue]Processing [bold green]{i}/{len(documents)} [{document}][bold green][/bold blue]")
        console.log("[bold cyan]\tExtracting text from PDF...[/bold cyan]")
        pdf_text = agent.run_sync([mp.EXTRACT_TEXT_FROM_PDF, *content])

        # console.log("[bold cyan]\tExtracting document context...[/bold cyan]")
        # document_context = agent.run_sync(f"{mp.IDENTIFY_DOCUMENT_CONTEXT}\n {pdf_text}")

        console.log("[bold cyan]\tExtracting important features...[/bold cyan]")
        important_features = agent.run_sync(f"{mp.INDENTIFY_IMPORTANT_FEATURES}\n {pdf_text}")

        console.log("[bold cyan]\tCombining with existing features...[/bold cyan]")
        combined_features = agent.run_sync(f"{mp.COMBINE_FEATURES}\n{features} \n{important_features} ")
        features.extend(list(feature.strip() for feature in combined_features.output.split(",")))

        console.log("[bold cyan]\tExtracting features from text...[/bold cyan]")
        extracted_features = agent.run_sync(f"{mp.EXTRACT_FEATURES}\n FEATURE: {features} \nTEXT: {pdf_text} ")
        extracted_features = json.loads(extracted_features.output.replace("```", "").replace("json", ""))
        extracted_features["Source file"] = document

        data.append(extracted_features)
        console.clear()

    df = pd.DataFrame(data)

    df.replace("N/A", pd.NA)
    df.dropna(axis="columns", how="all", inplace=True)

    console.log(df)

    # Save the features
    df.to_csv(output_file_path, index=False)

    console.log("[bold green] Successful! [/bold green]")
