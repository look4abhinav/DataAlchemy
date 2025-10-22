# DataAlchemy 🧪

DataAlchemy is an intelligent document processing tool that transforms unstructured PDF documents into structured, analyzable data. Using state-of-the-art AI models (Google's Gemini), it automatically extracts and standardizes features from similar PDF documents, making data analysis more efficient and accessible.

## ✨ Features

- **Smart PDF Processing**: Batch process multiple PDF documents with similar content structure
- **AI-Powered Analysis**: 
  - Automatic text and image description extraction
  - Dynamic feature identification based on document context
  - Intelligent feature combination and deduplication
- **Interactive Web Interface**:
  - Drag-and-drop PDF upload
  - Real-time processing progress tracking
  - Interactive data preview
  - One-click CSV export
- **Structured Output**: Organized CSV format with standardized features

## 🛠️ Technology Stack

- **Core Technologies**:
  - Python 3.13+
  - Google Gemini AI Model (gemini-2.5-flash)
  - Streamlit for web interface
- **Key Libraries**:
  - `pydantic-ai`: For structured AI interactions
  - `pandas`: Data manipulation and CSV export
  - `python-dotenv`: Environment configuration
  - `streamlit`: Web interface framework

## 📋 Prerequisites

- Python 3.13 or higher
- Google API key for Gemini model access
- PDF documents with similar content structure

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/look4abhinav/DataAlchemy.git
   cd DataAlchemy
   ```

2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

3. **Install UV Package Manager**:
   ```macos and linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   ```windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

4. **Install Dependencies using UV**:
   ```bash
   uv sync --upgrade
   ```

5. **Configure Environment**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 📁 Project Structure

```
DataAlchemy/
├── artifacts/
│   └── test_docs/          # Directory for PDF files (CLI mode)
├── prompts/
│   └── model_prompts.py    # AI model prompts for document processing
├── app.py                  # Streamlit web interface
├── pyproject.toml          # Project configuration and dependencies
└── README.md              # Project documentation
```

## 🚀 Usage

### Web Interface (Recommended)
1. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`
3. Upload PDF files using the file uploader
4. Click "Start Processing" to begin feature extraction
5. View results in the interactive table
6. Download the CSV file using the "Download CSV" button


## 🔧 Configuration Options

### Model Settings
- Temperature: 0.1 (default, for consistent results)
- Model: gemini-2.5-flash

### Processing Configuration
- Supported file types: PDF
- Output format: CSV with standardized columns
- Missing values: Represented as 'N/A' in output

## 📊 Output Format

The generated CSV file includes:
- Dynamically identified features as columns
- One row per processed document
- 'N/A' for missing features
- Source file reference for traceability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🔍 Troubleshooting

Common issues and solutions:
- **API Key Error**: Ensure your Google API key is correctly set in `.env`
- **PDF Processing Error**: Verify PDFs are text-based, not scanned images
- **Import Errors**: Confirm all dependencies are installed via UV
- **Memory Issues**: Process smaller batches for large PDF files

## 📝 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

Copyright 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 🌟 Acknowledgments

- Google Gemini API for AI capabilities
- Streamlit for the web interface framework
- PydanticAI for structured AI interactions

## 📫 Support

For issues and feature requests, please use the GitHub issue tracker.
