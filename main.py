import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

import prompts.model_prompts as mp

# Page configuration
st.set_page_config(
	page_title='DataAlchemy - PDF Feature Extractor',
	page_icon='🧪',
	layout='wide',
)

# Styling
st.markdown(
	"""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """,
	unsafe_allow_html=True,
)


def setup_agent(api_key: str = None) -> Agent:
	"""Sets up the model to be used and returns a Pydantic Agent

	Args:
		api_key (str, optional): Google API key. If None, will try to get from environment.
	"""
	load_dotenv()

	# Use provided API key or try to get from environment
	api_key = api_key or os.getenv('GOOGLE_API_KEY', '')

	if not api_key:
		raise ValueError('Google API key is not set')

	model = GoogleModel(
		model_name='gemini-2.5-flash',
		provider=GoogleProvider(api_key=api_key),
		settings={'temperature': 0.1},
	)

	return Agent(model, system_prompt=mp.SYSTEM_PROMPT)


def process_files(files, progress_bar, status_text):
	"""Process uploaded PDF files and extract features"""
	agent = setup_agent(api_key=st.session_state.api_key)
	features = []
	data = []

	# Create a temporary directory to store uploaded files
	with tempfile.TemporaryDirectory() as temp_dir:
		# Process each uploaded file
		for i, uploaded_file in enumerate(files, start=1):
			# Save uploaded file temporarily
			temp_file = Path(temp_dir) / uploaded_file.name
			with open(temp_file, 'wb') as f:
				f.write(uploaded_file.getbuffer())

			# Update progress
			progress = (i - 1) / len(files)
			progress_bar.progress(progress)
			status_text.text(f'Processing file {i}/{len(files)}: {uploaded_file.name}')

			# Process the file
			content = BinaryContent(temp_file.read_bytes(), media_type='application/pdf')

			# Extract text from PDF
			with st.spinner(f'Extracting text from {uploaded_file.name}...'):
				pdf_text = agent.run_sync([mp.EXTRACT_TEXT_FROM_PDF, content])

			# Extract important features
			with st.spinner(f'Identifying important features in {uploaded_file.name}...'):
				important_features = agent.run_sync(f'{mp.INDENTIFY_IMPORTANT_FEATURES}\n {pdf_text}')

			# Combine features
			with st.spinner('Combining features...'):
				combined_features = agent.run_sync(f'{mp.COMBINE_FEATURES}\n{features} \n{important_features} ')
				features.extend(list(feature.strip() for feature in combined_features.output.split(',')))

			# Extract features from text
			with st.spinner(f'Extracting specific features from {uploaded_file.name}...'):
				extracted_features = agent.run_sync(f'{mp.EXTRACT_FEATURES}\n FEATURE: {features} \nTEXT: {pdf_text} ')
				extracted_features = extracted_features.output.replace('```', '').replace('json', '')
				extracted_features = eval(extracted_features)  # Convert string to dict
				extracted_features['Source file'] = uploaded_file.name
				data.append(extracted_features)

			# Update final progress
			progress_bar.progress((i) / len(files))

	# Create dataframe
	df = pd.DataFrame(data)
	df.replace('N/A', pd.NA, inplace=True)
	df.dropna(axis='columns', how='all', inplace=True)

	return df


def main():
	# Header
	st.title('🧪 DataAlchemy')
	st.subheader('PDF Feature Extractor')

	# Initialize uploaded_files as None
	uploaded_files = None

	# Check for API key in environment
	load_dotenv()
	api_key = os.getenv('GOOGLE_API_KEY', '')

	# If API key is not in environment, show input field
	if not api_key:
		st.warning('⚠️ Google API key not found in environment')
		api_key = st.text_input(
			'Enter your Google API Key', type='password', help='Get your API key from Google Cloud Console'
		)

		if not api_key:
			st.info('Please enter your Google API key to proceed')
			st.markdown("""
            #### How to get your API key:
            1. Go to [Google Cloud Console](https://console.cloud.google.com)
            2. Create a new project or select an existing one
            3. Enable the Gemini API
            4. Create credentials (API key)
            5. Copy and paste your API key above
            """)
			return

	# Store API key in session state
	if 'api_key' not in st.session_state:
		st.session_state.api_key = api_key

	try:
		# File uploader
		uploaded_files = st.file_uploader(
			'Upload your PDF files',
			type=['pdf'],
			accept_multiple_files=True,
			help='Select one or more PDF files to process',
		)
	except Exception as e:
		st.error(f'Error during file upload: {str(e)}')
		return

	# Validate uploaded files
	if uploaded_files is not None and len(uploaded_files) > 0:
		st.write(f'📁 {len(uploaded_files)} files selected')

		if st.button('🚀 Start Processing', type='primary'):
			# Create progress tracking elements
			progress_bar = st.progress(0)
			status_text = st.empty()

			try:
				# Process the files
				df = process_files(uploaded_files, progress_bar, status_text)

				# Save successful API key to .env file if it's not already there
				if not os.getenv('GOOGLE_API_KEY'):
					with open('.env', 'a') as f:
						f.write(f'\nGOOGLE_API_KEY={st.session_state.api_key}')
					st.success('✅ API key has been saved to .env file')

				# Show success message
				st.success('✅ Processing completed successfully!')

				# Display the results
				st.subheader('📊 Extracted Features')
				st.dataframe(df, width='stretch')

				# Download button
				csv = df.to_csv(index=False).encode('utf-8')
				st.download_button(
					label='📥 Download CSV',
					data=csv,
					file_name='extracted_features.csv',
					mime='text/csv',
				)

			except Exception as e:
				st.error(f'❌ An error occurred: {str(e)}')
			finally:
				# Clean up progress elements
				progress_bar.empty()
				status_text.empty()
	else:
		# Show instructions when no files are uploaded
		st.info('👆 Please upload one or more PDF files to begin processing')


if __name__ == '__main__':
	main()
