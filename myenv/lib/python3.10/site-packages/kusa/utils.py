# kusa/utils.py

import requests
from .exceptions import DatasetSDKException
import nltk


# A good default User-Agent
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
# Or use a custom one for your app: 'MyKusaApp/1.0'

def make_request(url, headers=None, params=None):
    try:
        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)

        # Ensure User-Agent is set
        if 'User-Agent' not in request_headers:
            request_headers['User-Agent'] = DEFAULT_USER_AGENT
        
        response = requests.get(url, headers=request_headers, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_details = response.json()
            # Make sure DatasetSDKException is defined or imported
            raise DatasetSDKException(f"API request failed: Server response: {error_details}")
        except ValueError: # Renamed from json.JSONDecodeError to be more general for parsing
            raise DatasetSDKException(f"API request failed: {str(e)}. Response content: {response.text}")
    except requests.exceptions.RequestException as e:
        raise DatasetSDKException(f"API request failed: {str(e)}")
    except ValueError as e: # Catches JSONDecodeError if response.json() fails
        response_content = response.text if 'response' in locals() else 'N/A'
        raise DatasetSDKException(f"Failed to decode JSON response: {str(e)}. Response content: {response_content}") 

def ensure_nltk_tokenizer_resources():
        """Securely verify and download required NLTK resources"""
        try:
            # List of required NLTK resources
            resources = ['punkt', 'punkt_tab', 'stopwords']
            for res in resources:
                try:
                    # Check if the resource is already available
                    nltk.data.find(f'tokenizers/{res}' if res.startswith('punkt') else f'corpora/{res}')
                except LookupError:
                    # Download the resource if not found
                    nltk.download(res, quiet=True)
        except Exception as e:
            raise DatasetSDKException(f"Resource verification failed: {str(e)}")
        
        