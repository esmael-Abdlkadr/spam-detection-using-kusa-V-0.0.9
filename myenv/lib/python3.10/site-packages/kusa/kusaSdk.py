import requests
import os
import base64
import crypto

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
class KusaaSDK:
    """
    KusaaSDK handles secure access, decryption, and processing of datasets for ML/DL model training.
    The SDK operates internally to fetch and process data, ensuring no decrypted data is exposed to the client.
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str, encryption_key: str = None):
        """
        Initialize the KusaaSDK with API credentials and encryption settings.
        
        :param api_key: API key for authenticating requests
        :param secret_key: Secret key for authorizing requests
        :param base_url: Base URL for the API endpoint
        :param encryption_key: Environment encryption key (optional, falls back to process.env)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url =  os.getenv('BASE_URL')
        self.encryption_key = encryption_key or os.getenv("ENCRYPTION_KEY")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        if not self.encryption_key:
            raise ValueError("ENCRYPTION_KEY must be provided or set in environment variables.")

    def fetch_and_process_batch(self, dataset_id: str, batch_size: int, batch_number: int, preprocess_func=None):
        """
        Fetch encrypted batch data, decrypt it in memory, and process it according to the client's needs.
        
        :param dataset_id: Dataset ID provided by the client
        :param batch_size: Size of the batch to retrieve
        :param batch_number: Batch number to fetch
        :param preprocess_func: Optional custom preprocessing function to apply
        :return: The result after processing the batch (not the raw decrypted data)
        """
        # Fetch encrypted batch data and encrypted key from the server
        encrypted_data, encrypted_key = self._fetch_batch_data(dataset_id, batch_size, batch_number)

        # Decrypt the key using the environment encryption key
        decrypted_key = self._decrypt_key(encrypted_key)

        # Decrypt the batch data in memory using the decrypted key
        decrypted_data = self._decrypt_batch(encrypted_data, decrypted_key)

        # Preprocess the data (if a preprocessing function is provided)
        if preprocess_func:
            processed_data = preprocess_func(decrypted_data)
        else:
            processed_data = self._default_preprocessing(decrypted_data)

        # Return results based on processed data (e.g., model training result, summary statistics)
        return self._perform_operation(processed_data)

    def _fetch_batch_data(self, dataset_id: str, batch_size: int, batch_number: int):
        """
        Internal method to fetch the encrypted batch data and encrypted key.
        
        :param dataset_id: ID of the dataset to retrieve
        :param batch_size: Size of the batch to retrieve
        :param batch_number: Batch number (1-based index)
        :return: Encrypted batch data and encrypted key
        """
        endpoint = f"{self.base_url}/dataset/get/{dataset_id}/batch"
        params = {"batchSize": batch_size, "batchNumber": batch_number}
        response = self.session.get(endpoint, params=params)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch batch data: {response.text}")

        return response.json()["batchData"], response.json()["encryptedKey"]

    def _decrypt_key(self, encrypted_key: str) -> str:
        """
        Decrypt the encrypted key using the environment's encryption key.
        
        :param encrypted_key: Base64 encoded encrypted key
        :return: Decrypted key as a string
        """
        encrypted_key_bytes = base64.b64decode(encrypted_key)
        decrypted_key = self._decrypt_data(encrypted_key_bytes, self.encryption_key)
        return decrypted_key.decode("utf-8")

    def _decrypt_batch(self, encrypted_data: str, decrypted_key: str) -> str:
        """
        Decrypt the encrypted batch data using the decrypted key.
        
        :param encrypted_data: Base64 encoded encrypted batch data
        :param decrypted_key: Decrypted key to decrypt batch data
        :return: Decrypted batch data as a string (CSV format)
        """
        encrypted_data_bytes = base64.b64decode(encrypted_data)
        decrypted_data = self._decrypt_data(encrypted_data_bytes, decrypted_key)
        return decrypted_data.decode("utf-8")

    def _decrypt_data(self, encrypted_data: bytes, key: str) -> bytes:
        """
        Internal method to handle decryption logic for batch data and keys.
        
        :param encrypted_data: Encrypted data (bytes)
        :param key: Key used to decrypt the data
        :return: Decrypted data (bytes)
        """
        iv = encrypted_data[:16]  # Assuming first 16 bytes are the IV
        encrypted_content = encrypted_data[16:]

        decipher = crypto.create_decipheriv("aes-256-cbc", key.encode(), iv)
        decrypted_data = decipher.update(encrypted_content) + decipher.final()
        return decrypted_data

    def _default_preprocessing(self, data: str) -> str:
        """
        Internal method for default preprocessing of decrypted data.
        
        :param data: Decrypted CSV data
        :return: Preprocessed data (e.g., normalized, scaled, etc.)
        """
        # Implement default preprocessing logic (e.g., normalization)
        return data  # Placeholder: return data as is

    def _perform_operation(self, processed_data: str):
        """
        Internal method to perform the operation requested by the client.
        The client never gets access to the decrypted data, only the result.
        
        :param processed_data: Data after preprocessing (still in-memory)
        :return: Result of the operation (e.g., model training summary, statistics)
        """
        # Placeholder for an operation like model training
        # Example: Return summary statistics or training result
        return {
            "status": "success",
            "message": "Operation completed",
            "result": processed_data[:100]  # Return part of the processed data
        }
