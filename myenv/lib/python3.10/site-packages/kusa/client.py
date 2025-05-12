import os
import base64
import pandas as pd
from io import StringIO

from kusa.config import Config
from kusa.utils import make_request
from kusa.exceptions import DatasetSDKException

from kusa.preprocessing_manager import PreprocessingManager
from kusa.model_manager import ModelManager
from kusa.encryption_manager import EncryptionManager

import math # For math.ceil
import gc 

class SecureDatasetClient:
    def __init__(self, public_id=None, secret_key=None, encryption_key=None):
        self.public_id = public_id or os.getenv("PUBLIC_ID")
        self.secret_key = secret_key or os.getenv("SECRET_KEY")
        self.encryption_key = encryption_key or Config.get_encryption_key()
        self.base_url = Config.get_base_url()

        self.__raw_df = None      # Private memory store for raw data
        self.__processed = None   # Private memory for preprocessed
        self.__metadata = {}
        self.__transformers={}

        self._validate_keys()
        self.headers = self._build_headers()
        
        self.__trained_model = None
        self.__X_val = None
        self.__y_val = None
        self.__input_feature_names = None
        
        self.preprocessing_manager = PreprocessingManager()
        self.model_manager = ModelManager()
        self.encryption_manager = EncryptionManager()
        
        def __getattribute__(self, name):
            if name in ['_SecureDatasetClient__encryption_key', 
                       '_SecureDatasetClient__secret_key',
                       'encryption_key', 'secret_key']:
                raise AttributeError("Access to sensitive attribute denied")
            return object.__getattribute__(self, name)

        

    def __dict__(self):
        # Return only non-sensitive details
        return {
            'dataset_id': self.public_id,
            'secret_key': self.secret_key,
            # Exclude encryption_key and other sensitive data
        }

    def __repr__(self):
        return f"SecureDatasetClient(dataset_id={self.public_id}, secret_key={self.secret_key})"
    
    def _validate_keys(self):
        if not self.public_id or not self.secret_key:
            raise DatasetSDKException("Missing PUBLIC_ID or SECRET_KEY.")
        if not self.encryption_key or len(self.encryption_key.encode()) != 32:
            raise DatasetSDKException("ENCRYPTION_KEY must be 32 bytes (AES-256).")

    def _build_headers(self):
        return {
            "Authorization": f"key {self.secret_key}"
        }

    def initialize(self):
            """Initializes metadata, including totalRows for the dataset."""
            print("🚀 Initializing SDK and fetching dataset metadata...")
            url = f"{self.base_url}/initialize/{self.public_id}"
            try:
                data = make_request(url, headers=self.headers)
            except Exception as e:
                raise DatasetSDKException(f"Failed to initialize SDK: {e}")

            # IMPORTANT: Your server's /initialize endpoint MUST return 'totalRows'
            # representing the total number of DATA rows (excluding header).
            total_data_rows = data.get("totalRows") # Assuming 'totalRows' from server IS data rows
            if total_data_rows is None:
                raise DatasetSDKException("Server response from /initialize did not include 'totalRows'.")
            
            columns = data.get("columns", [])
            if not columns:
                print("⚠️ Warning: Server response from /initialize did not include 'columns'.")

            preview_csv_string = data.get("first10Rows", "")
            preview_df = None
            if preview_csv_string:
                try:
                    preview_df = pd.read_csv(StringIO(preview_csv_string))
                except Exception as e:
                    print(f"⚠️ Warning: Could not parse preview CSV from initialize: {e}")
                    preview_df = pd.DataFrame(columns=columns)
            else:
                print("⚠️ Warning: No preview data ('first10Rows') received from initialize.")
                preview_df = pd.DataFrame(columns=columns)


            self.__metadata = {
                "totalDataRows": int(total_data_rows), # Explicitly calling it totalDataRows internally
                "columns": columns,
                "preview_df": preview_df 
            }

            print(f"✅ SDK initialized successfully. Total data rows: {self.__metadata['totalDataRows']}.")
            return {
                "preview": self.__metadata["preview_df"].head(10) if self.__metadata["preview_df"] is not None else "No preview available",
                "metadata": { # Return a copy or specific fields to avoid external modification of __metadata
                    "totalDataRows": self.__metadata["totalDataRows"],
                    "columns": self.__metadata["columns"]
                }
            }

    def preview(self):
        """
        Returns the current metadata dictionary which includes preview dataframe,
        total data rows, and column information.
        """
        if not self.__metadata:
            raise DatasetSDKException("Metadata is not initialized. Call initialize() first.")

        return self.__metadata


    def _fetch_and_decrypt_one_batch(self, batch_size: int, batch_number_for_api: int):
        """
        Internal helper: Fetches and decrypts a single batch.
        batch_number_for_api is 1-indexed.
        Returns a DataFrame or None if no more data/error.
        """
        # This method reuses most of your original fetch_and_decrypt_batch logic
        # but is now an internal helper.
        # print(f"   Fetching batch number: {batch_number_for_api}, size: {batch_size}...")
        url = f"{self.base_url}/get/{self.public_id}/encryptbatch"
        params = {"batchSize": batch_size, "batchNumber": batch_number_for_api}

        try:
            api_response_data = make_request(url, headers=self.headers, params=params)
        except DatasetSDKException as e:
            if "no more data available" in str(e).lower() or "invalid batch number" in str(e).lower():
                print(f"   ℹ️ Server indicated no more data for batch API call {batch_number_for_api}.")
                return None
            raise

        encrypted_data_b64 = api_response_data.get("batchData")
        encrypted_batch_key_b64 = api_response_data.get("encryptedKey")

        if not encrypted_data_b64 or not encrypted_batch_key_b64:
            print(f"   ⚠️ API response missing data/key for batch {batch_number_for_api}. Assuming end.")
            return None

        try:
            encrypted_data_bytes = base64.b64decode(encrypted_data_b64)
            encrypted_batch_key_bytes = base64.b64decode(encrypted_batch_key_b64)

            # self.encryption_key is K_common
            batch_specific_key_bytes = self.encryption_manager.decrypt(
                encrypted_batch_key_bytes, self.encryption_key.encode()
            )
            batch_specific_key_str = batch_specific_key_bytes.decode()

            raw_csv_bytes = self.encryption_manager.decrypt(
                encrypted_data_bytes, batch_specific_key_str.encode()
            )
            raw_csv_str = raw_csv_bytes.decode()
        except Exception as e:
            raise DatasetSDKException(f"Decryption failed for batch {batch_number_for_api}: {e}")

        try:
            df_batch = pd.read_csv(StringIO(raw_csv_str))
            # Server should always send header. If rowsInBatch from server is 0, df_batch might be empty (header only).
            actual_rows_in_batch_server = api_response_data.get("rowsInBatch", -1) # From server response
            if df_batch.empty and actual_rows_in_batch_server == 0:
                #  print(f"   ✅ Batch {batch_number_for_api} is header-only (0 data rows).")
                 # Return empty DataFrame with correct columns if possible
                 return pd.DataFrame(columns=self.__metadata.get("columns", df_batch.columns if not df_batch.empty else []))

            # print(f"   ✅ Batch {batch_number_for_api} decrypted ({len(df_batch)} rows including header, server reported {actual_rows_in_batch_server} data rows).")
            return df_batch
        except Exception as e:
            raise DatasetSDKException(f"Failed to parse CSV for batch {batch_number_for_api}: {e}")


    def fetch_and_decrypt_batch(self, batch_size: int = 500):
        """
        Fetches all batches for the dataset sequentially, decrypts them,
        and appends them to the internal `self.__raw_df`.
        This method effectively replaces the old single-batch `fetch_and_decrypt_batch`.
        """
        if not self.__metadata or "totalDataRows" not in self.__metadata:
            # If initialize() wasn't called or failed to get totalDataRows
            print("Metadata not initialized or totalDataRows missing. Calling initialize()...")
            self.initialize() 
            if "totalDataRows" not in self.__metadata: # Check again
                 raise DatasetSDKException("Failed to get totalDataRows after re-initialization.")

        total_data_rows = self.__metadata["totalDataRows"]
        if total_data_rows == 0:
            print("ℹ️ Dataset has 0 data rows according to metadata. No data to fetch.")
            self.__raw_df = pd.DataFrame(columns=self.__metadata.get("columns", []))
            return True

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        num_expected_batches = math.ceil(total_data_rows / batch_size)

        all_fetched_batches_list = []
        accumulated_data_rows = 0

        print("🚀 Encrption Decryption on progress")
        for i in range(num_expected_batches):
            current_batch_number_for_api = i + 1 # Server expects 1-indexed

            df_one_batch = self._fetch_and_decrypt_one_batch(
                batch_size=batch_size,
                batch_number_for_api=current_batch_number_for_api
            )

            if df_one_batch is None:
                print(f"🏁 No more data returned by server at batch API call {current_batch_number_for_api}. Stopping fetch.")
                break
            
            all_fetched_batches_list.append(df_one_batch)
          
            server_reported_data_rows = df_one_batch.attrs.get('server_rowsInBatch', len(df_one_batch)) # Placeholder
            
            if not df_one_batch.empty:
                accumulated_data_rows += len(df_one_batch) # This counts header if present in each non-empty batch

            # print(f"   Appended batch {current_batch_number_for_api}. DataFrame has {len(df_one_batch)} rows (incl. header).")

        if not all_fetched_batches_list:
            print("⚠️ No data batches were successfully fetched and decrypted.")
            self.__raw_df = pd.DataFrame(columns=self.__metadata.get("columns", []))
        else:
            try:
                self.__raw_df = pd.concat(all_fetched_batches_list, ignore_index=True)
                # print(f"✅ Entire dataset fetched. Final DataFrame has {len(self.__raw_df)} rows (incl. header).")
                # if len(self.__raw_df) != total_data_rows:
                #     print(f"ℹ️  Note: Assembled DataFrame has {len(self.__raw_df)} rows. Expected data rows from metadata: {total_data_rows}.")

            except Exception as e:
                raise DatasetSDKException(f"Failed to concatenate fetched batches: {e}")
        
        return True
    
    
    def configure_preprocessing(self, config: dict):
        """
            Accepts user-defined preprocessing config.
            Validates and stores it internally.
        """
        self.preprocessing_manager.configure(config)

    def run_preprocessing(self):
        """
        Full secure preprocessing pipeline for tabular/text data.
        Steps: Clean → Tokenize → Reduce → Target Encode → Store.
        """
        self.preprocessing_manager.run(self.__raw_df)
        self.__raw_df = None 
        gc.collect()
        
    def train(self, user_train_func, hyperparams: dict = None, target_column: str = None, 
            task_type: str = "classification", framework: str = "sklearn"):
        """
        Train a model using the user's function, securely inside the SDK.
        Supports sklearn, tensorflow, and pytorch frameworks.
        
        Args:
            user_train_func: User-defined training function
            hyperparams: Dictionary of hyperparameters
            target_column: Name of the target column
            task_type: Type of ML task ('classification' or 'regression')
            framework: ML framework to use ('sklearn', 'tensorflow', or 'pytorch')
        """
        self.model_manager.train(
            processed_df=self.preprocessing_manager.get_processed_data(),
            user_train_func=user_train_func,
            hyperparams=hyperparams,
            target_column=target_column,
            task_type=task_type,
            framework=framework
        )
        self.__trained_model = self.model_manager.get_modal()
        self.__X_val = self.model_manager.get_x_val()
        self.__y_val = self.model_manager.get_y_val()
        

            
    def evaluate(self):
        return self.model_manager.evaluate()
       
    def predict(self, input_df):

        return self.model_manager.predict(input_df, self.preprocessing_manager.get_transformers())


    def save_model(self, filepath: str):
        self.model_manager.save(filepath)
        

    def load_model(self, filepath: str, training_framework: str) -> None:
        self.model_manager.load(filepath, training_framework)