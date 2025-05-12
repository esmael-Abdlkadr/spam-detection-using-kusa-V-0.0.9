import pandas as pd
import numpy as np
import re
import nltk
import spacy
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from kusa.utils import ensure_nltk_tokenizer_resources
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from kusa.exceptions import DatasetSDKException

class PreprocessingManager:
    def __init__(self):
        self.config = {}
        self.transformers = {}
        self.processed_df = None
        
    def configure(self, config):
        
        default_config = {
            "tokenizer": "nltk",             # or 'spacy', 'split', 'none'
            "stopwords": True,
            "lowercase": True,
            "remove_punctuation": True,
            "lemmatize": False,
            "reduction": "none",             # 'tfidf', 'pca', or 'none'
            "n_components": 2,               # For PCA
            "tfidf_max_features": 500,       # For TF-IDF
            "target_column": None,
            "output_format": "pandas"        # Can support tensor/numpy later
        }
        config = config or {}
        for key in config:
            if key not in default_config:
                print(f"⚠️ Unknown config key: '{key}' – ignoring.")
        # Merge with defaults
        self.config = {**default_config, **config}

    def run(self, raw_df):
        if raw_df is None:
            raise DatasetSDKException("Raw dataset not loaded. Fetch a batch first.")

        df = raw_df.copy()
        
        config = self.config
        target_column = config.get("target_column")

        # Check if target column exists
        if target_column and target_column not in df.columns:
            raise DatasetSDKException(f"Target column '{target_column}' not found in dataset.")

        # Extract target column if exists
        y = None
        if target_column:
            y = df[target_column]
            df = df.drop(columns=[target_column])

        # Preprocessing on text columns
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            df[col] = df[col].astype(str)

            # Lowercase
            if config.get("lowercase", True):
                df[col] = df[col].str.lower()

            # Remove punctuation
            if config.get("remove_punctuation", True):
                df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)

            # Tokenization
            tokenizer_type = config.get("tokenizer", "nltk")
            if tokenizer_type == "nltk":
                ensure_nltk_tokenizer_resources()
                df[col] = df[col].apply(word_tokenize)
                
            elif tokenizer_type == "spacy":
                try:
                    # Disable components not needed. Lemmatizer might need 'tagger'.
                    # If only tokenizing: disable=["parser", "ner", "tagger", "attribute_ruler", "lemmatizer"]
                    # If tokenizing + lemmatizing: disable=["parser", "ner"] (tagger & lemmatizer are often linked)
                    nlp_components_to_disable = ["parser", "ner"]
                    if not config.get("lemmatize", False): # If not lemmatizing, can disable more
                        nlp_components_to_disable.extend(["tagger", "attribute_ruler", "lemmatizer"])
                    
                    nlp = spacy.load("en_core_web_sm", disable=nlp_components_to_disable)
                except OSError:
                    # ... (error handling as before) ...
                    raise DatasetSDKException("Spacy model 'en_core_web_sm' not found.")

                print(f"SpaCy: Processing column '{col}' with nlp.pipe()...")
                texts_to_process = df[col].tolist()
                processed_docs = list(nlp.pipe(texts_to_process))
                
                if config.get("lemmatize", False):
                    print(f"SpaCy: Extracting lemmas for column '{col}'...")
                    df[col] = [[token.lemma_ for token in doc] for doc in processed_docs]
                else:
                    print(f"SpaCy: Extracting tokens for column '{col}'...")
                    df[col] = [[token.text for token in doc] for doc in processed_docs]
                
                print(f"SpaCy: Finished initial processing for column '{col}'.")

            elif tokenizer_type == "split":
                df[col] = df[col].apply(lambda x: x.split())
            elif tokenizer_type == "none":
                df[col] = df[col].apply(lambda x: [x])  # wrap as list

            # Stopword removal
            if config.get("stopwords", True):
                stop_words = set(stopwords.words("english"))
                df[col] = df[col].apply(lambda tokens: [w for w in tokens if w.lower() not in stop_words])

            # Lemmatization (optional, spaCy only)
            if config.get("lemmatize", False) and tokenizer_type == "spacy":
                df[col] = df[col].apply(lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))])

            # Rejoin tokens to string for vectorizers
            df[col] = df[col].apply(lambda tokens: " ".join(tokens))

        # ===============================
        # Feature Reduction (TF-IDF / PCA)
        # ===============================
        
        # "reduction": "pca" → it applies only to numeric columns.
        # "reduction": "tfidf" → it applies only to text columns.
        # "reduction": "tfidf_pca" → they need to handle both separately (TF-IDF for text, then PCA for numeric).
        
        reduction = config.get("reduction", "none") # Get reduction from merged config
        
        # --- Separate numeric and text columns from the current df ---
        # df at this point has original numeric columns and text columns as processed strings
        current_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        current_text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist() # Re-check text cols

        df_numeric_pca_components = pd.DataFrame(index=df.index) # To store PCA results from numeric data
        df_text_features = pd.DataFrame(index=df.index) # To store TF-IDF results from text data
        
        if reduction == "pca" or reduction == "tfidf_pca": # PCA on original numerics
            if current_numeric_cols:
                numeric_data_for_pca = df[current_numeric_cols].copy()
                # Handle NaNs in numeric columns before scaling (e.g., fill with mean or drop rows/cols carefully)
                # For simplicity, let's assume they are clean or dropna() is acceptable for now.
                numeric_data_for_pca.fillna(numeric_data_for_pca.mean(), inplace=True) # Example: fill with mean

                if not numeric_data_for_pca.empty and numeric_data_for_pca.shape[1] > 0:
                    n_components_numeric = config.get("n_components", 2) # Use general n_components
                    if n_components_numeric > numeric_data_for_pca.shape[1]:
                        n_components_numeric = numeric_data_for_pca.shape[1]

                    if n_components_numeric > 0:
                        scaler_numeric = StandardScaler()
                        scaled_numeric = scaler_numeric.fit_transform(numeric_data_for_pca)
                        
                        pca_numeric_model = PCA(n_components=n_components_numeric)
                        reduced_numeric_data = pca_numeric_model.fit_transform(scaled_numeric)
                        
                        pca_numeric_feature_names = [f"numeric_pca_{i+1}" for i in range(n_components_numeric)]
                        df_numeric_pca_components = pd.DataFrame(reduced_numeric_data, columns=pca_numeric_feature_names, index=df.index)
                        
                        self.transformers["numeric_pca"] = { # Store for numeric PCA
                            "scaler": scaler_numeric,
                            "pca": pca_numeric_model,
                            "original_numeric_columns": current_numeric_cols
                        }
            else: # No numeric columns to apply PCA on
                print("PCA reduction: No numeric columns found for PCA.")
                # df_numeric_pca_components remains empty

        # --- Part 2: Handle Text Columns ---
        # If reduction is "pca", we need to convert text to numbers (e.g., TF-IDF)
        # If reduction is "tfidf" or "tfidf_pca", TF-IDF is applied.
        
        if current_text_cols:
            if reduction == "tfidf" or reduction == "tfidf_pca" or reduction == "pca":
                # For "pca" only, we apply TF-IDF as a default way to numerize text
                # before potential further global PCA (which is more complex).
                # Here, we'll just get TF-IDF features if reduction is "pca".
                
                self.transformers["tfidf_vectorizers"] = {} # Initialize/reset
                all_tfidf_dfs = []

                for col in current_text_cols:
                    max_feats_tfidf = config.get("tfidf_max_features", 500)
                    vectorizer = TfidfVectorizer(max_features=max_feats_tfidf)
                    
                    # df[col] should be strings here
                    tfidf_matrix = vectorizer.fit_transform(df[col].astype(str))
                    self.transformers["tfidf_vectorizers"][col] = vectorizer
                    
                    feature_names = vectorizer.get_feature_names_out()
                    tfidf_df_col = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=[f"{col}_tfidf_{w}" for w in feature_names],
                        index=df.index
                    )
                    all_tfidf_dfs.append(tfidf_df_col)
                
                if all_tfidf_dfs:
                    df_text_features = pd.concat(all_tfidf_dfs, axis=1)
            
            elif reduction == "none" and not current_numeric_cols:
                # If reduction is none and there are ONLY text columns, what to do?
                # The model will likely fail. This case needs to be handled or warned.
                # For now, df_text_features would be empty, and df would only have processed text strings.
                print("Warning: Reduction is 'none' and only text columns exist after processing. Model may not work.")
                df_text_features = df[current_text_cols].copy() # Keep processed text strings
                                                              # This is unlikely to work for most sklearn models unless
                                                              # they are specifically NLP models that take string lists.


        # --- Part 3: Combine Features based on reduction strategy ---
        final_feature_dfs = []

        if reduction == "pca":
            # Combine PCA of original numerics + TF-IDF of text
            if not df_numeric_pca_components.empty:
                final_feature_dfs.append(df_numeric_pca_components)
            if not df_text_features.empty: # df_text_features contains TF-IDF if text_cols existed
                final_feature_dfs.append(df_text_features)
            if not final_feature_dfs: # If both were empty (e.g. no numeric, no text)
                df = pd.DataFrame(index=df.index) # Empty df
            else:
                df = pd.concat(final_feature_dfs, axis=1)

        elif reduction == "tfidf":
            # Only TF-IDF features from text + original numeric columns (if any, not PCA'd)
            if not df_text_features.empty: # This will be TF-IDF features
                 final_feature_dfs.append(df_text_features)
            if current_numeric_cols: # Add original numeric columns if they exist
                 final_feature_dfs.append(df[current_numeric_cols].copy()) # Use original numeric
            if not final_feature_dfs:
                df = pd.DataFrame(index=df.index)
            else:
                df = pd.concat(final_feature_dfs, axis=1)

        elif reduction == "tfidf_pca":
            # This one is more complex:
            # 1. TF-IDF on text -> combined_tfidf_matrix
            # 2. PCA on combined_tfidf_matrix -> reduced_tfidf_components
            # 3. Original numeric columns (optionally PCA'd if numeric_pca was also done - current code does this)
            # 4. Concatenate reduced_tfidf_components + (PCA'd or original) numeric columns

            # Assuming df_text_features contains the *combined* TF-IDF matrix from all text columns
            # (The current TF-IDF loop creates separate df_text_features, needs adjustment for combined TF-IDF before PCA)

            # For tfidf_pca, let's reconstruct the logic carefully:
            # 1. Get all TF-IDF outputs
            if current_text_cols:
                tfidf_matrices_list = []
                # (Re-run TF-IDF fitting here if not already done in a combined way)
                # For simplicity, assume df_text_features IS the combined TF-IDF output from earlier TF-IDF loop
                # This part of your original code was better structured for tfidf_pca:
                #   from scipy.sparse import hstack
                #   combined_tfidf = hstack(tfidf_outputs).toarray()
                #   pca_on_tfidf = PCA(...)
                #   reduced_tfidf = pca_on_tfidf.fit_transform(combined_tfidf)
                #   df_reduced_tfidf = pd.DataFrame(reduced_tfidf, ...)
                #   self.transformers["pca_for_tfidf"] = pca_on_tfidf
                # Then:
                #   final_feature_dfs.append(df_reduced_tfidf)
                
                # To reuse the structure:
                # Ensure df_text_features from the "Handle Text Columns" part is actually the
                # combined TF-IDF matrix (e.g., if only one text col, it's fine, otherwise concat earlier).
                # If multiple text cols, the TF-IDF logic needs to hstack them first.
                
                # Let's assume 'df_text_features' holds the (potentially large) combined TF-IDF matrix
                if not df_text_features.empty:
                    n_components_tfidf = config.get("n_components", 2) # Can be different from numeric PCA
                    if n_components_tfidf > df_text_features.shape[1] and df_text_features.shape[1] > 0:
                        n_components_tfidf = df_text_features.shape[1]
                    
                    if df_text_features.shape[1] > 0 and n_components_tfidf > 0:
                        pca_for_tfidf_model = PCA(n_components=n_components_tfidf)
                        reduced_tfidf_data = pca_for_tfidf_model.fit_transform(df_text_features.fillna(0)) # Fill NaNs for PCA
                        
                        pca_tfidf_feature_names = [f"tfidf_pca_{i+1}" for i in range(n_components_tfidf)]
                        df_reduced_tfidf_components = pd.DataFrame(reduced_tfidf_data, columns=pca_tfidf_feature_names, index=df.index)
                        final_feature_dfs.append(df_reduced_tfidf_components)
                        self.transformers["pca_for_tfidf"] = pca_for_tfidf_model # Store this PCA for TF-IDF features
            
            # Add original numeric columns (or PCA'd numeric columns if numeric_pca was also active)
            if not df_numeric_pca_components.empty: # If numeric PCA was done
                final_feature_dfs.append(df_numeric_pca_components)
            elif current_numeric_cols: # Else, if no numeric PCA but original numerics exist
                final_feature_dfs.append(df[current_numeric_cols].copy())


            if not final_feature_dfs:
                df = pd.DataFrame(index=df.index)
            else:
                df = pd.concat(final_feature_dfs, axis=1)
                
        elif reduction == "none":
            # Keep original numeric columns and processed text strings (which are in df already)
            # This means df from text processing (strings) + original numerics is the final df.
            # No further changes to 'df' columns if reduction is none, beyond what text processing did.
            # We just need to ensure numeric columns are still there if they existed.
            temp_dfs_for_none = []
            if current_numeric_cols:
                temp_dfs_for_none.append(df[current_numeric_cols].copy())
            if current_text_cols: # df[current_text_cols] are processed strings
                temp_dfs_for_none.append(df[current_text_cols].copy())
            
            if not temp_dfs_for_none:
                df = pd.DataFrame(index=df.index)
            else:
                df = pd.concat(temp_dfs_for_none, axis=1)


        # ===============================
        # Target Encoding (Auto / Custom)
        # ===============================
        if y is not None:
            # ... (your target encoding logic remains the same) ...
            # Ensure y is added back with correct index
            df[config.get("target_column")] = y.values if isinstance(y, pd.Series) else y


        self.processed_df = df
        print(f"✅ Preprocessing completed successfully. Final features shape: {df.shape}")
        
         
        # ===============================
        # Target Encoding (Auto / Custom)
        # ===============================
        if y is not None:
            encoding_mode = config.get("target_encoding", "auto")

            if encoding_mode == "auto":
                if y.dtype == object or y.dtype == "O":
                    unique_values = y.unique()
                    if len(unique_values) == 2:
                        mapping = {unique_values[0]: 0, unique_values[1]: 1}
                        y = y.map(mapping)
                        print(f"🔢 Auto target encoding applied: {mapping}")
                    else:
                        raise DatasetSDKException(f"Cannot auto-encode target with >2 classes: {unique_values}")
            elif isinstance(encoding_mode, dict):
                y = y.map(encoding_mode)
                print(f"🔢 Custom target encoding applied: {encoding_mode}")
            elif encoding_mode == "none":
                pass  # leave y unchanged
            else:
                raise DatasetSDKException(f"Invalid target_encoding value: {encoding_mode}")

            df[target_column] = y.reset_index(drop=True)

        # Final output format
        output_format = config.get("output_format", "pandas")
        if output_format != "pandas":
            raise DatasetSDKException(f"Output format '{output_format}' is not supported yet.")

        self.processed_df = df
        print("✅ Preprocessing completed successfully.")

    def get_processed_data(self):
        return self.processed_df

    def get_transformers(self):
        return self.transformers
