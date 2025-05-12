import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from kusa.exceptions import DatasetSDKException
import os
from sklearn.decomposition import PCA
import numpy as np

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False



class ModelManager:
    def __init__(self):
        self.model = None
        self.training_framework = None
        self.task_type = None
        self.input_feature_names = None
        self.X_val = None
        self.y_val = None
        self.__transformers= None

    def train(self, processed_df, user_train_func, hyperparams, target_column, task_type, framework):
       
            if processed_df is None:
                raise DatasetSDKException("No processed data available. Run preprocessing first.")

            if not target_column or target_column not in processed_df.columns:
                raise DatasetSDKException(f"Invalid or missing target_column: '{target_column}'.")

            # Separate features and labels
            X =  processed_df.drop(columns=[target_column])
            y =  processed_df[target_column]

            print("📊 Class counts:\n", y.value_counts())

            # Prepare train/val split
            stratify_y = y if task_type == "classification" else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )

            self.training_framework = framework
            self.task_type = task_type
            self.input_feature_names = X_train.columns.tolist()
            
            model = None
            self.X_val_processed = X_val 
            self.y_val_processed = y_val

            try:
                # Framework-aware logic
                if framework == "sklearn":
                        model = user_train_func(X_train, y_train, **(hyperparams or {}))

                elif framework == "tensorflow":
                    if not TENSORFLOW_AVAILABLE:
                        raise ImportError(
                                "TensorFlow framework was selected, but TensorFlow is not installed. "
                                "Please install it by running: pip install kusa[tensorflow]"
                            )
                    if not isinstance(X_train, tf.Tensor):
                        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
                        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
                        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

                    model = user_train_func(X_train, y_train, X_val, y_val, **(hyperparams or {}))
                
                elif framework == "pytorch":
                    # Convert pandas to PyTorch Tensors for the training function
                    # The user_train_func (from factory) should expect tensors.
                    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                    # Target for BCELoss with Sigmoid output needs to be [N, 1] and float32
                    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
                    
                    X_val = torch.tensor(X_val.values, dtype=torch.float32)
                    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
                    
                    model = user_train_func(X_train_tensor, y_train_tensor, X_val, y_val, **(hyperparams or {}))
                
                else:
                    raise DatasetSDKException(f"Unsupported framework: '{framework}'")

            except Exception as e:
                raise DatasetSDKException(f"Model training failed: {str(e)}")

            # Save model and validation set
            self.model = model
            self.X_val = X_val
            self.y_val = y_val            

            print("✅ Training complete.")
            return model


    def predict(self, input_df,transformers):
        if not self.__transformers:
          self.__transformers = transformers

        if self.model is None:
            raise DatasetSDKException("Model not trained or loaded.")

        # ✅ Ensure input is always a DataFrame (even if it's a Tensor)
        if not isinstance(input_df, pd.DataFrame):
            if isinstance(input_df, tf.Tensor):
                input_df = pd.DataFrame(input_df.numpy(), columns=self.input_feature_names)
            elif isinstance(input_df, torch.Tensor):
                input_df = pd.DataFrame(input_df.numpy(), columns=self.input_feature_names)
            else:
                raise DatasetSDKException("Unsupported input type for prediction.")

        # ✅ Apply saved vectorizers
        if "tfidf_vectorizers" in self.__transformers:
           # Prepare input_df to match trained TF-IDF structure
            all_cols = []

            for col, vec in self.__transformers["tfidf_vectorizers"].items():
                if col in input_df.columns:
                    tfidf = vec.transform(input_df[col].astype(str))
                    feature_names = [f"{col}_tfidf_{w}" for w in vec.get_feature_names_out()]
                    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=feature_names)

                    # Save the column names for padding later
                    all_cols.extend(feature_names)

                    input_df = input_df.drop(columns=[col]).join(tfidf_df)

            # Ensure consistent column order and fill missing ones
            if self.training_framework == "sklearn":
                trained_cols = self.model.feature_names_in_  # sklearn sets this after .fit()
            else:
                trained_cols = self.input_feature_names


            input_df = input_df.reindex(columns=trained_cols, fill_value=0)

                    
        elif "pca_tfidf" in self.__transformers:
            tfidf_vecs = self.__transformers["tfidf_vectorizers"]
            tfidf_outputs = []

            for col, vec in tfidf_vecs.items():
                if col in input_df.columns:
                    tfidf_matrix = vec.transform(input_df[col].astype(str))
                    tfidf_outputs.append(tfidf_matrix)
                else:
                    # If missing column, fill with zeros
                    tfidf_outputs.append(np.zeros((1, vec.max_features)))

            from scipy.sparse import hstack
            combined_tfidf = hstack(tfidf_outputs).toarray()

            # Apply PCA
            pca = self.__transformers["pca_tfidf"]["pca"]
            reduced = pca.transform(combined_tfidf)
            reduced_df = pd.DataFrame(reduced, columns=self.__transformers["pca_tfidf"]["pca_feature_names"])

            # Include numeric columns if any
            numeric_input = input_df.select_dtypes(include=[np.number])
            input_df = pd.concat([numeric_input.reset_index(drop=True), reduced_df], axis=1)

        elif "pca" in self.__transformers:
            scaler = self.__transformers["pca"]["scaler"]
            pca = self.__transformers["pca"]["pca"]

            # Only scale numeric columns (just like training)
            numeric_input = input_df.select_dtypes(include=[np.number]).dropna(axis=1)

            # ⚠️ Handle missing columns if user omits any
            trained_cols = self.__transformers["pca"].get("trained_numeric_columns", [])
            numeric_input = numeric_input.reindex(columns=trained_cols, fill_value=0)

            # Scale → Apply PCA
            scaled = scaler.transform(numeric_input)
            reduced = pca.transform(scaled)

            # input_df = pd.DataFrame(reduced, columns=[f"pca_{i+1}" for i in range(reduced.shape[1])])
            input_df = pd.DataFrame(reduced, columns=self.__transformers["pca"]["trained_pca_feature_names"])


        try:
            if self.training_framework == "sklearn":
                return self.model.predict(input_df)

            elif self.training_framework == "tensorflow":
                return (self.model.predict(input_df) > 0.5).astype("int32").flatten()

            elif self.training_framework == "pytorch":
                self.model.eval()
                with torch.no_grad():
                    inputs = torch.tensor(input_df.values, dtype=torch.float32)
                    outputs = self.model(inputs)
                    return torch.argmax(outputs, dim=1).numpy()

            else:
                raise DatasetSDKException(f"Unsupported framework: {self.training_framework}")

        except Exception as e:
            raise DatasetSDKException(f"Prediction failed: {str(e)}")

    def evaluate(self):
        if self.model is None or self.X_val is None:
            raise DatasetSDKException("No trained model or validation data available.")

        try:
            if self.training_framework == "sklearn":
                preds = self.model.predict(self.X_val)

            elif self.training_framework == "tensorflow":
                preds = (self.model.predict(self.X_val) > 0.5).astype("int32").flatten()

            elif self.training_framework == "pytorch":
                self.model.eval()
                with torch.no_grad():
                    # Ensure self.X_val is converted to tensor correctly
                    if not isinstance(self.X_val, torch.Tensor):
                        inputs = torch.tensor(self.X_val.values, dtype=torch.float32)
                    else:
                        inputs = self.X_val # If it was already converted and stored as tensor
                    
                    outputs = self.model(inputs) # outputs are probabilities if model ends with Sigmoid
                    preds = (outputs > 0.5).int().cpu().numpy().flatten() # Convert probs to 0/1
        
            else:
                raise DatasetSDKException(f"Unsupported framework: {self.training_framework}")

        except Exception as e:
            raise DatasetSDKException(f"Evaluation failed: {str(e)}")

        if self.task_type == "classification":
            accuracy = accuracy_score(self.y_val, preds)
            report = classification_report(self.y_val, preds)
            return {
                "accuracy": accuracy,
                "report": report
            }

        elif self.task_type == "regression":
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(self.y_val, preds)
            r2 = r2_score(self.y_val, preds)
            return {
                "mse": mse,
                "r2_score": r2
            }

        else:
            raise DatasetSDKException(f"Unsupported task_type: '{self.task_type}'")


    def save(self, filepath):
        if self.model is None:
            raise DatasetSDKException("No trained model to save.")

        # Ensure extension for TensorFlow
        if self.training_framework == "tensorflow":
            if not filepath.endswith(".keras") and not filepath.endswith(".h5"):
                filepath += ".keras"

        # Save model
        if self.training_framework == "sklearn":
            joblib.dump(self.model, filepath)

        elif self.training_framework == "tensorflow":
            self.model.save(filepath)

        elif self.training_framework == "pytorch":
            torch.save(self.model.state_dict(), filepath)

        else:
            raise DatasetSDKException(f"Unsupported framework for saving: {self.training_framework}")

        # 🔐 Bundle and save transformers + input feature names
        vec_path = f"{filepath}.bundle.pkl"
        bundle = {
            "transformers": self.__transformers,
            "input_feature_names": self.input_feature_names
        }
        joblib.dump(bundle, vec_path)
        print(f"🧠 Saved preprocessing bundle to: {vec_path}")
        print(f"✅ Model saved to: {filepath}")


    def load(self, filepath, training_framework):

        if not training_framework:
            raise DatasetSDKException("Training framework is required to load model.")
        if not os.path.exists(filepath):
            raise DatasetSDKException(f"No model found at: {filepath}")
        if training_framework not in ["pytorch", "tensorflow", "sklearn"]:
            raise DatasetSDKException(f"Unsupported framework: {training_framework}")

        self.training_framework = training_framework

        try:
            if training_framework == "sklearn":
                self.model = joblib.load(filepath)

            elif training_framework == "tensorflow":
                self.model = tf.keras.models.load_model(filepath)

            elif training_framework == "pytorch":
                if self.model is None:
                    raise DatasetSDKException(
                        "For PyTorch, model architecture must be initialized before loading weights. "
                        "Call initialize_pytorch_model() first or provide model architecture."
                    )
                self.model.load_state_dict(torch.load(filepath))
                self.model.eval()

        except Exception as e:
            raise DatasetSDKException(f"Failed to load {training_framework} model: {str(e)}")

        # 🔁 Load preprocessing config
        vec_path = f"{filepath}.bundle.pkl"
        if os.path.exists(vec_path):
            bundle = joblib.load(vec_path)
            self.__transformers = bundle.get("transformers", {})
            self.input_feature_names = bundle.get("input_feature_names", None)
            print("self.__transformers ",self.__transformers)
            print(f"✅ Loaded preprocessing bundle from: {vec_path}")
    
    
    def get_x_val(self): return self.X_val_processed # Returns the Pandas DataFrame
    def get_y_val(self): return self.y_val_processed # Returns the Pandas Series
    
    def get_modal(self):
        return self.model