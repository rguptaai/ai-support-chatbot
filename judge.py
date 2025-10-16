import pandas as pd
from datasets import Dataset
from ragas import evaluate
# --- FIXED: Import uppercase class names ---
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance 
from langchain_ollama import ChatOllama, OllamaEmbeddings
import ast

# --- Ollama Configuration ---
print("Initializing Ollama models for evaluation...")
try:
    # We use a powerful model as the judge
    judge_llm = ChatOllama(model="llama3")
    # The embedding model must match the one used in the RAG system
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    print("Models initialized.")
except Exception as e:
    print(f"Failed to connect to Ollama. Is the Ollama application running? Details: {e}")
    exit()

# --- Load and Prepare the Log Data ---
log_file = 'evaluation_log.csv'
print(f"Reading log file from: {log_file}")

try:
    log_df = pd.read_csv(log_file)
    
    # Convert the 'contexts' column from a string representation of a list back into an actual list
    log_df['contexts'] = log_df['contexts'].apply(ast.literal_eval)
    
    # Convert the pandas DataFrame to a Hugging Face Dataset, which Ragas expects
    evaluation_dataset = Dataset.from_pandas(log_df)
    print("Log file loaded and converted to dataset format.")
    
except FileNotFoundError:
    print(f"Error: The log file '{log_file}' was not found. Please run the chatbot first to generate some data.")
    exit()
except Exception as e:
    print(f"An error occurred while preparing the dataset: {e}")
    exit()

# --- Define Evaluation Metrics ---
# --- FIXED: Instantiate the metric classes ---
metrics = [
    Faithfulness(),      # How factually consistent is the answer to the context?
    AnswerRelevancy(),  # How relevant is the answer to the question?
    ContextRelevance(), # How relevant are the retrieved contexts to the question?
]

# --- Run the Evaluation ---
print("Starting evaluation... This may take a few minutes.")

try:
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embedding_model,
        raise_exceptions=True 
    )
    
    print("Evaluation complete.")
    
    # --- Display and Save Results ---
    result_df = result.to_pandas()
    print("\n--- Evaluation Results ---")
    print(result_df)
    
    # Save the detailed results to a new CSV file
    results_csv_path = 'evaluation_results.csv'
    result_df.to_csv(results_csv_path, index=False)
    print(f"\nDetailed results saved to {results_csv_path}")

except Exception as e:
    print(f"An error occurred during evaluation: {e}")

