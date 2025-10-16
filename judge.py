import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_ollama import ChatOllama, OllamaEmbeddings

class RagasEvaluator:
    """
    Uses the Ragas framework with local Ollama models to evaluate the performance
    of a RAG system based on a log file.
    """
    def __init__(self, llm_model="llama3", embedding_model="nomic-embed-text"):
        """
        Initializes the evaluator with the specified Ollama models.
        """
        print("Initializing Ollama models for evaluation...")
        # The LLM used for judging the responses
        self.llm = ChatOllama(model=llm_model)
        # The embedding model used for semantic similarity checks
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        print("Models initialized.")

    def evaluate_from_log(self, log_file_path):
        """
        Reads a CSV log file, prepares the data, and runs the Ragas evaluation.
        """
        try:
            print(f"Reading log file from: {log_file_path}")
            df = pd.read_csv(log_file_path)
            
            # Convert the pandas DataFrame into a Hugging Face Dataset object
            dataset = Dataset.from_pandas(df)
            print("Log file loaded and converted to dataset format.")

            print("Starting evaluation... This may take a few minutes.")
            # Run the evaluation with the specified metrics
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,       # How factually consistent is the answer to the context?
                    answer_relevancy,   # How relevant is the answer to the question?
                    context_recall,     # Does the retrieved context contain all necessary info?
                    context_precision,  # Is the retrieved context signal stronger than the noise?
                ],
                llm=self.llm,
                embeddings=self.embeddings,
            )
            print("Evaluation complete.")
            
            return result

        except FileNotFoundError:
            print(f"Error: The log file was not found at '{log_file_path}'")
            return None
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            return None

if __name__ == "__main__":
    evaluator = RagasEvaluator()
    evaluation_result = evaluator.evaluate_from_log('evaluation_log.csv')

    if evaluation_result:
        print("\n--- RAG System Evaluation Report ---")
        df_results = evaluation_result.to_pandas()
        print(df_results)
        
        # Save the detailed results to a new CSV file
        results_filename = 'evaluation_results_ollama.csv'
        df_results.to_csv(results_filename, index=False)
        print(f"\nDetailed evaluation results saved to '{results_filename}'")
