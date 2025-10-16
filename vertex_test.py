from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

print("--- Ollama Connection Test ---")
print("Make sure the Ollama application is running on your Mac.")

try:
    # --- 1. Initialize the Ollama Model ---
    # This connects to your locally running Ollama server.
    # We specify which model we want to use (e.g., "llama3").
    print("\nInitializing Ollama model (llama3)...")
    llm = ChatOllama(model="llama3")
    print("Model initialized successfully.")

    # --- 2. Send a Test Prompt ---
    # We send a simple message to the model to see if it responds.
    print("Sending a test prompt to Ollama...")
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant."),
            ("user", "{input}"),
        ]
    )
    
    chain = prompt_template | llm
    
    # --- 3. Get and Check the Result ---
    response = chain.invoke({"input": "Hello! In one short sentence, introduce yourself."})

    print("\n-----------------------------------------")
    if response and hasattr(response, 'content'):
        print("SUCCESS! Received a response from Ollama:")
        print(f"Ollama (llama3): {response.content}")
    else:
        print("FAILURE: The script ran but did not receive a valid response.")
    print("-----------------------------------------")

except Exception as e:
    # If any part of the process fails, this will print the error.
    print("\n-----------------------------------------")
    print("FAILURE: An error occurred while trying to connect to Ollama.")
    print(f"Error details: {e}")
    print("-----------------------------------------")
    print("\nCommon fixes:")
    print("1. Is the Ollama application running on your Mac?")
    print("2. Did you pull the model? (e.g., 'ollama pull llama3')")