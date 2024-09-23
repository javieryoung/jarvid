import numpy as np
import pandas as pd
import os
import faiss
from openai import Client, OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def prompt_handler(prompt: str):
    return generate_response(prompt)


# Main function to generate the response
def generate_response(user_prompt: str) -> str:
    """
    Generates the complete response depending on whether the user is looking for an expert or has a general inquiry.
    """
    # Classify the user's intent using GPT directly
    user_intent = classify_user_intent_with_gpt(user_prompt)

    if user_intent == "search expert":
        print("Wait a few moments while I search the database")
        print("This shouldn't take more than 20 seconds")
        print()
        
        # Handle expert search
        expert_search_prompt = handle_expert_search(user_prompt)
        
        # Generate the response using the GPT-4 model
        expert_helper_chat = gpt_completion("gpt-4", expert_search_prompt)
        
        return expert_helper_chat.choices[0].message.content
    
    else:
        # If it's a general inquiry, we construct a generic prompt
        system_prompt = build_general_prompt(user_prompt)

        # Generate the response using the GPT-4 model
        standard_chat = gpt_completion("gpt-4", system_prompt)
        
        return standard_chat.choices[0].message.content

def gpt_completion(model: str, prompt: str) -> dict:
    """
    Sends a prompt to the GPT model and retrieves the completion.

    Args:
        model (str): The model to be used, such as "gpt-4".
        prompt (str): The prompt or system message to be sent to the GPT model.

    Returns:
        dict: The response from the GPT model, containing the completion.
    """
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt,
            }
        ]
    )
    return chat


# Function to handle the expert search logic
def handle_expert_search(user_prompt: str) -> str:
    """
    Handles the complete expert search logic based on the user prompt.
    """
    # Generate embeddings for the user prompt
    embedding_user_array = generate_user_embeddings(user_prompt)
    
    # Find the nearest experts using FAISS
    indices, distances = find_similar_experts(embedding_user_array)
    
    # Load professional profiles directly here
    profile_df = pd.read_csv("experts_profile.csv")
    
    # Select the profiles of the most similar experts
    selected_professionals_string = select_professionals(indices, profile_df)
    
    # Build the system prompt by combining the user prompt and the selected expert profiles
    system_prompt = f""" You are a chatbot for a staff enhancement company. 
        Your goal is to assist employees in finding professionals inside the company who can help them if they have questions.
        
        The user query in this case is: {user_prompt}
        
        And you have the next list of colleagues who can help him:
        {selected_professionals_string}.

        Answer cordially with the name of the experts and a brief description of their expertise and technology knowledge. 
    """
    
    return system_prompt

# Function to generate embeddings for the user's prompt
def generate_user_embeddings(user_prompt: str) -> np.ndarray:
    """
    Generates embeddings for the user prompt using the specified model.
    """
    embedding_user = client.embeddings.create(
        input=user_prompt,
        model="text-embedding-ada-002"
    )
    return np.array(embedding_user.data[0].embedding)

# Function to find the nearest neighbors using FAISS
def find_similar_experts(embedding_user_array: np.ndarray, k: int = 4):
    """
    Finds the top k closest experts based on the user's embeddings.
    """
    # Load precomputed normalized embeddings
    normalized_embeddings = np.load("normalized_embeddings.npy")
    
    # Find the nearest neighbors using FAISS
    indices, distances = find_nearest_neighbors_faiss(normalized_embeddings, embedding_user_array, k=k)
    
    return indices, distances

# Function to select the most similar expert profiles
def select_professionals(indices: np.ndarray, profile_df: pd.DataFrame) -> str:
    """
    Selects the profiles of the most similar experts based on the provided indices.
    """
    selected_professionals = []
    
    # Append the profiles of the most similar professionals
    for n in indices[0]:
        selected_professionals.append(create_profile_prompt(profile_df.iloc[n]))

    # Convert the list into a single string, separated by new lines
    return "\n".join(selected_professionals)


# Function to classify the user's intent using GPT directly
def classify_user_intent_with_gpt(user_prompt: str) -> str:
    """
    Uses the GPT model to classify the user's intent.
    Returns "search expert" or "general inquiry" depending on the user's query.
    """
    classification_prompt = f"""You are a helpful assistant. The user has asked: '{user_prompt}'.
    
    Please classify this query as one of the following:
    1. "search expert" (The user is looking for an expert to assist them with a specific topic.)
    2. "general inquiry" (The user is asking a general question that doesn't require a company expert.)
    
    Respond only with one of the two options: "search expert" or "general inquiry".
    """

    # Call to GPT to classify the intent
    classification_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that classifies user queries."
            },
            {
                "role": "user",
                "content": classification_prompt
            }
        ]
    )
    
    # Extract the classification from the model
    intent = classification_response.choices[0].message.content
    
    #print(intent)
    return intent


def build_general_prompt(user_prompt: str) -> str:
    """
    Builds the system prompt for general inquiries.
    """
    system_prompt = f"""You are a helpful assistant. The user asked: {user_prompt}. 
        Please respond with a useful and informative answer."""
    return system_prompt


def find_nearest_neighbors_faiss(normalized_embeddings: np.ndarray, unit_vector: np.ndarray, k: int = 4) -> tuple:
    """
    Creates a FAISS index using L2 distance (which acts like cosine similarity for normalized vectors), 
    adds normalized embeddings to the index, and finds the k nearest neighbors to a query vector.

    Parameters:
    normalized_embeddings (np.ndarray): A 2D array of normalized embeddings to be added to the FAISS index.
    unit_vector (np.ndarray): The query vector, which should also be normalized and reshaped to (1, dimension).
    k (int): The number of nearest neighbors to find. Default is 4.

    Returns:
    tuple: A tuple containing two elements:
        - indices (np.ndarray): The indices of the k nearest neighbors in the embedding space.
        - distances (np.ndarray): The corresponding distances to the nearest neighbors.
    """
    
    # Step 1: Get the dimensionality of the embeddings
    dimension = normalized_embeddings.shape[1]
    
    # Step 2: Create a FAISS index based on L2 distance
    index = faiss.IndexFlatL2(dimension)
    
    # Step 3: Add normalized embeddings to the FAISS index
    index.add(normalized_embeddings)
    
    # Step 4: Ensure the query vector is reshaped to (1, dimension)
    query_vector = unit_vector.reshape(1, -1)
    
    # Step 5: Perform the search for the k nearest neighbors
    distances, indices = index.search(query_vector, k)
    
    # Return the indices of the nearest neighbors and their corresponding distances
    return indices, distances



def create_profile_prompt(input_dataframe: pd.DataFrame) -> str:
    """
    Generates a profile prompt string using specific information from the input DataFrame.
    
    Args:
        input_dataframe (pd.DataFrame): A DataFrame containing columns such as 'Name', 
                                        'Partner', 'Industry', 'Technologies', and 'Processed_CV'.
                                        
    Returns:
        str: A formatted string that describes the person's profile based on the DataFrame content.
    """
    profile_prompt_string = f""" {input_dataframe["Name"]} works in 
    {input_dataframe["Partner"]}, a company of {input_dataframe["Industry"]}
    which works with these technologies {input_dataframe["Technologies"]}.
    He has expertise in {input_dataframe["Processed_CV"]}
    """
    
    return profile_prompt_string