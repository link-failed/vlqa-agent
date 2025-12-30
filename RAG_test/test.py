import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('../.env')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

code_1 = """def filter_payment():
    # match attributes between fees.json and payments.csv
    # null value handling according to manual.md
"""

code_2 = """def match_merchant_capture_delay():
    # fuzzy matching
    # to filter applicable fees for merchant
"""

code_3 = """def match_merchant_month_volume():
    # map day_of_year to month
    # compute monthly volume
    # regex matching and unit conversion
    # to filter applicable fees for merchant"""

code_4 = """def null_attribute_handling():
    # null value handling for fee rule attributes like is_credit
"""

code_5 = """def match_merchant_monthly_fraud_level():
    # map day_of_year to month
    # filter fraudulent-dispute payments
    # compute and regex matching
    # to filter applicable fees for merchants
"""

code_6 = """def check_payment_intracountry():
    # to filter applicable payments
"""

question = "In January 2023 what delta would the merchant A pay if the relative fee of the fee with ID=384 changed to 1?"

def get_embedding(text, model="text-embedding-3-large"):
    """Get embedding for a given text using OpenAI's embedding model."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def compute_similarity_scores():
    """Compute similarity scores between code snippets and the question."""
    # Store all code snippets in a list
    code_snippets = [code_1, code_2, code_3, code_4, code_5, code_6]
    code_names = ['code_1', 'code_2', 'code_3', 'code_4', 'code_5', 'code_6']
    
    print("Computing embeddings...")
    
    # Get embedding for the question
    question_embedding = get_embedding(question)
    
    # Get embeddings for each code snippet
    code_embeddings = []
    for i, code in enumerate(code_snippets):
        print(f"Processing {code_names[i]}...")
        embedding = get_embedding(code)
        code_embeddings.append(embedding)
    
    # Convert to numpy arrays for cosine similarity calculation
    question_embedding = np.array(question_embedding).reshape(1, -1)
    code_embeddings = np.array(code_embeddings)
    
    # Compute cosine similarity scores
    similarities = cosine_similarity(question_embedding, code_embeddings)[0]
    
    # Create results
    results = []
    for i, similarity in enumerate(similarities):
        results.append({
            'code_snippet': code_names[i],
            'similarity_score': similarity,
            'description': code_snippets[i].split('#')[1].strip() if '#' in code_snippets[i] else 'No description'
        })
    
    # Sort by similarity score (highest first)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return results

def print_similarity_results(results):
    """Print the similarity results in a formatted way."""
    print("\n" + "="*80)
    print("SIMILARITY SCORES BETWEEN CODE SNIPPETS AND QUESTION")
    print("="*80)
    print(f"Question: {question}")
    print("\nRanked by similarity score (highest to lowest):")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['code_snippet']}")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Description: {result['description']}")
        print()

if __name__ == "__main__":
    try:
        # Compute similarity scores
        results = compute_similarity_scores()
        
        # Print results
        print_similarity_results(results)
        
        # Save results to a file
        import json
        with open('similarity_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to 'similarity_results.json'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install openai python-dotenv scikit-learn numpy")