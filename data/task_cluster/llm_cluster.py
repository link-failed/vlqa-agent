import pandas as pd
import openai
import json
import os
from dotenv import load_dotenv
from typing import List
import time

class LLMClusterer:
    def __init__(self, env_path: str = "../../.env"):
        """Initialize the LLM Clusterer with OpenAI API key from environment file."""
        load_dotenv(env_path)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = openai.OpenAI(api_key=api_key)
        print("✓ LLM Clusterer initialized")

    def create_prompt(self, question: str, level: str, reference_questions: List[str], 
                     reference_ids: List[int], reference_levels: List[str]) -> str:
        """Create a simple prompt for LLM clustering."""
        
        ref_text = "\n".join([f"{ref_id} ({ref_level}): {question}" 
                             for ref_id, question, ref_level in zip(reference_ids, reference_questions, reference_levels)])
        
        return f"""Find 1-3 most similar reference questions that could help solve this question.

REFERENCE QUESTIONS:
{ref_text}

QUESTION TO MATCH ({level}):
{question}

Return JSON with 1-3 most similar reference IDs:
{{"similar_ids": [1681, 1273]}}"""

    def cluster_questions(self, questions: List[str], levels: List[str], reference_questions: List[str], 
                         reference_ids: List[int], reference_levels: List[str]) -> List:
        """Cluster questions using LLM, one at a time."""
        all_results = []
        
        for i, (question, level) in enumerate(zip(questions, levels)):
            print(f"Processing question {i+1}/{len(questions)}...")
            
            prompt = self.create_prompt(question, level, reference_questions, reference_ids, reference_levels)
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                try:
                    result = json.loads(content)
                    similar_ids = result.get('similar_ids', reference_ids[:2])
                    
                    all_results.append({
                        "question": i + 1,
                        "similar_ids": similar_ids
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    # Create fallback result with 2 reference IDs
                    all_results.append({
                        "question": i + 1,
                        "similar_ids": reference_ids[:2]
                    })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"API call failed: {e}")
                # Create fallback result with 2 reference IDs
                all_results.append({
                    "question": i + 1,
                    "similar_ids": reference_ids[:2]
                })
        
        return all_results

    def apply_clustering(self, df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
        """Apply LLM clustering to a dataframe."""
        
        questions = df['question'].astype(str).str.strip().tolist()
        levels = df['level'].astype(str).str.strip().tolist()
        
        reference_questions = reference_df['question'].astype(str).str.strip().tolist()
        reference_ids = reference_df['task_id'].tolist()
        reference_levels = reference_df['level'].astype(str).str.strip().tolist()
        
        print(f"Clustering {len(questions)} questions with {len(reference_questions)} references")
        
        results = self.cluster_questions(questions, levels, reference_questions, reference_ids, reference_levels)
        
        df_output = df.copy()
        
        # Create mapping from reference_ids to reference_questions
        id_to_question = dict(zip(reference_ids, reference_questions))
        
        clusters = []
        referred = []
        referred_questions = []
        
        for result in results:
            similar_ids = result['similar_ids']
            
            clusters.append(str(similar_ids))
            referred.append(str(similar_ids))
            
            # Get the actual reference questions
            ref_questions = [id_to_question.get(ref_id, f"ID_{ref_id}") for ref_id in similar_ids]
            referred_questions.append(' || '.join(ref_questions))
        
        df_output['clusters'] = clusters
        df_output['referred'] = referred
        df_output['referred_question'] = referred_questions
        
        print("✓ Clustering completed")
        return df_output

def load_sample_data():
    """Load sample data from all.csv."""
    try:
        df = pd.read_csv('all.csv')
        columns_to_keep = ['task_id', 'question', 'level']
        df = df[[col for col in columns_to_keep if col in df.columns]]
        print(f"✓ Loaded {len(df)} questions from all.csv")
        return df
    except FileNotFoundError:
        print("❌ File 'all.csv' not found")
        return None

def load_reference_data():
    """Load reference data from dev.csv."""
    try:
        reference_df = pd.read_csv('dev.csv')
        # Keep all columns we need: task_id, question, level
        columns_to_keep = ['task_id', 'question', 'level']
        reference_df = reference_df[[col for col in columns_to_keep if col in reference_df.columns]]
        
        print(f"✓ Loaded {len(reference_df)} reference questions from dev.csv")
        return reference_df
        
    except FileNotFoundError:
        print("❌ File 'dev.csv' not found")
        # Fallback: create reference dataframe with hardcoded data
        reference_ids = [5, 49, 1273, 1305, 1464, 1681, 1753, 1871, 2697]
        reference_questions = [
            "Which issuing country has the highest number of transactions?",
            "What is the top country (ip_country) for fraud? A. NL, B. BE, C. ES, D. FR",
            "For credit transactions, what would be the average fee that the card scheme GlobalCard would charge for a transaction value of 10 EUR?",
            "For account type H and the MCC description: Eating Places and Restaurants, what would be the average fee that the card scheme GlobalCard would charge for a transaction value of 10 EUR? Provide the answer in EUR and 6 decimals",
            "What is the fee ID or IDs that apply to account_type = R and aci = B?",
            "For the 10th of the year 2023, what are the Fee IDs applicable to Belles_cookbook_store?",
            "What are the applicable fee IDs for Belles_cookbook_store in March 2023?",
            "In January 2023 what delta would Belles_cookbook_store pay if the relative fee of the fee with ID=384 changed to 1?",
            "For Belles_cookbook_store in January, if we were to move the fraudulent transactions towards a different Authorization Characteristics Indicator (ACI) by incentivizing users to use a different interaction, what would be the preferred choice considering the lowest possible fees?"
        ]
        reference_levels = ['easy', 'easy', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard']
        
        reference_df = pd.DataFrame({
            'task_id': reference_ids,
            'question': reference_questions,
            'level': reference_levels
        })
        print(f"✓ Using fallback reference data with {len(reference_df)} questions")
        return reference_df
    
    except Exception as e:
        print(f"❌ Error loading dev.csv: {e}")
        return None

def main():
    """Main function."""
    
    try:
        clusterer = LLMClusterer()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    df = load_sample_data()
    if df is None:
        return
    
    # Load reference data from dev.csv
    reference_df = load_reference_data()
    if reference_df is None:
        return
    
    print(f"Processing {len(df)} questions from all.csv using {len(reference_df)} reference questions from dev.csv")
    
    # For debugging, you can limit to a small number first
    # Uncomment the next line to test with just 5 questions
    # df = df.head(5)
    
    # Apply clustering to all questions
    df_clustered = clusterer.apply_clustering(df, reference_df)
    
    # Save results
    output_file = 'clustered_llm_based.csv'
    df_clustered.to_csv(output_file, index=False)
    print(f"✓ Results saved to {output_file}")
    
    # Show sample results
    print("\nSample results:")
    for i, row in df_clustered.head(3).iterrows():
        print(f"Q{i+1}: {row['question'][:60]}...")
        print(f"    Similar to: {row['referred']}")
        print()

if __name__ == '__main__':
    main()
