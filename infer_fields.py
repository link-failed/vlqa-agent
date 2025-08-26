import pandas as pd
import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def load_data_file_info():
    """Load the data file information from JSON"""
    with open('/Users/xinyi/Desktop/vlqa/vlqa/vlqa-agent/data/encode/data_file_info.json', 'r') as f:
        return json.load(f)

def create_field_mapping_prompt(data_info, question):
    """Create a prompt for the LLM to infer input and output fields"""
    
    # Convert data info to a readable format
    field_descriptions = ""
    for file_info in data_info:
        field_descriptions += f"\n{file_info['file']}:\n{file_info['text']}\n"
    
    prompt = f"""
You are an expert data analyst. Given the following data schema and a question, determine:
1. The input fields needed to answer the question
2. The output field that represents the answer

Data Schema:
{field_descriptions}

Question: "{question}"

Rules:
- Input fields should be in format "table.field" (e.g., "payments.merchant", "fees.ID")
- If the output is a direct field from a table, use "table.field" format
- If the output is fee calculation, like average/max/min/sum/delta, return "compute.fee"
- Else if the required output is relevant to aggregation of some field, just return in "table.field" format, like "payments.eur_amount"


Examples:
- For "What is the total fees for merchant X?": {{"input_fields": "payments.merchant", "output_fields": "compute.fee"}}
- For "What fee IDs apply to account type H?": {{"input_fields": "fees.account_type", "output_fields": "fees.ID"}}
- For "What percentage of transactions are credit?": {{"input_fields": "payments.is_credit", "output_fields": "payments.is_credit"}}
- For "In July 2023 what delta would Rafa_AI pay if the relative fee of the fee with ID=276 changed to 1?": {{"input_fields": "payments.merchant, payments.year, payments.day_of_year, fees.ID", "output_fields": "compute.fee"}}
- For "In 2023, which merchants were affected by the Fee with ID 10?": {{"input_fields": "payments.merchant, payments.year, fees.ID", "output_fields": "payments.merchant"}}

Return your answer as a JSON object with exactly this structure:
{{
    "input_fields": "comma-separated list of input fields",
    "output_fields": "comma-separated list of output fields"
}}
"""
    
    return prompt

def infer_fields_for_question(question, data_info):
    """Use OpenAI to infer fields for a single question"""
    prompt = create_field_mapping_prompt(data_info, question)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise data analyst that extracts field mappings from questions. Always respond with valid JSON."},
                {"role": "user", "content": prompt},

            ],
            # temperature=0.1,
            # max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        try:
            result = json.loads(content)
            input_fields = result.get('input_fields', '')
            output_fields = result.get('output_fields', '')
            return input_fields, output_fields
        except json.JSONDecodeError as json_error:
            print(f"Error parsing JSON response: {json_error}")
            print(f"Raw response: {content}")
            return "", ""
    
    except Exception as e:
        print(f"Error processing question: {question}")
        print(f"Error: {e}")
        return "", ""

def process_csv_file():
    """Process the CSV file and fill in missing input_fields and output_fieldss"""
    
    # Load the data file information
    data_info = load_data_file_info()
    
    # Load the CSV file
    csv_path = '/Users/xinyi/Desktop/vlqa/vlqa/vlqa-agent/data/task_cluster/all.csv'
    df = pd.read_csv(csv_path)
    
    # Process rows with missing input_fields or output_fields
    for index, row in df.iterrows():
        if pd.isna(row['input_fields']) or pd.isna(row['output_fields']) or row['input_fields'] == '' or row['output_fields'] == '':
            print(f"Processing task {row['task_id']}: {row['question']}...")
            
            input_fields, output_fields = infer_fields_for_question(row['question'], data_info)
            
            if input_fields and output_fields:
                df.at[index, 'input_fields'] = input_fields
                df.at[index, 'output_fields'] = output_fields
                print(f"  -> Input: {input_fields}")
                print(f"  -> Output: {output_fields}")
            else:
                print(f"  -> Failed to infer fields")
    
    # Save the updated CSV
        output_path = '/Users/xinyi/Desktop/vlqa/vlqa/vlqa-agent/data/task_cluster/all_updated.csv'
        df.to_csv(output_path, index=False)
        print(f"\nUpdated CSV saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    updated_df = process_csv_file()
    
    # # Show a sample of the updated data
    # print("\nSample of updated data:")
    # print(updated_df[['task_id', 'question', 'input_fields', 'output_fields']].head(10).to_string())
