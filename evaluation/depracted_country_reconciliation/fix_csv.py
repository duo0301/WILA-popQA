import pandas as pd
import re

# Function to clean the CSV file
def fix_csv():
    # Read the CSV file
    input_file = "inputs/qwen/property_pob_Qwen2.5-7B-Instruct.csv"
    output_file = "inputs/qwen/property_pob_Qwen2.5-7B-Instruct_fixed.csv"
    
    # Read the file with error handling
    df = pd.read_csv(input_file, encoding='utf-8-sig', on_bad_lines='skip')
    
    # Clean the llm_output column
    def clean_output(text):
        if pd.isna(text):
            return text
        
        # Convert to string
        text = str(text)
        
        # Remove newlines
        text = text.replace('\n', ' ')
        
        # Extract content from brackets if present
        bracket_match = re.search(r'\[(.*?)\]', text)
        if bracket_match:
            return f"[{bracket_match.group(1).strip()}]"
        
        # If no brackets, just return the text (limited to first 50 chars)
        return text[:50].strip()
    
    # Apply the cleaning function
    df['llm_output'] = df['llm_output'].apply(clean_output)
    
    # Save the cleaned file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"CSV file cleaned and saved to {output_file}")

# Run the function
fix_csv()