import openai
from dotenv import load_dotenv
import os

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_entities_and_relationships(text):
    prompt = f"""
Extract all relevant entities and relationships from the following text. 
Return a JSON object with two fields: 
- "entities": a list of unique names
- "relationships": a list of triples in the form:
  {{
    "source": "Entity A",
    "relation": "relation_type",
    "target": "Entity B",
    "extra": {{ optional extra info like date, amount, etc }}
  }}

TEXT:
\"\"\"
{text}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        return eval(content)
    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        return {
            "entities": [],
            "relationships": []
        }
