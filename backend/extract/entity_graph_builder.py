import openai
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_entities_and_relationships(text: str):
    """
    Use OpenAI to extract entities and relationships from text in a structured format.
    Returns a dictionary with "entities" and "relationships".
    """
    prompt = f"""
You are an expert at understanding technical documents. Extract a list of distinct entities and their relationships from the following text. 
Format your response as JSON with two keys: "entities" (a list of unique string names) and "relationships" 
(a list of dictionaries with "source", "relation", "target", and optional "extra").

Text:
\"\"\"
{text}
\"\"\"
Only return JSON. Example format:
{{
  "entities": ["Apple", "Beats"],
  "relationships": [
    {{"source": "Apple", "relation": "acquired", "target": "Beats", "extra": {{"year": "2014"}}}}
  ]
}}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You extract entities and relationships from technical content.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()

        # Safely evaluate JSON using standard library
        import json

        graph = json.loads(content)

        # Fallback structure if keys are missing
        entities = graph.get("entities", [])
        relationships = graph.get("relationships", [])
        return {"entities": entities, "relationships": relationships}

    except Exception as e:
        print("❌ LLM extraction failed:", e)
        return {"entities": [], "relationships": []}
