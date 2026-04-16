import json
import os
import base64
import re
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from PIL import Image
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from io import BytesIO
from typing import List, Optional
import logging
logging.basicConfig(level=logging.INFO)

logging.info("Extracting ingredients from image...")

# Load environment variables (e.g., API keys)
WATSONX_API_KEY = os.environ.get('WATSONX_API_KEY')
WATSONX_URL = os.environ.get('WATSONX_URL')
WATSONX_PROJECT_ID = os.environ.get('WATSONX_PROJECT_ID')

credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

class ExtractIngredientsTool():
    @tool("Extract ingredients")
    def extract_ingredient(image_input: str):
        """
        Extract ingredients from a food item image.
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A list of ingredients extracted from the image.
        """
        if image_input.startswith("http"):  # Check if input is a URL
            # Download the image from the URL
            response = requests.get(image_input)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
        else:
            # Open the local image file in binary mode
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"No file found at path: {image_input}")
            with open(image_input, "rb") as file:
                image_bytes = BytesIO(file.read())

        # Encode the image to a base64 string
        encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

        # Call the model with the encoded image
        model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={"max_tokens": 300},
        )
        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract ingredients from the food item image"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                    ],
                }
            ]
        )

        return response['choices'][0]['message']['content']


class FilterIngredientsTool:
    @tool("Filter ingredients")
    def filter_ingredients(raw_ingredients: str) -> List[str]:
        """
        Processes the raw ingredient data and filters out non-food items or noise.
        
        :param raw_ingredients: Raw ingredients as a string.
        :return: A list of cleaned and relevant ingredients.
        """
        # Example implementation: parse the raw ingredients string into a list
        # This can be enhanced with more sophisticated parsing as needed
        ingredients = [ingredient.strip().lower() for ingredient in raw_ingredients.split(',') if ingredient.strip()]
        return ingredients

class DietaryFilterTool:
    @tool("Filter based on dietary restrictions")
    def filter_based_on_restrictions(ingredients: List[str], dietary_restrictions: Optional[str] = None) -> List[str]:
        """
        Uses an LLM model to filter ingredients based on dietary restrictions.

        :param ingredients: List of ingredients.
        :param dietary_restrictions: Dietary restrictions (e.g., vegan, gluten-free). Defaults to None.
        :return: Filtered list of ingredients that comply with the dietary restrictions.
        """
        # If no dietary restrictions are provided, return the original ingredients
        if not dietary_restrictions:
            return ingredients

        # Initialize the WatsonX model
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={"max_tokens": 150},
        )

        # Create a prompt for the LLM to filter ingredients
        prompt = f"""
        You are an AI nutritionist specialized in dietary restrictions. 
        Given the following list of ingredients: {', '.join(ingredients)}, 
        and the dietary restriction: {dietary_restrictions}, 
        remove any ingredient that does not comply with this restriction. 
        Return only the compliant ingredients as a comma-separated list with no additional commentary.
        """

        # Send the prompt to the model for filtering
        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                }
            ]
        )

        # Parse the response to return the filtered list
        filtered = response['choices'][0]['message']['content'].strip().lower()
        filtered_list = [item.strip() for item in filtered.split(',') if item.strip()]
        return filtered_list

    
class NutrientAnalysisTool():
    @staticmethod
    def _extract_json_payload(raw_text: str) -> dict:
        """Parse model output into JSON, tolerating markdown fences and extra text."""
        if not raw_text:
            return {}

        text = raw_text.strip()

        # Remove markdown code fences if present.
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

        return {}

    @staticmethod
    def _run_model_chat(model_id: str, messages: list, max_tokens: int = 500) -> str:
        """Run a Watsonx chat call and return the textual content."""
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={"max_tokens": max_tokens},
        )
        response = model.chat(messages=messages)
        return response['choices'][0]['message']['content']

    @tool("Analyze nutritional values and calories of the dish from uploaded image")
    def analyze_image(image_input: str):
        """
        Provide a detailed nutrient breakdown and estimate the total calories of all ingredients from the uploaded image.
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A string with nutrient breakdown (protein, carbs, fat, etc.) and estimated calorie information.
        """
        if image_input.startswith("http"):  # Check if input is a URL
            # Download the image from the URL
            response = requests.get(image_input)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
        else:
            # Open the local image file in binary mode
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"No file found at path: {image_input}")
            with open(image_input, "rb") as file:
                image_bytes = BytesIO(file.read())

        # Encode the image to a base64 string
        encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

        # Phase 1: Vision-only raw observation extraction (no dish naming).
        phase_1_prompt = """
        You are a food-vision observer. Describe ONLY what is visually present in the image.
        Do NOT infer or mention any dish name.

        Return JSON only with this schema:
        {
          "dominant_colors": "...",
          "textures": "...",
          "proteins_visible": "...",
          "cooking_style": "...",
          "vessel": "...",
          "garnishes": "...",
          "additional_visual_cues": ["..."]
        }
        """
        phase_1_raw = NutrientAnalysisTool._run_model_chat(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": phase_1_prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                    ],
                }
            ],
        )
        phase_1_features = NutrientAnalysisTool._extract_json_payload(phase_1_raw)

        # Phase 2: Text-only dish inference and grounding context.
        phase_2_prompt = f"""
        Given ONLY these visual observations (JSON), infer the most likely dish.
        Observations:
        {json.dumps(phase_1_features, ensure_ascii=True)}

        Explain the reasoning concisely.
        Return JSON only with this schema:
        {{
          "dish": "...",
          "cuisine": "...",
          "category": "...",
          "confidence": "high|medium|low",
          "reasoning": "..."
        }}
        """
        phase_2_raw = NutrientAnalysisTool._run_model_chat(
            model_id="ibm/granite-3-8b-instruct",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": phase_2_prompt}
                    ],
                }
            ],
        )
        phase_2_inference = NutrientAnalysisTool._extract_json_payload(phase_2_raw)

        inferred_dish = phase_2_inference.get("dish", "Unknown dish")
        inferred_cuisine = phase_2_inference.get("cuisine", "Unknown cuisine")
        inferred_category = phase_2_inference.get("category", "Unknown category")

        # Phase 3: Grounded nutrition analysis using image + inferred dish context.
        phase_3_prompt = f"""
        You are an expert nutritionist.

        Ground your analysis using:
        - inferred dish name: {inferred_dish}
        - inferred cuisine: {inferred_cuisine}
        - inferred category: {inferred_category}
        - visual observations: {json.dumps(phase_1_features, ensure_ascii=True)}
        - inference reasoning: {phase_2_inference.get('reasoning', '')}

        Return JSON only in this exact schema:
        {{
          "dish": "string",
          "portion_size": "string",
          "estimated_calories": 0,
          "nutrients": {{
            "protein": "string",
            "carbohydrates": "string",
            "fats": "string",
            "vitamins": [{{"name": "string", "percentage_dv": "string"}}],
            "minerals": [{{"name": "string", "amount": "string"}}]
          }},
          "health_evaluation": "string"
        }}

        Keep estimates realistic and concise.
        """
        phase_3_raw = NutrientAnalysisTool._run_model_chat(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            max_tokens=700,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": phase_3_prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                    ],
                }
            ],
        )

        phase_3_analysis = NutrientAnalysisTool._extract_json_payload(phase_3_raw)

        # Ensure core fields are present even when model output is partially structured.
        if not phase_3_analysis:
            phase_3_analysis = {
                "dish": inferred_dish,
                "portion_size": "Not specified",
                "estimated_calories": 0,
                "nutrients": {
                    "protein": "Not specified",
                    "carbohydrates": "Not specified",
                    "fats": "Not specified",
                    "vitamins": [],
                    "minerals": []
                },
                "health_evaluation": "Nutrition details could not be fully parsed from model output."
            }

        phase_3_analysis.setdefault("dish", inferred_dish)
        phase_3_analysis.setdefault("portion_size", "Not specified")
        phase_3_analysis.setdefault("estimated_calories", 0)
        phase_3_analysis.setdefault("nutrients", {
            "protein": "Not specified",
            "carbohydrates": "Not specified",
            "fats": "Not specified",
            "vitamins": [],
            "minerals": []
        })
        phase_3_analysis.setdefault("health_evaluation", "No health evaluation provided.")

        # Add grounded metadata for downstream consumers while keeping the schema backward compatible.
        phase_3_analysis["cuisine"] = inferred_cuisine
        phase_3_analysis["category"] = inferred_category
        phase_3_analysis["inference_confidence"] = phase_2_inference.get("confidence", "unknown")
        phase_3_analysis["inference_reasoning"] = phase_2_inference.get("reasoning", "")
        phase_3_analysis["visual_features"] = phase_1_features

        return json.dumps(phase_3_analysis, ensure_ascii=True)