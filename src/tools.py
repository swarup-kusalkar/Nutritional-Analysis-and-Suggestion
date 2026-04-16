import json
import os
import base64
import re
import requests
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
from langchain.tools import tool
from PIL import Image
from io import BytesIO
from typing import List, Optional
import logging
logging.basicConfig(level=logging.INFO)

logging.info("Extracting ingredients from image...")

# Load environment variables (e.g., API keys)
load_dotenv()
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

class ExtractIngredientsTool():
    @tool("Extract ingredients")
    def extract_ingredient(image_input: str):
        """
        Extract ingredients from a food item image using Gemini vision.
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A list of ingredients extracted from the image.
        """
        try:
            if image_input.startswith("http"):
                response = requests.get(image_input, timeout=30)
                response.raise_for_status()
                image_data = response.content
            else:
                if not os.path.isfile(image_input):
                    raise FileNotFoundError(f"No file found at path: {image_input}")
                with open(image_input, "rb") as file:
                    image_data = file.read()

            # Use Gemini vision to extract ingredients
            image_part = {
                "mime_type": "image/jpeg",
                "data": base64.standard_b64encode(image_data).decode("utf-8")
            }

            prompt = "Extract all visible ingredients from this food image. Return as a comma-separated list."
            response = gemini_model.generate_content([prompt, image_part])
            return response.text
        except Exception as e:
            return f"Error extracting ingredients: {str(e)}"


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
        Uses Groq LLM to filter ingredients based on dietary restrictions (fast text inference).

        :param ingredients: List of ingredients.
        :param dietary_restrictions: Dietary restrictions (e.g., vegan, gluten-free). Defaults to None.
        :return: Filtered list of ingredients that comply with the dietary restrictions.
        """
        if not dietary_restrictions:
            return ingredients

        try:
            prompt = f"""You are an AI nutritionist specialized in dietary restrictions. 
Given the following list of ingredients: {', '.join(ingredients)}, 
and the dietary restriction: {dietary_restrictions}, 
remove any ingredient that does not comply with this restriction. 
Return only the compliant ingredients as a comma-separated list with no additional commentary."""

            message = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=150,
            )
            filtered = message.choices[0].message.content.strip().lower()
            filtered_list = [item.strip() for item in filtered.split(',') if item.strip()]
            return filtered_list
        except Exception as e:
            logging.error(f"Error filtering ingredients: {str(e)}")
            return ingredients

    
class NutrientAnalysisTool():
    @staticmethod
    def _validate_api_keys() -> tuple:
        """Validate Gemini and Groq API keys at runtime."""
        gemini_key = os.environ.get('GEMINI_API_KEY')
        groq_key = os.environ.get('GROQ_API_KEY')

        missing = []
        if not gemini_key:
            missing.append("GEMINI_API_KEY")
        if not groq_key:
            missing.append("GROQ_API_KEY")

        if missing:
            raise RuntimeError(
                "Missing API configuration: " + ", ".join(missing) +
                ". Set GEMINI_API_KEY and GROQ_API_KEY in environment or .env file."
            )

        return gemini_key, groq_key

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
    def _run_gemini_vision(prompt: str, image_data: str, max_tokens: int = 500) -> str:
        """Run Gemini Flash with vision capability."""
        try:
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data
            }
            response = gemini_model.generate_content(
                [prompt, image_part],
                generation_config={"max_output_tokens": max_tokens}
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini vision call failed: {str(e)}")

    @staticmethod
    def _run_groq_text(prompt: str, max_tokens: int = 500) -> str:
        """Run Groq for fast text inference (no vision)."""
        try:
            message = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return message.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq text call failed: {str(e)}")

    @tool("Analyze nutritional values and calories of the dish from uploaded image")
    def analyze_image(image_input: str):
        """
        Provide a detailed nutrient breakdown using a 3-phase hybrid approach:
        Phase 1: Gemini Flash vision (raw observations).
        Phase 2: Groq text (fast dish inference).
        Phase 3: Gemini Flash vision (grounded nutrition analysis).
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A JSON string with nutrient breakdown and health evaluation.
        """
        try:
            NutrientAnalysisTool._validate_api_keys()

            # Load and encode image
            if image_input.startswith("http"):
                response = requests.get(image_input, timeout=30)
                response.raise_for_status()
                image_data = response.content
            else:
                if not os.path.isfile(image_input):
                    raise FileNotFoundError(f"No file found at path: {image_input}")
                with open(image_input, "rb") as file:
                    image_data = file.read()

            encoded_image = base64.standard_b64encode(image_data).decode("utf-8")

            # Phase 1: Gemini vision - raw visual observations only
            phase_1_prompt = """You are a food-vision observer. Describe ONLY what is visually present in the image.
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
}"""

            phase_1_raw = NutrientAnalysisTool._run_gemini_vision(
                phase_1_prompt, encoded_image, max_tokens=400
            )
            phase_1_features = NutrientAnalysisTool._extract_json_payload(phase_1_raw)

            # Phase 2: Groq text - fast dish inference from observations (NO vision needed)
            phase_2_prompt = f"""Given ONLY these visual observations (JSON), infer the most likely dish.
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
}}"""

            phase_2_raw = NutrientAnalysisTool._run_groq_text(phase_2_prompt, max_tokens=300)
            phase_2_inference = NutrientAnalysisTool._extract_json_payload(phase_2_raw)

            inferred_dish = phase_2_inference.get("dish", "Unknown dish")
            inferred_cuisine = phase_2_inference.get("cuisine", "Unknown cuisine")
            inferred_category = phase_2_inference.get("category", "Unknown category")

            # Phase 3: Gemini vision - grounded nutrition analysis with inferred context
            phase_3_prompt = f"""You are an expert nutritionist.

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

Keep estimates realistic and concise."""

            phase_3_raw = NutrientAnalysisTool._run_gemini_vision(
                phase_3_prompt, encoded_image, max_tokens=700
            )
            phase_3_analysis = NutrientAnalysisTool._extract_json_payload(phase_3_raw)

            # Ensure core fields are present
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

            # Add grounded metadata
            phase_3_analysis["cuisine"] = inferred_cuisine
            phase_3_analysis["category"] = inferred_category
            phase_3_analysis["inference_confidence"] = phase_2_inference.get("confidence", "unknown")
            phase_3_analysis["inference_reasoning"] = phase_2_inference.get("reasoning", "")
            phase_3_analysis["visual_features"] = phase_1_features

            return json.dumps(phase_3_analysis, ensure_ascii=True)

        except Exception as exc:
            # Return structured fallback instead of throwing tool error
            fallback = {
                "dish": "Unknown dish",
                "portion_size": "Not specified",
                "estimated_calories": 0,
                "nutrients": {
                    "protein": "Not specified",
                    "carbohydrates": "Not specified",
                    "fats": "Not specified",
                    "vitamins": [],
                    "minerals": []
                },
                "health_evaluation": f"Analysis unavailable: {str(exc)}"
            }
            return json.dumps(fallback, ensure_ascii=True)