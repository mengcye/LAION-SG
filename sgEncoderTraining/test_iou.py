import json
import re
import base64
import requests
from PIL import Image

# Define paths
image_path = "desert_house.jpg"  # Path to your uploaded image

# Define the predefined scene graph for the image (based on the desert house with mountains)
predefined_triples = [
    {"item1": "house", "relation": "near", "item2": "water"},
    {"item1": "house", "relation": "made_of", "item2": "stone"},
    {"item1": "mountain", "relation": "behind", "item2": "house"},
    {"item1": "rock", "relation": "near", "item2": "water"},
    {"item1": "bush", "relation": "around", "item2": "house"}
]

def convert_predefined_triples_to_tuples(triples):
    """
    Convert the predefined triples format to tuple format
    """
    return [(triple["item1"], triple["relation"], triple["item2"]) for triple in triples]

def get_response(image_path, unique_items_list, unique_relations_list):
    """
    Get scene graph relations from an image using GPT-4o API
    """
    # OpenAI API Key
    api_key = ""

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    system_prompt = f''' Please extract the scene graph of the given image. 
    The scene graph just needs to include the relations of the salient objects and exclude the background. 
    The scene graph should be a list of triplets like ["subject", "predicate", "object"].
    Both subject and object should be selected from the following list: {unique_items_list}.
    The predicate should be selected from the following list: {unique_relations_list}.
    Besides the scene graph, please also output the objects list in the image like ["object1", "object2", ..., "object"].
    The object should be also selected from the above-mentioned object list. The output should only contain the scene graph and the object list.

    Return the results in the following JSON format:
    {{
        "scene_graph": [
            ["subject", "predicate", "object"],
            ...
        ],
        "object_list": [
            "object1", "object2", ...
        ],
        "predicate_list":[
            "predicate1", "predicate2", ...
        ]
    }}'''

    payload = {
        "model": "gpt-4o",
        "temperature": 0.7,
        "seed": 1234,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    }

    session = requests.Session()
    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            print("Trying API call...")
            response = session.post("https://api.openai.com/v1/chat/completions",
                                    headers=headers,
                                    json=payload,
                                    verify=False)  # Note: verify=False should be used with caution
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content']

            if result:
                break

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            retry_count += 1

    if retry_count == max_retries:
        raise Exception("Reached maximum retry attempts.")
    else:
        return result

def clean_json(gpt_response):
    """
    Clean and parse JSON response from GPT-4o
    """
    # Remove Markdown code blocks if present
    cleaned_json_str = re.sub(r'```json|```', '', gpt_response).strip()

    # Parse JSON
    parsed_json = json.loads(cleaned_json_str)

    # Extract data
    scene_graph = parsed_json.get("scene_graph", [])
    object_list = parsed_json.get("object_list", [])
    predicate_list = parsed_json.get("predicate_list", [])

    # De-duplicate
    unique_scene_graph = list(set(tuple(triple) for triple in scene_graph))
    unique_object_list = list(set(object_list))
    unique_predicate_list = list(set(predicate_list))

    return unique_scene_graph, unique_object_list, unique_predicate_list

def calculate_iou(list_full, list_mini):
    """
    Calculate Intersection over Union (IoU) between two lists
    """
    # Convert lists to sets
    set_full = set(list_full)
    set_mini = set(list_mini)

    # Calculate intersection
    intersection = set_full.intersection(set_mini)

    # Calculate union
    union = set_full.union(set_mini)

    # Calculate IoU
    iou = len(intersection) / len(union) if len(union) != 0 else 0

    return iou

def main():
    # Load the image
    try:
        img = Image.open(image_path)
        print(f"Successfully loaded image from {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Extract unique items and relations from predefined triples
    unique_items = set()
    unique_relations = set()

    for triple in predefined_triples:
        unique_items.add(triple["item1"])
        unique_relations.add(triple["relation"])
        unique_items.add(triple["item2"])

    unique_items_list = list(unique_items)
    unique_relations_list = list(unique_relations)

    # Convert predefined triples to the tuple format for comparison
    predefined_sg_tuples = convert_predefined_triples_to_tuples(predefined_triples)

    print("Predefined Scene Graph:", predefined_sg_tuples)
    print("Unique Items:", unique_items_list)
    print("Unique Relations:", unique_relations_list)

    # Get relations from the image using GPT-4o (real API call)
    result = get_response(image_path, unique_items_list, unique_relations_list)

    # Print the raw response for debugging
    print("\nRaw GPT-4o Response:")
    print(result)

    # Clean and parse the response
    scene_graph, object_list, predicate_list = clean_json(result)

    print("\nDetected Scene Graph:", scene_graph)
    print("Detected Objects:", object_list)
    print("Detected Predicates:", predicate_list)

    # Calculate IoU metrics
    iou_sg = calculate_iou(predefined_sg_tuples, scene_graph)
    iou_items = calculate_iou(unique_items_list, object_list)
    iou_relations = calculate_iou(unique_relations_list, predicate_list)

    print("\nIoU Metrics:")
    print(f"Scene Graph IoU: {iou_sg:.4f}")
    print(f"Items IoU: {iou_items:.4f}")
    print(f"Relations IoU: {iou_relations:.4f}")


if __name__ == "__main__":
    main()