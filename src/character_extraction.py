import json
import re
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

COMMON_PRONOUNS = {"He","She","His","Her","They","We","I","It"}
COMMON_WORDS = {
    "The","A","An","Next","Ow","No","Stop","That","This",
    "Both","And","But","Today","There","They"
}

def extract_characters_from_text(text, dialogues=None):
    """
    Extract character names using:
    1. SpaCy PERSON entities
    2. Dialogue attributions (e.g., 'said the Teacher', 'told her Mom')
    """
    doc = nlp(text)
    names = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.add(ent.text.strip())

    attribution_patterns = [
        r'said the ([A-Za-z]+)',
        r'said ([A-Za-z]+)',
        r'told her ([A-Za-z]+)',
        r'told him ([A-Za-z]+)',
        r'replied ([A-Za-z]+)',
    ]
    for pattern in attribution_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            names.add(match.strip().capitalize())

    if dialogues:
        dialogue_text = " ".join(dialogues)
        names = {name for name in names if name in dialogue_text or name in ["Mom", "Teacher"]}

    names = {name for name in names if name not in COMMON_PRONOUNS and name not in COMMON_WORDS}

    return sorted(list(names))

def extract_characters(story_json):
    story_text = story_json.get("story", "")
    dialogues = story_json.get("dialogues", [])
    characters = extract_characters_from_text(story_text, dialogues)
    story_json["characters"] = characters
    return story_json

def extract_characters_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for story in tqdm(data, desc="Extracting characters"):
        processed_story = extract_characters(story)
        processed_data.append(processed_story)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Character extraction completed. Saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/processed/stories_preprocessed.json"
    output_file = "data/processed/stories_with_characters.json"
    extract_characters_dataset(input_file, output_file)