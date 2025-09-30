import json
import re
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    """
    Use spaCy sentence tokenizer for accurate sentence splitting.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def extract_dialogues(text):
    """
    Extract all text inside double quotes.
    """
    dialogues = re.findall(r'"(.*?)"', text)
    return [d.strip() for d in dialogues if d.strip()]

def preprocess_story(story_json):
    """
    Input: single story JSON object
    Output: story JSON with 'sentences' and 'dialogues' added
    """
    story_text = story_json.get("story", "")
    sentences = split_sentences(story_text)
    dialogues = extract_dialogues(story_text)
    
    story_json["sentences"] = sentences
    story_json["dialogues"] = dialogues
    return story_json

def preprocess_dataset(input_file, output_file):
    """
    Process all stories in the dataset
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for story in tqdm(data, desc="Processing stories"):
        processed_story = preprocess_story(story)
        processed_data.append(processed_story)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/raw/stories.json"        
    output_file = "data/processed/stories_preprocessed.json"
    preprocess_dataset(input_file, output_file)
