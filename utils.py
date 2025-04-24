import spacy
import re
import pandas as pd
from typing import Tuple, List, Dict
import os

nlp = spacy.load("en_core_web_sm")

EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
PHONE_PATTERN = r"\b(?:\+91[\s-]?)?(?:\d{10}|\d{5}[\s-]?\d{5})\b"

FULL_NAME_PATTERN = r"\b(?!(?:January|February|March|April|May|June|July|August|September|October|November|December|Road|Street|Lane|Colony|Sector|Nagar|Shanti|Pradesh|Bhopal|Indraprastha)\b)[A-Z][a-z]+(?: [A-Z][a-z]+)+?\b"
DOB_PATTERN = r"\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b"
AADHAR_PATTERN = r"\b(?<!\d[-\s])\d{4}\s?\d{4}\s?\d{4}\b(?!\s?\d)"

CREDIT_CARD_PATTERN = r"\b(?:4\d{3}|5[1-5]\d{2}|6(?:011|5\d{2})|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
CVV_PATTERN = r"\b(?<!\d[-\s])\d{3,4}\b(?!\d)"
EXPIRY_PATTERN = r"\b(?:0[1-9]|1[0-2])/\d{2}\b"


def validate_credit_card(number: str) -> bool:
    number = "".join(filter(str.isdigit, number))

    if not number.isdigit() or len(number) < 13 or len(number) > 19:
        return False

    total = 0
    reverse = number[::-1]
    for i, digit in enumerate(reverse):
        digit = int(digit)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


def validate_cvv(cvv: str) -> bool:
    cvv = "".join(filter(str.isdigit, cvv))
    return cvv.isdigit() and len(cvv) in [3, 4]


def validate_expiry(expiry: str) -> bool:
    expiry = "".join(filter(str.isdigit, expiry))
    if len(expiry) != 4:
        return False

    month = int(expiry[:2])
    year = int(expiry[2:])

    return 1 <= month <= 12


def load_data(
    file_path: str = "data/combined_emails_with_natural_pii.csv",
) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)


def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    doc = nlp(text)
    masked_text = text
    entities = []
    offset = 0

    ordered_patterns = [
        ("credit_card_no", CREDIT_CARD_PATTERN),
        ("aadhar_num", AADHAR_PATTERN),
        ("expiry_date", EXPIRY_PATTERN),
        ("cvv_no", CVV_PATTERN),
        ("phone_number", PHONE_PATTERN),
        ("email", EMAIL_PATTERN),
        ("dob", DOB_PATTERN),
    ]

    masked_positions = set()
    regex_entities = []

    for entity_type, pattern in ordered_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()

            if all(p in masked_positions for p in range(start, end)):
                continue

            if (
                len(set(range(start, end)).intersection(masked_positions))
                / (end - start)
                > 0.5
            ):
                continue

            matched_text = match.group()

            skip = False
            if entity_type == "credit_card_no" and not validate_credit_card(
                matched_text
            ):
                skip = True

            if entity_type == "cvv_no":
                if not validate_cvv(matched_text):
                    skip = True
                else:
                    for re_ent in regex_entities:
                        if (
                            re_ent["type"] in ["credit_card_no", "aadhar_num"]
                            and start >= re_ent["start"]
                            and end <= re_ent["end"]
                        ):
                            skip = True
                            break
            if entity_type == "expiry_date" and not validate_expiry(matched_text):
                skip = True

            if skip:
                continue

            regex_entities = [
                e
                for e in regex_entities
                if not (e["start"] >= start and e["end"] <= end)
            ]

            regex_entities.append(
                {"start": start, "end": end, "text": matched_text, "type": entity_type}
            )
            masked_positions.update(range(start, end))

    potential_names = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            potential_names.append(
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "type": "full_name",
                    "source": "spacy",
                }
            )

    for match in re.finditer(FULL_NAME_PATTERN, text):
        potential_names.append(
            {
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "type": "full_name",
                "source": "regex",
            }
        )

    name_entities = []
    address_keywords = {
        "road",
        "street",
        "lane",
        "colony",
        "sector",
        "nagar",
        
    }
    for name in potential_names:
        start, end = name["start"], name["end"]

        if any(p in masked_positions for p in range(start, end)):
            continue

        if any(keyword in name["text"].lower() for keyword in address_keywords):
            continue

        preceding_text = text[max(0, start - 10) : start]
        if re.search(r"\d+,\s*$", preceding_text):
            continue

        name_entities.append(name)
        masked_positions.update(range(start, end))

    all_entities = sorted(regex_entities + name_entities, key=lambda x: x["start"])

    for entity in all_entities:
        start = entity["start"] - offset
        end = entity["end"] - offset
        original_text = entity["text"]
        entity_type = entity["type"]

        current_masked_segment = masked_text[start:end]
        if f"[{entity_type}]" in current_masked_segment:
            continue

        entities.append(
            {
                "position": [
                    entity["start"],
                    entity["end"],
                ],
                "classification": entity_type,
                "entity": original_text,
            }
        )

        mask = f"[{entity_type}]"

        masked_text = masked_text[:start] + mask + masked_text[end:]
        offset += len(original_text) - len(mask)

    return masked_text, entities


def preprocess_text(text: str) -> str:
    doc = nlp(text.lower())

    tokens = []
    for token in doc:
        if (
            not token.is_punct
            and not token.is_stop
            and not token.is_space
            and token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
        ):

            lemma = token.lemma_

            if token.ent_type:
                lemma = f"[{token.ent_type_}]"

            tokens.append(lemma)

    processed_text = " ".join(tokens)

    processed_text = re.sub(r"[^\w\s\[\]]", " ", processed_text)
    processed_text = re.sub(r"\s+", " ", processed_text).strip()

    return processed_text
