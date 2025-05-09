import requests
from pydantic import BaseModel
from typing import List

class Entry(BaseModel):
    shortdef: List[str]

def test_mw_api(key: str, word: str = "test"):
    url = f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}"
    response = requests.get(url, params={"key": key})
    
    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return

    data = response.json()
    if isinstance(data, list) and isinstance(data[0], dict):
        entry = Entry(**data[0])
        print(f"Meaning of the word '{word}': \n" + ", ".join(entry.shortdef))
    elif isinstance(data, list):
        print(f"Suggestions: {data}")
    else:
        print("❌ Unexpected response format.")

if __name__ == "__main__":
    test_mw_api("840734a9-c9eb-432e-8f03-32a4d9c06ac8")
