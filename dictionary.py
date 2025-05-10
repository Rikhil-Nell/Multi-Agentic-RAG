import requests
from pydantic import BaseModel
from typing import List

from settings import Settings

settings = Settings()

class Entry(BaseModel):
    shortdef: List[str]

async def dictionary_api(search_word: str):
    url = f"https://www.dictionaryapi.com/api/v3/references/collegiate/json/{search_word}"
    response = requests.get(url, params={"key": settings.mw_api_key})
    
    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return

    data = response.json()
    if isinstance(data, list) and isinstance(data[0], dict):
        entry = Entry(**data[0])
        print(f"Meaning of the word '{search_word}': \n" + ", ".join(entry.shortdef))
        print("Meaning Recieved")
        return f"Meaning of the word '{search_word}': \n" + ", ".join(entry.shortdef)
    
    elif isinstance(data, list):
        print(f"Suggestions: {data}")
        return f"Suggestions: {data}"
    
    else:
        print("❌ Unexpected response format.")

if __name__ == "__main__":
    dictionary_api("help")
