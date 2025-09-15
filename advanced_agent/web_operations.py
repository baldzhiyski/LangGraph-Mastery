from dotenv import load_dotenv
import os
import requests
from urllib.parse import quote_plus

load_dotenv()

def _make_api_request(url,**kwargs):
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        print(f"API Request failed : {e}")
        return None
    except Exception as e:
        print(f"Unknown error : {e}")
        return None

def serp_search(query,engine="google"):
    if engine== 'google':
        base_url = "https://www.google.com/search"
    elif engine== 'bing':
        base_url = "https://www.bing.com/search"
    else:
        raise Exception(f"Unknown engine : {engine}")

    url = "https://api.brightdata.com/request"
    payload = {
        "zone": "ai_agent",
        "url": f"{base_url}?q={quote_plus(query)}&brd_json=1",
        "format":"raw"
    }

    full_response = _make_api_request(url, json=payload)
    if not full_response:
        return None

    extracted_data = {
        "knowledge": full_response.get("knowledge",{}),
        "organic": full_response.get("organic",{})
    }

    return extracted_data