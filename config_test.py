from dotenv import load_dotenv
import os
import requests

# load .env
load_dotenv()

# ---------------------------------------------------------------------
# Check environment variables
# ---------------------------------------------------------------------
print("=" * 50)
print("Checking environment variables")
print("=" * 50)

required = [ 
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_MODEL"
]

ok = True
missing = []
for var in required:
    value = os.getenv(var)
    if not value:
        ok = False
        missing.append(var)
if not ok:
    print(f"✗ Missing environment variables: {missing}")
    exit()

print("✓ all environment variables loaded successfully.\n")


# ---------------------------------------------------------------------
# Tavily Test
# ---------------------------------------------------------------------
print("=" * 50)
print("Testing Tavily")
print("=" * 50)

try:
    from langchain_tavily import TavilySearch
    
    tavily = TavilySearch(max_results=2)
    results = tavily.invoke("What is LangChain?")
    
    print("✓ Tavily connected")

except Exception as e:
    print("✗ Tavily failed")
    print(e)
    exit()

print()


# ---------------------------------------------------------------------
# Server Access Test
# ---------------------------------------------------------------------
print("=" * 50)
print("Testing the Access to the Server")
print("=" * 50)
base_url = os.environ["OLLAMA_BASE_URL"].rstrip("/")
api_key = os.environ["OLLAMA_API_KEY"]

try:
    response = requests.get(
    f"{base_url}/api/tags",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    models = []
    for model in data.get("models", []):
        models.append(model["name"])
    assert set(models) == set(['translategemma:latest','all-minilm:latest','llama3.1:latest'])
    print("✓ accessed the server")

except Exception as e:
    print("✗ access to server failed: confirm that you are using either eduroam network or VPN access to it")
    exit()

print()

print("=" * 50)
print("Configuration Tests Successfully Completed")
print("=" * 50)
