import os
from dotenv import load_dotenv
load_dotenv()

# OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_VERSION_O4_MINI = os.getenv("AZURE_OPENAI_API_VERSION_O4_MINI")

# Deployment Models
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT_GPT4o = os.getenv("CHAT_DEPLOYMENT_GPT4o")
CHAT_DEPLOYMENT_O4_MINI = os.getenv("CHAT_DEPLOYMENT_O4_MINI")

# Azure Cognitive Search Configuration
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
VECTOR_FIELD = os.getenv("VECTOR_FIELD")

# Logging Configuration
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Feedback Configuration
FEEDBACK_DIR = os.getenv("FEEDBACK_DIR", "feedback_data")

# --- Database Configuration ---
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_SSL_MODE = os.getenv("POSTGRES_SSL_MODE", "require")

def get_cost_rates(model: str) -> dict:
    """
    Get the cost rates for a given model.
    Args:
        model: The name of the model (e.g., 'gpt-4o', 'o4_mini').
    Returns:
        A dictionary with the prompt and completion cost rates.
    """
    model_upper = model.upper().replace("-", "_")
    prompt_rate = float(os.getenv(f"{model_upper}_PROMPT_COST_PER_1K", 0.0))
    completion_rate = float(os.getenv(f"{model_upper}_COMPLETION_COST_PER_1K", 0.0))
    return {"prompt": prompt_rate, "completion": completion_rate}
