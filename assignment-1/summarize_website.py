import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
SYSTEM_PROMPT = (
    "You are an assistant that analyzes the contents of a website "
    "and provides a short summary, ignoring text that might be navigation related. "
    "Respond in markdown."
)

# Load environment variables
def load_api_key():
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    validate_api_key(api_key)
    return api_key

def validate_api_key(api_key):
    if not api_key:
        raise ValueError("No API key was found. Please check your .env file.")
    if not api_key.startswith("sk-proj-"):
        raise ValueError("API key does not start with 'sk-proj-'. Please check your key.")
    if api_key.strip() != api_key:
        raise ValueError("API key contains leading or trailing spaces. Please fix it.")
    print("API key found and looks good.")

# Website class
class Website:
    def __init__(self, url):
        self.url = url
        self.title, self.text = self._fetch_website_content()

    def _fetch_website_content(self):
        response = requests.get(self.url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else "No title found"
        self._remove_irrelevant_tags(soup)
        text = soup.body.get_text(separator="\n", strip=True)
        return title, text

    @staticmethod
    def _remove_irrelevant_tags(soup):
        for tag in soup.body(["script", "style", "img", "input"]):
            tag.decompose()

# OpenAI interaction
class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key=api_key)

    def summarize(self, website):
        messages = self._create_messages(website)
        response = self.client.chat.completions.create(
            model="llama3.2",
            messages=messages
        )
        return response.choices[0].message.content

    @staticmethod
    def _create_messages(website):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": OpenAIClient._create_user_prompt(website)}
        ]

    @staticmethod
    def _create_user_prompt(website):
        return (
            f"You are looking at a website titled {website.title}\n"
            "The contents of this website are as follows; "
            "please provide a short summary of this website in markdown. "
            "If it includes news or announcements, then summarize these too.\n\n"
            f"{website.text}"
        )

# Display summary
def display_summary(url, api_key):
    website = Website(url)
    openai_client = OpenAIClient(api_key)
    summary = openai_client.summarize(website)
    display(Markdown(summary))

# Main execution
if __name__ == "__main__":
    try:
        api_key = load_api_key()
        display_summary("https://edwarddonner.com", api_key)
    except ValueError as e:
        print(e)