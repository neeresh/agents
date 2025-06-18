import os
from clarifai.client.user import User
from dotenv import load_dotenv

load_dotenv()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
assert CLARIFAI_PAT and "YOUR_CLARIFAI_PAT" not in CLARIFAI_PAT, f"PAT issue: {CLARIFAI_PAT}"

client = User(user_id="nperla", pat=CLARIFAI_PAT)
app = client.create_app(app_id="mcp-examples", base_workflow="General")

print("✅ Created app:", app.id)
