
from pyngrok import ngrok

# Set your ngrok auth token
ngrok.set_auth_token("2vPA7SKN47Vsb9T0Zx9uexr72RT_5Ha8R9r49Trs4ZLsnBUTE")

# Open tunnel to port 8000 (where api.py is already running)
tunnel = ngrok.connect(8000)
public_url = tunnel.public_url
print(f"\n--- EXPLAIN API LIVE AT: {public_url} ---")
print(f"Use this URL in your frontend: {public_url}/explain\n")

# Keep the tunnel alive
input("Press Enter to close the tunnel...\n")