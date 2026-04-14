import json
from playwright.sync_api import sync_playwright

def get_guest_token():
    token = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Listen to responses from GraphQL
        def handle_request(request):
            nonlocal token
            # We are looking for bearer tokens in requests to GraphQL
            if "graphql.uzum.uz" in request.url:
                auth = request.headers.get("authorization")
                if auth and auth.startswith("Bearer "):
                    token = auth.split(" ")[1]

        page.on("request", handle_request)
        
        print("Visiting Uzum...")
        # Initialize the target page
        try:
            # wait_until="commit" is fast, we don't need fully loaded page, just the API calls
            page.goto("https://uzum.uz/uz", wait_until="commit", timeout=15000)
            
            # Additional wait until the token is captured or timeout
            for _ in range(20):
                if token:
                    break
                page.wait_for_timeout(500)
        except Exception as e:
            print(f"Navigation soft timeout/error: {e}")
        
        browser.close()
    return token

if __name__ == "__main__":
    t = get_guest_token()
    if t:
        print(f"SUCCESS_TOKEN={t}")
    else:
        print("FAILED_TO_GET_TOKEN")
