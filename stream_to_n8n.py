import csv
import json
import urllib.request
import sys

# --- CONFIGURATION ---
CSV_FILE = 'hx.csv'  # UPDATE THIS to your actual filename
# UPDATE THIS to your Production Webhook URL
WEBHOOK_URL = 'https://cbook.app.n8n.cloud/webhook/f2685af3-bed5-4be6-a6ec-7e5ac1806c15'
PAYLOAD_COPY = 'hx_payload.json'  # Copy of payload sent to n8n 

def main():
    print(f"Reading {CSV_FILE}...")
    
    all_rows = []
    
    try:
        # Read the entire file into memory
        # encoding='utf-8-sig' handles Excel BOM if present
        with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                
        count = len(all_rows)
        print(f"Loaded {count} rows.")
        
        if count == 0:
            print("Error: CSV file is empty!")
            return

        # Prepare the payload
        # We wrap the list in a "data" key so n8n receives 1 item containing the list
        payload = {"data": all_rows}
        json_bytes = json.dumps(payload).encode('utf-8')

        # Save a copy of the payload
        with open(PAYLOAD_COPY, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"Saved payload copy to {PAYLOAD_COPY}")

        print(f"Sending payload ({len(json_bytes)/1024/1024:.2f} MB) to n8n...")
        
        req = urllib.request.Request(WEBHOOK_URL, data=json_bytes, method='POST')
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req) as response:
            print(f"Success! n8n received the full dataset. Status: {response.status}")
            
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
