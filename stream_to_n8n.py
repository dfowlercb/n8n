import csv
import json
import urllib.request
import sys

# --- CONFIGURATION ---
CSV_FILE = 'hx.csv'
WEBHOOK_URL = 'https://cbook.app.n8n.cloud/webhook/f2685af3-bed5-4be6-a6ec-7e5ac1806c15'
PAYLOAD_COPY = 'hx_payload.json'


def main():
    print(f"Reading {CSV_FILE}...")

    all_rows = []

    try:
        # Read the entire file into memory
        with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

        count = len(all_rows)
        print(f"Loaded {count} rows.")

        if count == 0:
            print("Error: CSV file is empty!")
            return

        # Build payload
        payload = {"data": all_rows}
        json_str = json.dumps(payload)
        json_bytes = json_str.encode('utf-8')
        
        payload_mb = len(json_bytes) / 1024 / 1024
        print(f"Payload size: {payload_mb:.2f} MB")

        # Save a copy of the payload
        with open(PAYLOAD_COPY, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"Saved payload copy to {PAYLOAD_COPY}")

        # Send ALL data in a single request
        print(f"Sending {count} rows to n8n...")
        
        req = urllib.request.Request(WEBHOOK_URL, data=json_bytes, method='POST')
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req, timeout=120) as response:
            status = response.status
            print(f"OK (status {status})")

        print(f"\nSuccess! All {count} rows sent to n8n in a single request.")

    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
