import csv
import json
import urllib.request
import sys
import time

# --- CONFIGURATION ---
CSV_FILE = 'hx.csv'  # UPDATE THIS to your actual filename
# UPDATE THIS to your Production Webhook URL
WEBHOOK_URL = 'https://cbook.app.n8n.cloud/webhook/f2685af3-bed5-4be6-a6ec-7e5ac1806c15'
PAYLOAD_COPY = 'hx_payload.json'  # Copy of payload sent to n8n
BATCH_SIZE = 1000  # Number of rows per batch


def send_batch(rows, batch_num, total_batches):
    """Send a batch of rows to n8n."""
    payload = {"data": rows, "batch": batch_num, "total_batches": total_batches}
    json_bytes = json.dumps(payload).encode('utf-8')

    req = urllib.request.Request(WEBHOOK_URL, data=json_bytes, method='POST')
    req.add_header('Content-Type', 'application/json')

    with urllib.request.urlopen(req) as response:
        return response.status


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

        # Save a copy of the full payload
        payload = {"data": all_rows}
        with open(PAYLOAD_COPY, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"Saved payload copy to {PAYLOAD_COPY}")

        # Calculate batches
        total_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Sending {count} rows in {total_batches} batches of up to {BATCH_SIZE} rows each...\n")

        # Send in batches
        for i in range(total_batches):
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, count)
            batch_rows = all_rows[start:end]
            batch_num = i + 1

            batch_size_mb = len(json.dumps({"data": batch_rows}).encode('utf-8')) / 1024 / 1024
            print(f"Batch {batch_num}/{total_batches}: rows {start+1}-{end} ({batch_size_mb:.2f} MB)...", end=" ")

            status = send_batch(batch_rows, batch_num, total_batches)
            print(f"OK (status {status})")

            # Small delay between batches to avoid overwhelming n8n
            if batch_num < total_batches:
                time.sleep(0.5)

        print(f"\nSuccess! All {count} rows sent to n8n in {total_batches} batches.")

    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
