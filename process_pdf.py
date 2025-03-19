import requests
import json

# Replace with the path to your PDF file
pdf_path = "uploads/your_document.pdf"
# A friendly name to identify the source (will be shown in citations)
source_name = "NAU Course Catalog 2024"

# Call the admin endpoint
response = requests.post(
    'http://localhost:5000/api/admin/process-pdf',
    json={
        'pdf_path': pdf_path,
        'source_name': source_name
    }
)

# Print the response
print(json.dumps(response.json(), indent=2))