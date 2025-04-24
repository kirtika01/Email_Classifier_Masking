import requests
import json

def test_classify_endpoint():
    url = "http://localhost:8000/classify"
    
    test_emails = [
        {
            "email_body": "Hi, my name is Alice Smith and my email is alice.smith@email.com. I need help with login."
        },
        {
            "email_body": "Technical issue: Server 10.0.0.1 is down. Contact Bob Johnson at +1-555-0123 for details."
        },
        {
            "email_body": "Please update my shipping address. My customer ID is 123-45-6789."
        }
    ]
    
    print("\nTesting /classify endpoint...")
    print("-" * 50)
    
    for i, email in enumerate(test_emails, 1):
        try:
            response = requests.post(url, json=email)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\nTest Case {i}:")
                print("Original Email:", result["input_email_body"])
                print("\nMasked Email:", result["masked_email"])
                print("\nDetected Entities:")
                for entity in result["list_of_masked_entities"]:
                    print(f"- Type: {entity['classification']}, "
                          f"Value: {entity['entity']}, "
                          f"Position: {entity['position']}")
                print("\nPredicted Category:", result["category_of_the_email"])
                
            else:
                print(f"\nTest Case {i} failed with status code: {response.status_code}")
                print("Error:", response.text)
                
        except Exception as e:
            print(f"\nTest Case {i} failed with error: {str(e)}")
        
        print("-" * 50)

def test_health_endpoint():
    url = "http://localhost:8000/health"
    
    print("\nTesting /health endpoint...")
    try:
        response = requests.get(url)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    print("Starting API tests...")
    
    test_health_endpoint()
    
    test_classify_endpoint()
    
    print("\nAPI testing complete!")