#!/usr/bin/env python3
"""
Test API endpoints for the Content Ingestion Pipeline
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test a single API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n[TEST] {method} {endpoint}")
    if description:
        print(f"       {description}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=5)
        elif method == "DELETE":
            response = requests.delete(url, timeout=5)
        
        print(f"       Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if isinstance(result, dict) and len(result) < 5:
                    print(f"       Response: {result}")
                else:
                    print(f"       Response: [JSON data - {len(str(result))} chars]")
                return True, result
            except:
                print(f"       Response: [Non-JSON - {len(response.text)} chars]")
                return True, response.text
        else:
            print(f"       Error: {response.text[:200]}...")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print("       [ERROR] Server not running or not accessible")
        return False, None
    except requests.exceptions.Timeout:
        print("       [ERROR] Request timed out")
        return False, None
    except Exception as e:
        print(f"       [ERROR] {e}")
        return False, None

def main():
    """Test all pipeline API endpoints"""
    print("API ENDPOINT TESTING")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("\n[1] Server Health Check")
    success, _ = test_endpoint("/api/status", description="Check if FastAPI server is running")
    
    if not success:
        print("\n[CRITICAL] Server not running!")
        print("Please start the server first:")
        print("  myenv/Scripts/python.exe app.py")
        return
    
    print("[OK] Server is running!")
    
    # Test 2: Pipeline Status
    print("\n[2] Pipeline Status")
    success, status = test_endpoint("/api/pipeline/status", description="Get pipeline status")
    
    # Test 3: Pipeline Configuration
    print("\n[3] Pipeline Configuration")
    test_endpoint("/api/pipeline/config", description="Get pipeline configuration")
    
    # Test 4: Monitored Sources
    print("\n[4] Monitored Sources")
    success, sources = test_endpoint("/api/pipeline/sources", description="List monitored sources")
    
    # Test 5: Start Pipeline
    print("\n[5] Start Pipeline")
    test_endpoint("/api/pipeline/start", method="POST", description="Start the pipeline")
    
    time.sleep(2)  # Give it time to start
    
    # Test 6: Check if pipeline is running
    print("\n[6] Verify Pipeline Started")
    success, status = test_endpoint("/api/pipeline/status", description="Check if pipeline is now running")
    
    if success and status:
        is_running = status.get('is_running', False)
        print(f"       Pipeline running: {is_running}")
        
        if is_running:
            print("[SUCCESS] Pipeline is running!")
            
            # Test 7: Recent Items
            print("\n[7] Recent Items")
            test_endpoint("/api/pipeline/recent-items", description="Get recently processed items")
            
            # Test 8: Add a PDF source
            print("\n[8] Add PDF Source")
            new_source = {
                "name": "api_test_source",
                "path_or_url": "api_test_pdfs/",
                "enabled": True,
                "recursive": True
            }
            test_endpoint("/api/pipeline/sources/pdf", method="POST", data=new_source, 
                         description="Add new PDF directory source")
            
            # Test 9: List sources again
            print("\n[9] Updated Sources List")
            test_endpoint("/api/pipeline/sources", description="List sources after adding new one")
            
            # Test 10: Create test directory and file
            print("\n[10] File Detection Test")
            test_dir = Path("api_test_pdfs")
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / "api_test_document.pdf"
            with open(test_file, 'w') as f:
                f.write(f"API test document created at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print(f"        Created test file: {test_file}")
            print("        Waiting 5 seconds for pipeline to detect...")
            time.sleep(5)
            
            # Check for processing
            test_endpoint("/api/pipeline/recent-items", description="Check if file was detected")
            
            # Test 11: Stop Pipeline
            print("\n[11] Stop Pipeline")
            test_endpoint("/api/pipeline/stop", method="POST", description="Stop the pipeline")
            
            # Cleanup
            print("\n[CLEANUP]")
            try:
                test_file.unlink()
                test_dir.rmdir()
                print("        Removed test files")
            except Exception as e:
                print(f"        Cleanup warning: {e}")
        else:
            print("[WARNING] Pipeline failed to start")
    
    # Test 12: Health Check
    print("\n[12] Pipeline Health")
    test_endpoint("/api/pipeline/health", description="Pipeline health check")
    
    print("\n" + "=" * 50)
    print("API ENDPOINT TESTING COMPLETE")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")