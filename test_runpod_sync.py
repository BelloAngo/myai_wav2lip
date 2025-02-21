import runpod
import os
import json

runpod.api_key = "AB8VTORYC3FJ40SFG9DASLOCCP87GPWJPJ686CMZ"

endpoint = runpod.Endpoint("za2iqcbvgiufvk")

try:
    #Read the test_input.json as dictionary
    with open("test_input.json", "r") as f:
        input_data = json.load(f)
    
    
    
    run_request = endpoint.run_sync(
        input_data, timeout=6000
    )

    print(run_request)
except TimeoutError:
    print("Job timed out.")