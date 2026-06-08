import os
import time
from locust import HttpUser, task, between

class PneumoDetectUser(HttpUser):
    wait_time = between(1.0, 3.0)
    image_bytes = None

    def on_start(self):
        # Locate or fall back to mock image bytes
        img_path = os.path.join(os.path.dirname(__file__), "sample_chest.png")
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                self.image_bytes = f.read()
        else:
            # Simple 1x1 black pixel PNG raw representation as a fallback
            self.image_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x00\x00\x00\x00\x3a\x7e\x9b\x55\x00\x00\x00\nIDATx\x9cc\x60\x00"
                b"\x00\x00\x02\x00\x01\x48\xaf\xa4\x72\x00\x00\x00\x00IEND\xaeB`\x82"
            )

    @task
    def test_predict_and_poll(self):
        """
        Simulates enqueuing an image prediction and polling for results.
        """
        files = {"file": ("sample_chest.png", self.image_bytes, "image/png")}
        
        # 1. Trigger prediction async
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Trigger prediction failed with code {response.status_code}")
                return
            
            try:
                task_id = response.json().get("task_id")
                if not task_id:
                    response.failure("Response JSON missing task_id")
                    return
            except Exception as e:
                response.failure(f"JSON parsing error: {e}")
                return

        # 2. Poll result endpoint
        polling = True
        start_time = time.time()
        while polling:
            # Enforce 30s timeout per poll loop in the client task simulation
            if time.time() - start_time > 30:
                response.failure("Polling timed out after 30 seconds")
                break
                
            with self.client.get(f"/result/{task_id}", catch_response=True) as poll_resp:
                if poll_resp.status_code != 200:
                    poll_resp.failure(f"Polling result failed with code {poll_resp.status_code}")
                    break
                
                try:
                    status = poll_resp.json().get("status")
                    if status in ["SUCCESS", "FAILED"]:
                        polling = False
                        if status == "FAILED":
                            poll_resp.failure(f"Async task failed with details: {poll_resp.json()}")
                    else:
                        time.sleep(0.5)
                except Exception as e:
                    poll_resp.failure(f"Polling JSON parse error: {e}")
                    break
