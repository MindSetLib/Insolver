from locust import HttpUser, task
from loading_test.request_testing import req_json_ml_ok


class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.post("", json=req_json_ml_ok)
