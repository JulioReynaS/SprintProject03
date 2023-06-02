from locust import HttpUser, between, task


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
<<<<<<< HEAD
    raise NotImplementedError
=======
    @task
    def index(self):
        self.client.get("/")
        print("Landing page loaded...")
    
    @task
    def predict(self):
        data_predicted = {"file": ("dog.jpeg", open("./dog.jpeg", "rb"), "image/jpeg")}
        response = self.client.post("/predict", files=data_predicted)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")

>>>>>>> 2ca48ea (Final commit)
