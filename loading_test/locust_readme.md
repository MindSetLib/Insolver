Load testing with Locust
====

[example from the library itself](https://docs.locust.io/en/stable/quickstart.html#locust-s-web-interface)


1. Open a separate terminal
2. Go to the folder with the project
3. Launch the virtual environment
4. Launch Locust
```
locust -f loading_test/locustfile.py
```
5. Launch the Locust application in the browser
locust address: http://localhost:8089

6. Fill in the fields of the window\
- the first is the number of users
- second - what steps to increase users until the size in the first window is reached
- the third one is the address of the launched project.\
Example for flask: http://localhost:8000/predict \
Example for django and fastapi: http://localhost:8000/predict/

7. In the upper right corner there is a button **Stop**, it is also **Start**
