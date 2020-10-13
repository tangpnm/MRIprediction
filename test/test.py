import requests 

img = {'image': open('mildDem0.jpg', 'rb')}
resp = requests.post("http://localhost:5000/predict",
                     files=img)

print(resp.text)

