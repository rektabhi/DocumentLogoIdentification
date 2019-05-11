Document Logo Identification

1. Install Anaconda or a similar distribution of python.

2. Open conda prompt

3. Navigate to the project folder and install the dependencies using the command 
	
	pip install -r requirements.txt

4. Connect mobile and PC to the same local network.
	
5. Navigate to src/ folder and run the commands
	
	set FLASK_APP=driver.py
	python -m flask run --host=0.0.0.0

6. Check the IP address of the PC using ipconfig.

7. Change the IP address in the Camera2BasicFragment.java
	Goto Line 978 in the above file and change http://192.168.0.7:5000/predict_image to http://your_ip/predict_image