




Vorgefertige Docker Apps
	z.B. PostgreSQL
	
	
Source to Image
	1 Python App mit Code, App.py, Requirements.txt
	Das wird von OpenShift erkannt und die App gebaut
	

Via Dockerfile
	Nimm ein Python Basis Image
	Kopiere den Python Code rein
	Starte app.py
	
	
Kann meine Anwendung zig mal skalieren,
indem ich die Anzahl der Pods erhöhe

Der Load Balancer verteilt dann die http Anfragen auf die Pods


Dann noch zur App eine Route setzen (=> URL)
	oc expose service servicename --hostname hostname