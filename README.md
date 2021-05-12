"# CTImageAnalysis" 
"# CTImageAnalysis" 

for Tensorboard:

	- on Server: 
	sudo tensorboard --logdir=/home/paperspace/PyCharmProjects/CTANALYSIS/LOGS --port=6000
	
	- local: 
	ssh -NfL localhost:8898:localhost:6000 ps
