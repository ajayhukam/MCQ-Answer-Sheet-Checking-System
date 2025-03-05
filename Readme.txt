                                        **MCQ Answer Sheet Checking System**


**INSTRUCTIONS**
To run the project:

1. open the folder in any code editor(eg.VSCode) #python version should be upto 3.11 only as tensorflow not working in python 3.11 above
   
2. install the following libraries: 
   numpy, pandas, tensorflow, scikit-learn, matplotlib, OpenCV (if prompted, install the other required libraries)
3. Run main.py
4. Login page will pop-up,
	USERNAME: admin
	PASS: 1234

    if having trouble in Login, RUN without_login.py

5. GUI will pop up, load ModelAnswer.png from AnswerKey folder
6. Click Generate metadata and verify if you want to. (metadata will be saved in Metadata sub-folder)
7. load metadata (json file from metadata folder) 
8. student answersheet folder
9. select output report format and location at which you want to save output file.
10. Click Run Grading.




The folder contains the following files:
AnswerKey folder : contains ModelAnswerSheet (Marking Scheme)
Metadata : generated metadata json file
prepared_dataset : the training dataset split into Train and Test folders
StudentAnswerSheets :  folder containing scanned answer sheets of all students
train_cnn.py : python code to train model using given dataset
cnn_model.h5 : the trained model
main.py: the python code containg functions like generating metadata, grading and gui






