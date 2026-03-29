Main Assignment Goal – Maximise classification accuracy, precision and recall as much as possible
on the supplied dataset (evaluate on the test set).
The first thing you should do is create a python virtual environment (venv) for your deep learning work.
A Python virtual environment is an isolated space where you can work on your Python projects,
separately from your system-installed Python. You can set up your own libraries and dependencies
without affecting the system Python. See the list of steps below to set this up.

1. Pick a directory on your hard drive that you want your python virtual environments to be
   located in.
2. Open a command prompt and change into that directory.
3. Now enter the command (on Windows): ‘py -m venv <venv_name> ‘ where <venv_name> is
   the name you choose for the virtual environment
4. Now you need to update the venv with the latest version of python (tensorflow needs version
   3.8+): ‘python -m venv --upgrade <venv_name>’
5. Now you need to activate the virtual environment: ‘<venv_name>\Scripts\activate’. To
   deactivate the venv type: ‘deactivate’ at the cmd prompt.
6. Now you can install tensorflow and keras with the following command: ‘pip install tensorflow’
7. You may need to install matplotlib too: ‘pip install matplotlib’
8. Now you should be able to navigate to the folder where your python script is and run it:
   ‘python myscript.py’
   Once you are setup and ready to go download the two .py files that are on Moodle/Brightspace
   (mnist_classification.py and pneumonia_classification.py) and store them in a local directory. Verify
   your tensorflow installation is working correctly by running the mnist classification example. This is a
   very simple keras example that performs classification on the mnist datatset (handwritten digitis).
   Once verified download the chest xray dataset from the following link:
   https://drive.google.com/file/d/1lP9l3FBNNmWJD7q69v9Bnf4TMDxay_tx/view?usp=sharing
   Extract this dataset somewhere appropriate on your hard drive (you will need 1.1 GB for it). Now
   modify the pneumonia_classification script to include the path of the directory to the extracted image
   data. Browse through the directory structure and look at some of the images.
   Now you should be in a position to execute the pneumonia_classification.py script. It should produce
   an output as follows:
   You will see the progress of the neural network training phase. Depending on your PC this can take a
   few minutes but it should not take more than this.
   For this assignment you are required to improve the validation and test scores of the network
   performing pneumonia classification. This will involve some research on your behalf. In particular
   answer the following questions?
9. How long did the network take to train?
10. Is the dataset balanced? What is the distribution? If not can I do anything to address this?
11. Is the network overfitting? Why or why not? If so what can I do to address this, see
    GlobalAveragePooling2D ?
12. Can you perform any data augmentation?
13. What layers are in the network? Can I alter anything here to improve matters? Perhaps take a
    look at the keras tuner (https://www.tensorflow.org/tutorials/keras/keras_tuner)
14. Is transfer learning an option here using a pre-trained model?
15. What are the per class precision, recall and F1 scores? What do they mean and which is best
    here.
16. How might you make the model better at finding the sick patients?
17. Is it possible to see what the CNN is seeing? Can I use tf-explain (e.g.GradCam) to see this?
18. Anything else that you come across in your research to improve the model.
    For this assessment you should submit the following:
19. Your updated python script or a Jupyter notebook (preferred) performing the chest xray
    classification and a link to your Github repo showing all your commits.
20. A report detailing your trials, tribulations and findings.
