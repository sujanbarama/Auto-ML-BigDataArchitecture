# README

This is a Python Flask web application that aims to provide a platform to preprocess and train machine learning models on large datasets. The application uses Google Cloud Storage to store the data and models. It also provides a user interface for data preprocessing and model training.

The app.py file contains the code for the Flask application. It includes the following functions:

1) cloud_read(bucket_name, blob_name): This function reads a CSV file from the specified Google Cloud Storage bucket and returns it as a Pandas dataframe.

2) cloud_write(bucket_name, blob_name, csv_file): This function writes a CSV file to the specified Google Cloud Storage bucket.

3) list_blobs(bucket_name): This function lists all CSV files in the specified Google Cloud Storage bucket.

4) list_blobs_preprocessed(bucket_name): This function lists all preprocessed CSV files in the specified Google Cloud Storage bucket.

5)connection(): This function creates a connection to Firebase, which is used for user authentication.

6)allowed_file(filename): This function checks if the uploaded file is a CSV file.

7)home(): This function renders the home page.

8) uploading(): This function renders the upload page.

9)upload(): This function handles the file upload, saves the uploaded file to a local directory, and uploads it to Google Cloud Storage.

10) preprocess(): This function lists all CSV files in the Google Cloud Storage bucket and renders the preprocess page.

11) preprocessResults(): This function renders the preprocess results page.

12) Trainbutton(): This function renders the training page.

13) preprocessing(): This function handles the data preprocessing and renders the results page.

14)The cleaning.py and modeling.py files contain the code for data preprocessing and model training, respectively. The app.py file imports these files to use the functions in them.

The application also uses the Google Cloud Storage and Firebase Authentication services. The GOOGLE_APPLICATION_CREDENTIALS and Firebase files are included in the code to authenticate the application with these services. These files should be replaced with the appropriate credentials for your project.

To run the application, you need to install the required packages by running pip install -r requirements.txt. You also need to set the FLASK_APP environment variable to app.py and run the application by executing the command flask run.
