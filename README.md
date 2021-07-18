# Reading data/ classification of data from MonBox v1.0 and analysing data:
Report for Routers/ Sensors data and its classification done do far

The data we obtained before are:
* Data from routers:
  * First: MonBox v1.0 = w1.mat
  
    w1.mat = 500 samples, each sample consists of 100 000 elements

  * Second: MonBox v1.0 = w2.mat

    w1.mat = 500 samples, each sample consists of 100 000 elements

The example of the data sample is given below (given as - matlab.m):

The data was manipuated using matlab - **matlab.m**:


Then, the csv file "**feature.csv**" was imported and processed using Python:

Using different statistical tools, the data was investigated and erroneous data samples were deleted, relative features was obtained from a data for further investigation. Two sample data were erroneous, and they were removed.

After doing manipulations on data, these features were right candidates to choose: Feature 1 - Mean, Feature 2 - Standard Deviation, Feature 3 - Variance. Then, the data was transformed into new data set with three columns and 988 data points; each column corresponds to a feature and each row corresponds to the value of operation. The generated new data set is visualized in Figure 1 and the code is the same which is given under - "**matlab.m**" and dataset is named as "**feature.csv**".

Then, by using Support Machine Vector Machine, the data was classified with 100 % accuracy which is given in Figure 2 and "**svm.py**" was used:

* Data from sensors:
  * Sensor1 = "sec1_9952_sensor1"
  * Sensor2 = "sec1_992B_sensor2" 
The process of pre-processing data, visualization of the data, taking mean, variance, standard deviation of samples and creating a final dataset is shown in "**section_process_of_creating_dataset_april.ijmp**"

Then, for a section 200Mb of data: mean, standard variation and variation is taken which is shown in **Figure 2**. Classification is done using Support Vector Machine, and the SVM classfication is shown in **Figure 3**, and the script is provided under "**code_svm.py**"

Spectrogram of Sensors were tired to obtain which is shown in "**SensorSection_spectrogram_images**"
This will be used to classify the data using CNN (Convolutional Neural Network)
