# Capstone Project
### Machine Learning Engineer with Microsoft Azure Program
###### Scholarship recipient: Herbert Fernández Tamayo

### Table of Content
1. Project's overview
2. Project set up and installation
3. Dataset
- Overview
- Task
- Access
4. Automated ML
- Results
5. Hyperparameter tuning
- Results
6. Model deployment
7. Screen recording
8. Standout suggestions

## 1. Project's overview
As a part of the Machine Learning Engineer with Microsoft Azure Nanodegree, the third project is related to apply the knowledge acquired to solve or analyze a problem of the real life, it is important to choose a dataset related to a real scenario rather than a sample dataset, in this case I've chosen the Dataset of year 2018 related to Homicides in El Salvador, I work for the Statistics Department of the National Institute of Forensic Sciences of El Salvador, this institution is one of the most respected sources in the country related to the analysis about the behavior of different crimes in El Salvador. 

The main goal of this experiment is to predict the incidence of homicides in males compare to females; once we obtain and analysis the results, it should be a point of start for planning programs and policies of prevention in the cohort with a higher risk obtained.

The project has two models: the first one is based on Automated Machine Learning method which consists in the evaluation of the dataset using multiples algorithms, then we may choose the best one based on its level of accuracy; the second one is based on HyperDrive method, the user sets different hyperparameters to obtain results, these ones may vary depending of the algorithms and hyperparameters sets chosen. 

Comparing the results of both models, it is encourage to deploy the best performing model through a web service and test it sending a data request.

The below diagram was provided by Capstone project's instructor with the idea to understand better how both models should be run:
![pdiagram.png](./img/pdiagram.png?raw=true "Project diagram")


## 2. Project Set Up and Installation
To run this project on your own Azure Machine Learning Studio environment you should follow the next steps:
1. Download from this repo the next files: automl.ipynb, hyperparameter_tuning.ipynb, train.py, score.py and cad2018.csv, you don't need to clone the entire repo
2. Sign in your Azure ML Studio, upload the ipynb and py files 
3. Create a type DS2_V2 Compute Instance to run the above jupyter notebooks files, you may call it "notebooks"
4. From the new Compute Instance load Jupyter Notebook framework.
5. Next, open automl.ipynb and run each cell at a time, pay attention to the results
6. When step 7 has been finished you may open and run each cell on hyperparameter_tuning.ipynb file; again, pay attention to the results
7. It is not mandatory to run one file at a time but it is a good idea in order to identify the differences in the processes executed.
8. If you want to try your own dataset, you may change the objects called: rawdata_homic2018 as well as the target column name in train.py and automl.ipynb
9. By the time I run my experiments I need to update some dependencies, in case it is your case, open a terminal and run the next command: pip install --upgrade azureml-sdk[notebooks,automl]


## 3. Dataset

### 3.1 Overview
The dataset, "cad2018.csv", contains the records of deaths by homicides in El Salvador during 2018, the data is gathered in 7 different offices around the country and monthly is validated by different data analyst of the Statistics Department of the National Institute of Forensic Science of El Salvador, since 2005 this country has been affected by the increase of deaths related to different way of violence, just in 2018 the official number of deaths related to homicides was 3,346. 

The original dataset has more than 70 columns; for this experiment, I've chosen 7 columns of them because the other ones didn't have a close relation with the hypothesis I need to test; the information of each column ,and how each was recoded, is the next:

- id: internal ID field for each record, this is the primary key field. [no recoded]

- regfalle: national ID number assigned for each case of homicide. [no recoded]

- edad: age by the time the person has been murdered. [no recoded]

- sexo: sex of the murdered person. Recoded values: [male -> 0, female -> 1]

- deptoocuhe: name of the state where the homicide was commited: Recoded values: [Ahuachapan->1, Santa Ana->2, Sonsonate->3, Chalatenango->4, La Libertad->5, San Salvador->6, Cuscatlan->7, La Paz->8, Cabanas->9, San Vicente->10, Usulutan->11, San Miguel->12, Morazan->13, La Union->14]

- tipoarma: name of the object used to commit the crime: Recoded values: [arma de fuego->1, asf x estrangulacion->2, asf x ahorcadura->3, asf x sofocacion->4, asf x sumersion->5, blanca sin espec->6, caida provocada->7, cortante->8, cortocontundente->9, cortopunzante->10, manos y pies->11, no datos->12, objeto contundente->13, lapidado->14, punzante->15, quemadura x fuego->16

- pracaut: Authopsy practiced to the corpse: Recoded values: [si->1, no->0]

Any user may request a copy of the dataset, it is necessary to specify what variables are needed, also it is mandatory to fulfill the next form:
[Request information](https://transparencia.oj.gob.sv/es/solicitud-informacion)

In the particular case of El Salvador related to homicides, the international community has been supported a lot of prevention programs with a very few positive results, why was that? I think we still have a poor understanding of the situation and where to focus stategies of prevention; for this project, I chose the variable "sex" as the target column with the idea to gain a deeper understanding in the incidences and "risk" of homicides in males and females. 


### 3.2 Task
The main objective of this project is to run and find the best of two models, one using HyperDrive experiment and the other one using Automated Machine Learning experiment, that can help us to determine if the salvadoran male population will have a higher risk than salvadoran women of been murdered in homicides circumstances.

From 2010 in El Salvador there are different campaings to report and prevent murders just in women population -which is a great innitiative- but the purpose running this project is to put in the map that salvadoran male have almost thrice possilibites to died in homicides circumstances.

The column key -sexo- has this categories: 0->male, 1-> female, which resembles it is a classification problem.


### 3.3 Access
For this experiment, the recoded data has been uploaded to this repository, from the jupyter notebook files the source code to access it is the next one:

from azureml.data.dataset_factory import TabularDatasetFactory

rawdata_homic2018 = "https://raw.githubusercontent.com/hftamayo/azuremlproj03/main/cad2018.csv"
dshomic2018 = TabularDatasetFactory.from_delimited_files(path=rawdata_homic2018, separator=',')


## 4. Automated ML
The configuration as well as other technical details related to the Automated ML experiment are the next ones:
- experiment_timeout_minutes : 20 (defines the exit criteria of each iteration)

- max_concurrent_iterations: 5 (number of thread -iteration- that can be executed simultaneously)

- primary_metric : 'accuracy' (this is the metric the experiment will try to optimize)

- task: 'classification' (key task that the experiment will focus to solve)

- label_column_name='sexo' --> key column the experiment will try to predict)

I decided to choose "accuracy" as the primary metric value because it is an intuitive measure, according to its definition "is the ratio of predictions that exactly match the true class labels" (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml) . However, some authors refers accuracy as an improper metric for unbalanced classes, other ones afirm accuracy is improper even for balanced classes.  It is encourage to test different metrics such as F1-score, confusion matrix, precision, recall between others, also it is important to test your experiments using "real data", after all, the idea is to load our models in a production environment and this is impossible using no other than real datasets.


### 4.1 Results
The algorithm with the best performance during the tests was "VotingEnsemble" with a score of 0.8822, for a detailed list of the results you may check the file "automl.ipynb" section "Run Details"; in the next picture we can see the last algorithms executed and its results:

- Figure 1: Run Details

![automl_01.png](./img/automl_01.png?raw=true "AutoML best result")


One of the most useful tool running this experiment is the RunDetails widget where we can get different graphical elements related to the results, in the next picture we can observe a 2D graphical of the result during their execution:

- Figure 2: 2D Graphical based on the results

![automl_02.png](./img/automl_02.png?raw=true "2D graphical")

One fact to be taken in count is we can get a slight different results during the execution of the same experiment - no changes in the jupyter notebook- at different times in Machine Learning Studio, it might be related to technical reasons such as  CPU performance, bandwith, between others, it would be great in the future to know the exact reasons. In the next picture you can see the result of VotingEnsemble was 0.8825 which is a little bit higher that 0.8822:

- Figure 3: RunDetails Widget

![automl_03.png](./img/automl_03.png?raw=true "results")

Details of the best model are shown in the next picture:

- Figure 4: Best model's metrics

![automl_04.png](./img/automl_04.png?raw=true "best model")


Also, in the next picture, the parameters for the best model are shown, more details about the source code will be find in the notebook file:

- Figure 5: Best model's parameters

![automl_05.png](./img/automl_05.png?raw=true "best model parameters")


In a near future I would like to expand some options in order to evaluate if the results may be improved: 
- Try running the experiment in compute cluster with more resources (GPU instead of CPU for example)

- Increase the running time in order to evaluate if the results obtained are more accurated and finally 

- Expand the number of columns evaluated

- Increase the number of cross validations trying to reduce the bias.


## 5. Hyperparameter Tuning
I decided to use the Logistic Regression algorithm which is part of the SciKitLearn library, to run the HyperDrive Experiment ins necessary to set the next parameters:

- C: which determines the strength of the regularization, higher values of C correspond to less regularization

- max_iter: It's the number of iteration over the full dataset.

Also, the experiment uses random parameter sampling, this one, on one hand, is very useful for discover more hyperamater combinations; on the other hand it demands more time during the execution of the experiment.

For this experiment, C was set with these values: (1, 2, 3, 4), and max_iter with (40, 80, 120, 130, 200). 

Another sets of parameters used in this experiment are:
- evaluation_interval: 1

- slack_factor: 0.2

- delay_evaluation: 5

The above parameters have relation with the accuracy of the experiment, they are useful to stop the experiment in case some conditions may be reached (that is an early termination policy), this is useful to the efficient use of compute resources.


### 5.1 Results
After the experiment was completed, the highest accuracy obtained was 0.891434, in the next picture is presented details of the results:

- Figure 6: Hyperdrive's best metrics

![hyper_05.png](./img/hyper_05.png?raw=true "best model")


In the next pictures and using the Run Details Widget, it is possible to have more details about the experiment:

- Figure 7: RunDetails Widget: Experiment completed

![hyper_01.png](./img/hyper_01.png?raw=true "Experiment completed")

- Figure 8: RunDetails Widget: Details of the results for each run:

![hyper_02.png](./img/hyper_02.png?raw=true "Experiment completed")

- Figure 9: Graphics of the result

![hyper_03.png](./img/hyper_03.png?raw=true "Experiment completed")

- Figure 10: Graphics of the result

![hyper_04.png](./img/hyper_04.png?raw=true "Experiment completed")


Of course, the experiment can be improved, this is a list of suggestions for future changes:
- Use Bayesian Parameter Sampling

- Use a different set of primary metrics

## 6. Model Deployment
Once we have a trained model, the next step is to deploy it using Azure Machine Learning Endpoints, to achieve this task we need:
- A trained model
- Configuration files: such as scoring and environment files
- Deploy technical details: choosing what type of container we will use (Azure Container Interfaces, Azure Kubernetes), CPU/GPU and Memory allocation

For general purpose in this project I decided to choose Azure Container Interface (ACI) with 1 CPU and 1 GB of Memory.

Even though the rubric of the project clearly specify I should deploy the best model obtained, I decided to implement both of them for the next reasons:
- I want to compare if the deployment process is the same for both models.
- Learn about pitfalls or special configuration for each model.
- The ooportunity to learn something new.

In the next paragraphs I will describe how I deployed each model:

### 6.1 Deploying AutoML Model:

In the case of the AutoML experiment, the score file was generated by the model, the purpose of this file is to give information about what type of input data the model expects to process it and return results; It is useful to have the information from the environment method of the best model obtained to deploy the webservice. In the next image shows the detail of the code of how the webservice is created and deployed:

- Figure 11: AutoMl Best Model Deployment part 1

![automlexp_wservice_scode.png](./img/automlexp_wservice_scode.png?raw=true "AutoML webservice sourcecode")

How to query the endpoint? first we need to choose and amount of records from the dataset, in this case I chose 10 records, then we may create a dictionary object with the above records, next we need to convert it to a JSON Format. The interaction with the WebService is possible to its REST API which is capable to receive data, in our case in JSON format, and retrieve predictions to the user, in the jupyter notebook file the variable "response" receive the data from the API.


Also, details in the scoring.py file are important to obtain output, in the case of this experiment, the output was like this (remember 0 in this experiment resembles "male"):



- Figure 12: AutoMl Best Model Deployment part 2

![automlexp_wservice_requests_results.png](./img/automlexp_wservice_requests_results.png?raw=true "AutoML webservice results")


Details about the Endpoint are given in the next images:

- Figure 13: AutoMl Best Model Deployment part 3

![automlexp_endpoint_widget.png](./img/automlexp_endpoint_widget.png?raw=true "AutoML Endpoint widget")

- Figure 14: AutoMl Best Model Deployment part 4

![automlexp_endpoint_status.png](./img/automlexp_endpoint_status.png?raw=true "AutoML Endpoint widget")

- Figure 15: AutoMl Best Model Deployment part 5

![automlexp_endpoint_status02.png](./img/automlexp_endpoint_status02.png?raw=true "AutoML Endpoint widget")

- Figure 16: AutoMl Best Model Deployment part 6

![automlexp_endpoint_status03.png](./img/automlexp_endpoint_status03.png?raw=true "AutoML Endpoint widget")


As a good practice, in the jupyter file, at the end of it, the webservice and the compute cluster are deleted because I won't be accesible anymore, this process reminds me the garbage class collection of JAVA:

- Figure 17: Cleaning up work environment

![automlexp_deleting_ccluster.png](./img/automlexp_deleting_ccluster.png?raw=true "AutoML Endpoint widget")


Some pitfalls during the deployment process:
- Be careful how your model is named and passed to the InferenceConfig.

- Most common exception: FileNotFoundException, PathNotFoundException.

- Double check your dataset, any inconsistency may affect the deployment process and the interaction with the webservice

- Understand first how you will interact with the model and how the test data has to be transformed to JSON.


### 6.2 Deploying HyperParameter Model:

The most important conclusion is deploying the hyperparameter model is not the same as the AutoML -no, copy and paste won't work- one of the aspect to be taken in count is how the model is registered, the best way I found is to obtained during the training of the experiment, that's mean, train.py is the key for this. Then, please check if the model's file is uploaded to the environment, be sure of the name of it. From here, you need to register the model using the "register_model()" method.


Another diference is I have to write the scoring file (score.py) following the guidelines from the official documentation, please check score.py from this repo for further details.

The sourcode from my hyperdrive jupyter notebook file to implement the model is this:

- Figure 18: Hyperparameter Best Model Deployment part 1

![hyper_wservice_scode.png](./img/hyper_wservice_scode.png?raw=true "Hyper webservice sourcecode")

How to query the endpoint? In this case I decided to create a list data structure with the values of a record in particular, so for each field in the dataset we need to extract its value and add them into the list, I designed this method of testing because in a future releases of the experiment I would like to test data from regions of the El Salvador with different rates of homicides. After the list is ready, we have to convert it to a JSON format. The interaction with the WebService is possible to its REST API which is capable to receive data, in our case in JSON format, and retrieve predictions to the user, in the jupyter notebook file the variable "response" receive the data from the API, also I printed the status code of the Edpoint (200 indicates that the request has succeeded). More details in the next picture:

- Figure 19: Hyperparameter Best Model Deployment part 2

![hyperws_09.png](./img/hyperws_09.png?raw=true "AutoML webservice testing")

- Figure 20: Hyperparameter Best Model Deployment part 3

![hyperws_10.png](./img/hyperws_10.png?raw=true "AutoML webservice testing")

About the endpoint, in the next images there are technical information that might be useful to check its healthy:

- Figure 21: Hyperparameter Best Model Deployment part 4

![hyperws_06.png](./img/hyperws_06.png?raw=true "Hyper webservice deployed")

- Figure 22: Hyperparameter Best Model Deployment part 5

![hyperws_07.png](./img/hyperws_07.png?raw=true "Hyper webservice deployed")

- Figure 23: Hyperparameter Best Model Deployment part 6

![hyperws_08.png](./img/hyperws_08.png?raw=true "Hyper webservice deployed")


Some pitfalls during the deployment process:
- Be sure to understand the process how your model is obtained and registered.

- Don't try to obtain the scoring and the environment files automatically, it was a waste of time at least for me because I couldn't do that.

- The output of your webservice is heavily influenced by how you coded your score.py

- Be sure how to convert your test data into JSON format.

## 7. Screen Recording

For an overview of the project, please refer to the next video:

[Project's Overview](https://youtu.be/foNxcrx8xMM)

A detailed video about this project can be found here:
[Project's Details](https://youtu.be/9jupGgBN27k)

Disclaimer: in the rubric there is a policy about the limit time of the video, however, I extended its duration thinking in give more details to any person interested in the project and in general interested in the learning process of Machine Learning with Microsoft Azure Studio, for entry levels users it can be a challenge to design, run and debug this kind of project so I wanted to give more elements to that kind of public. My intention was not to break rules from the rubric, just to be as explanatory as it may be needed.

## 8. Standout Suggestions
1. Use a dataset from real scenario is totally different than uses a sample dataset, it needs to run a validation a reclassification of the categories, double checked the values, output format, between other facts, despite all that work it has a good feeling to face with real results from real data, from my point of view, it's an extra skill from the ML Engineer this kind of experience. Most of the sample databases are so accurate than the results are almost automatically expected
2. Develop a Front End interface to interact with the deployed model, the FrontEnd will be more comprehensible for end-users
3. Suggest to the National Institute of Forensic Sciences of El Salvador to redesign the database of homicides, simplifying some redundant variables and recoded those ones with character only options
4. Deploy the best model found in a container, such as Kubernetes
5. Explore different algorithms to analyze the database of homicides and predict future behaviors of the phenomenon


## 9. Best practice:
In the last part of each experiments, the compute cluster as well as the webservices are deleted:

- Figure 24: cleaning up work environment

![bp01.png](./img/bp01.png?raw=true "Best practices")
