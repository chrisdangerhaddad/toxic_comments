# Prediction of Toxic Comments, Multinomial Classification


	Team Members
		- Sanjay Roberts 
		- Jeff Coady
		- Chris Haddad
	
	Justification	
		- Our problem involves NLP and predictive modeling.  Jeff Coady has a background in linguistics, 
		- Chris Haddad has experience with predictive modeling, and Sanjay Roberts has some experience with both NLP and modeling.
		- Further, Sanjay Roberts has experience with Kaggle competitions and the rest of the team is anxious to learn about the process of competing.
	
	
	Problems/Motivation
		- Detect toxic comments and minimize unintended model bias
		- Our goal is to build a model that recognizes toxicity in comments and minimizes unintended bias with respect to mentions of identities. 
		- https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview
	
	
	
	
	Libraries and Tools (What you already know and what else you need to evaluate)
		- Known Packages:
			+ matplotlib
			+ pandas
			+ tensorflow
			+ scikit-learn
			+ scipy
			+ imbalanced-learn
			+ keras
			+ ipython-autotime
			+ psutil
			+ nltk
			+ gensim
		- Potential Packages:
			+ elmo
			+ bird
			+ spacy
			+ textblob
		
	Data Collection
		- We will be using a Kaggle dataset.  The dataset is labeled for identity mentions and optimizing a metric designed to measure unintended bias. 
		- The dataset can be found here:
			+ https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
		
	Features:
		+ male
    	+ female
    	+ transgender
    	+ other_gender
    	+ heterosexual
    	+ homosexual_gay_or_lesbian
    	+ bisexual
    	+ other_sexual_orientation
    	+ christian
    	+ jewish
    	+ muslim
    	+ hindu
    	+ buddhist
    	+ atheist
    	+ other_religion
    	+ black
    	+ white
    	+ asian
    	+ latino
    	+ other_race_or_ethnicity
   		+ physical_disability
    	+ intellectual_or_learning_disability
    	+ psychiatric_or_mental_illness
    	+ other_disability
    	
	Any Literature review
		- Our intention is to begin by reviewing the leading kernels here:
			+ https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/kernels
    	- And the following papers:
    		+ https://www.hergertarian.com/detecting-toxic-comments-with-multitask-deep-learning
    			- ompl
    		+ https://becominghuman.ai/my-solution-to-achieve-top-1-in-a-novel-data-science-nlp-competition-db8db2ee356a
    			- kaggle and NLP
			+ https://arxiv.org/pdf/1802.09957.pdf
				- ensemble that outperforms individual deep or shallow models
			+ http://demo.clab.cs.cmu.edu/ethical_nlp/
				- ethics 
			+ https://medium.com/@nehabhangale/toxic-comment-classification-models-comparison-and-selection-6c02add9d39f
				- LSTM v CNN, tokenization, embedding
			
	Required work detail before build model
		- data cleansing
		- Tokenize and Pad
		- FastText Embedding

	What is the predictive task and model detail.
		- multinomial classification of toxicity classes
		- seeking to use ensemble methods
		- grid search 
		- k-folds
		- imbalance-learn
		
	Model evaluation and selection strategy.
		- k-folds for cross validation
		- Matthews Correlation Coefficient for accuracy
		- start with LSTM and CNN, move to ensemble
		- use kaggle submissions to test against other competitors
			+ retrain, resubmit to kaggle, evaluate rank, retrain, ...

	How a user is going to test the final model. is there any webpage/command line interface.
		- TBD
	
	Tentative time line of activities.
		- Week of April 22nd
			+ read literature
		- Week of April 29th
			+ make ML ready dataset
		- Week of May 6th
			+ test LSTM using grid search for hyper parameter tuning, k-folds for bias analysis
		- Week of May 13th 
			+ test CNN, grid search, k-folds
		- Week of May 20th
			+ begin testing ensemble learning method
		- Week of May 27th
			+ continue testing ensemble learning method
		- Week of June 3rd
			+ final submission to kaggle with best results
		- Week of June 10th
			+ summarize results

    
