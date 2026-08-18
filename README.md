
# Spotify-audio-Prediction-repository

!(https://pin.it/7K1UgKIuQ)


This project aims at build a model that with recommend music of certain similar characteristics to the users using Spotify. 
## Problem Statement.
* Music streaming platform contain a large number of songs, making it difficult users to discover songs that match their personal preferences.
* This project aims to analyze users' songs preferences using audio features such as danceability, energy, acousticness, valance, tempo nd loudness and develop a system that can predict whether a user is a likely to enjoy a song and recommend songs with similar characteristics.
## Aim of Analysis.
* To investigate the relationship between song audio features and user preferences in order to identify patterns that can be used to predict whether a user will like a song and recommend suitable songs.
## Business Understanding.
(a)What are the characteristics of the songs in the dataset?
(b)Which audio features are associated with the songs being liked?
(c)Which model predicts whether the song is liked?
(d)Can i recommend suggested songs with user preferences?

## Data understanding
* The dataset source is from [kaggle](https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation)
   * * Key features;*
* Danceability- how suitable a song is for dancing,from low to high
* Energy- how intense the song is
* speechiness - amount of spoken words in the song
* Acousticness - how acoustic the song sounds
* Instrumentalness - likelihood that the song contains little/no vocals
* Liveness - likelihood that the recoding sounds like a live performance.
* Valance- musical positivity/happiness of a song
* Tempo - speed of the song, measured in BPM.
* liked - whether the user liked the song; 1= liked, 0= not liked
  ## Key Visualizations

  
  <img width="618" height="470" alt="image" src="https://github.com/user-attachments/assets/820cb1dd-7b80-4cab-8960-45fbbef69e01" />


  *Findings*
* 100 songs liked
* 95 songs not liked
* The target class is relatively balanced hence good for classification.


<img width="686" height="470" alt="image" src="https://github.com/user-attachments/assets/54192a55-ff65-4900-bb9d-387f30562d05" />



*Findings*
* The distribution is negatively (left) skewed because most observations are concentrated at higher danceability vales while a smaller number tracks have low danceability.



  <img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/30222651-05e7-46d3-b1c7-79dcc59a7a21" />


  *Findings*
  * The scatter plot indicates a positive relationship between danceability and valence. Liked songs tend to be concentrated at higher danceability and moderate-to-high valence levels, although there is considerable overlap between liked and disliked songs.
* This suggests that danceability and valence may contribute to predicting song preference but are not sufficient on their own.


  ## Conclusions
* Song recommendation is after by many factors danceability, energy, speechiness, etc.
* The recommendation is based on content-based recommendation due to columns such as user_id missing.
  
## Model and evaluation
*The following models were used:
   * Logistic Regression - baseline model
   * Decision tree - interpretable model
   * Random Forest - Strong general purpose model
   * K-Nearest Neighbors - particularly relevant, the project involves songs similarity.
      * CONFUSION MATRIX FOR RANDOM FOREST MODEL:
    
        
   <img width="530" height="455" alt="image" src="https://github.com/user-attachments/assets/333c4330-572a-4f2c-889e-6a2213306b21" />
   


   *Finding*
    18 -True Negative
  * 18 songs were not liked and the model corectly not liked(0)
* 1 - False positive
  * 1 song was not liked and the model predicted liked.
* 1 - False negative 
  * 1 song was actually liked but the model predicted not liked
* 19 - True positive
  * 19 songs were actually liked (1) and the model correctly predicted liked.

*Random Forest* achieved 94.87%  accurate on test data.
     

## Core Findings
 Random Forest performed best overall.
   * It achieved the highest accuracy (94.87%), precision (95%), and F1 score (95%), making it the strongest model among the five tested.
* Logistic Regression had the highest recall (100%).
    * This means it successfully identified all of the songs that were actually liked in the test set. However, its precision was lower at 86.96%, meaning it produced more false-positive predictions than Random Forest.
* KNN performed reasonably well.
  * Its accuracy was 92.31% and precision was 94.74%, but its recall of 90% was lower than Random Forest.
* Decision Tree and SVM had the weakest overall performance.
   * Both achieved 89.74% accuracy and an F1 score of 90.48%, so they were less effective than Random Forest, KNN, and Logistic Regression on this dataset.
* Random Forest provides the best balance.
   * Its precision and recall are both 95%, indicating that it is relatively balanced in identifying both liked and non-liked songs.

  ### Future work plans
  * Random Forest as primary prediction model
* Logistic regression as a supporting model.
* Improve the dataset by collecting more user interactions

  



