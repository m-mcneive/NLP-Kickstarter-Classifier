# Natural Language Processing Course
## Final Project

During my time at college, I took a Natural Language Processing course where we used various ML models to analyze and classify text data. We spent 
the majority of the course wotking with IMDB movie reviews. I was able to build some models to determine whether a review was positive or negative 
based on keywords that were used. However, this is not what we used for the final project, but rather found some new data. For this project, we
found a database of the descriptions of thousands of Kickstarted projects and whether or not they met their contribution goal. Our goal was to see
whether or not we could build a model to predict if a project succeded or failed in reaching it's goal.

This project was built on the Keras model using 3 dense layers and measured loss using a binary-crossentroty loss function. In the end, we were able 
to acheive an accuracy of 67%, which is good given the nature of the data. 

The code for the Keras model can be found in ```script.py```
