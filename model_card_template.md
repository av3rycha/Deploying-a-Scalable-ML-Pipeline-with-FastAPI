# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a logistic regression model used to predict if an individual has an income of over $50,000 based on demographic data. This model is trained on US Census Bureau Census Income data and produces a binary classification of either greater than $50,000 or less than $50,000.

## Intended Use

The model was developed as part of a machine learning project to demonstrate the development, evaluation, and deployment of a classification model. The model is intended for educational purposes.

## Training Data

This model was training using the Census Income dataset from the US Census Bureau. The dataset contains demographic and employment-related information including sex, race, age, work class, education, occupation, and salary.

Categorical features were encoded before being used by the model. The salary column was used as the target variable. The data was divided into training and tests sets with an 80/20 split. The training test was used to fit the logistic regresssion model while the test set was used to evaluate its performance.

## Metrics

The metrics used to evaluate performance are precision, recall, and F1 score. On the test dataset, the model achieved a precision of 0.7280, a recall of 0.2794, and an F1 score of 0.4039.

## Ethical Considerations

There is a risk of oversimplifying the factors that influence a person's income. The demographic and employment-related features in the dataset do not fully explain why an individual earns a particular salary.

This model may also reflect biases in the underlying Census data. Particularly when using demographic characteristics such as race and sex.

## Caveats and Recommendations

The dataset contains an uneven representation of demographic groups with males and white individuals comprising a disproportionately large portion of the observations. Additionally, model performance varies across gender groups. The model has substantially higher precision and F1 scores for males than females, which may indicate that the model does not perform equally across these groups.

Further versions of this model should evaluate and monitor performance across demographic groups. Additional data from underrepresented groups could improve the reliability of the model. Futher investigation should be conducted to determine why model performance differs between males and females.