#sleepQualityPredictor

This project develops a neural-network regression model that predicts a person’s sleep quality score based on lifestyle, health measurements, and daily habits. Instead of using traditional sleep-tracking hardware, the model relies entirely on easily measurable attributes such as sleep duration, stress level, physical activity, heart rate, BMI category, and occupation. The work uses the publicly available Sleep Health and Lifestyle Dataset, which includes demographic, behavioral, and physiological features for 374 individuals.

Project Motivation:
Good sleep is one of the strongest predictors of overall health, productivity, and mental well-being, yet most people do not have access to medical sleep studies or wearable sleep-tracking devices. The motivation behind this project is to determine whether basic daily metrics — the kind nearly anyone can self-report — can be used to estimate sleep quality reliably. Features such as stress, step count, blood pressure, and activity level strongly correlate with sleep, and the goal is to learn these patterns using a data-driven approach. A neural network built with PyTorch allows the model to capture non-linear relationships across lifestyle factors.

Dataset:
The project uses the “Sleep Health and Lifestyle Dataset,” which contains 374 samples and 13 original input features. These include demographic variables (gender, age, occupation), lifestyle factors (daily steps, physical activity, stress level), and health indicators (BMI category, heart rate, blood pressure). The target variable is “Quality of Sleep,” an integer from 1 to 10. Missing values in the Sleep Disorder column are handled, and categorical variables are one-hot encoded so they can be processed numerically.

Methodology:
The dataset is loaded, cleaned, and preprocessed using pandas and scikit-learn. Blood pressure is split into systolic and diastolic components, categorical variables are one-hot encoded, and all numerical features are standardized using StandardScaler. The data is split into training and testing sets. A fully-connected neural network is implemented in PyTorch with two hidden layers (64 and 32 neurons) and ReLU activations. The model is trained using MSE loss and the Adam optimizer. A custom Dataset and DataLoader pipeline is implemented to handle batching. Model performance is tracked across training epochs, and final performance is measured using mean squared error on the held-out test set.

Model Performance:
The neural network learned the dataset’s patterns extremely effectively. Training loss decreased smoothly from over 50 down to approximately 0.12 by the final epoch. The final model achieved a test MSE of about 0.27, indicating that predictions are typically within roughly half a point of the true sleep quality score. Given that the target variable ranges only from 1 to 10, this level of accuracy shows strong generalization and demonstrates that lifestyle and health metrics can reliably estimate sleep quality through a data-driven approach.

Final Model:
The final selected model is a PyTorch feed-forward neural network consisting of two hidden layers with ReLU activations. Its strong performance suggests that small, well-designed neural networks can effectively learn non-linear lifestyle–sleep relationships. The trained weights can be saved and later used to make predictions on new lifestyle data. Because the project avoids device-specific or medical inputs, it serves as a lightweight and accessible method for estimating sleep quality.

Running the Project:
To run the project, install the required dependencies using pip. After installing the packages, run the preprocessing and training scripts. The code handles loading the dataset, preprocessing features, creating PyTorch dataloaders, training the neural network, and evaluating model performance. Users may modify the model architecture, optimization parameters, or preprocessing steps to further experiment with prediction quality.

Future Work:
Future improvements may include experimenting with deeper models, adding dropout or batch normalization for regularization, performing hyperparameter tuning, or testing additional regression architectures such as XGBoost, LightGBM, or multilayer perceptrons with different activation functions. Another direction involves extending the dataset with wearable-sensor features or integrating the model into a lightweight web or mobile application for real-time estimation.

Mohammad Jeneidi
Florida State University
Computer Science
