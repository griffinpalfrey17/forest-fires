
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset
data = pd.read_csv('forestfires.csv')

# Prepare your feature matrix X and target vector y
X = data[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]
y = data['area']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'forestfires_model.joblib')


'''
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv('forestfires.csv')

# Preprocess the data
data['month'] = data['month'].astype('category').cat.codes
data['day'] = data['day'].astype('category').cat.codes

X = data.drop('area', axis=1)
y = data['area']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'forestfires_model.joblib')
'''