import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer


def preprocess_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


def extract_features(data):
    features = []
    labels = []
    ecoscore_values = []

    for product_id, product_data in data.items():
        product_features = {
            'est_co2_agriculture': product_data['est_co2_agriculture'],
            'est_co2_processing': product_data['est_co2_processing'],
            'est_co2_distribution': product_data['est_co2_distribution'],
            'est_co2_transportation': product_data['est_co2_transportation'],
            'est_co2_packaging': product_data['est_co2_packaging'],
            'est_co2_consumption': product_data['est_co2_consumption'],
            'non_recyclable_and_non_biodegradable_materials_count': product_data[
                'non_recyclable_and_non_biodegradable_materials_count']
        }

        features.append(product_features)

        if 'ecoscore_grade' in product_data:
            ecoscore_values.append(product_data['ecoscore_grade'])
            labels.append(product_data['ecoscore_grade'])
        else:
            labels.append(None)  # Handle the missing value in an appropriate way

    return features, labels


def create_model(hidden_layer_sizes=(100, 100), alpha=0.01):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, random_state=42, early_stopping=True)
    return model


def train_model(features, labels):
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the desired hyperparameters
    hidden_layer_sizes = (100, 100)
    alpha = 0.01

    model = create_model(hidden_layer_sizes, alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1_macro_val = f1_score(y_val, y_pred, average='macro')

    print(f1_macro_val)

    return model, vectorizer, f1_macro_val


# Preprocess the training data
train_data = preprocess_data('train_products.json')
train_features, train_labels = extract_features(train_data)

# Train the MLP model with hyperparameter search
model, vectorizer, f1_macro_val = train_model(train_features, train_labels)

# Preprocess the test data
test_data = preprocess_data('test_products.json')
test_features, _ = extract_features(test_data)

# Generate predictions using the MLP model
predictions_dict = {}

X_test = vectorizer.transform(test_features)
predictions = model.predict(X_test)

# Create a dictionary in the desired format
model_predictions = {"target": {}}
for i, prediction in enumerate(predictions):
    model_predictions["target"][str(i)] = int(prediction)

predictions_dict["MLPClassifier"] = model_predictions

# Save predictions to a file
with open('predictions.json', 'w') as file:
    json.dump(model_predictions, file)
