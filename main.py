import json
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


def extract_features(data):
    features = []
    labels = []

    for product_id, product_data in data.items():
        product_features = {
            'is_beverage': product_data['is_beverage'],
            'packaging_materials': product_data['packaging_materials'],
            'est_co2_agriculture': product_data['est_co2_agriculture'],
            'est_co2_distribution': product_data['est_co2_distribution'],
            'est_co2_packaging': product_data['est_co2_packaging'],
            'est_co2_processing': product_data['est_co2_processing'],
            'est_co2_transportation': product_data['est_co2_transportation'],
            'nutrition_grade': product_data['nutrition_grade'],
            'additives_count': product_data['additives_count'],
            'calcium_100g': product_data['calcium_100g'],
            'carbohydrates_100g': product_data['carbohydrates_100g'],
            'fat_100g': product_data['fat_100g'],
            'fiber_100g': product_data['fiber_100g'],
            'proteins_100g': product_data['proteins_100g'],
            'salt_100g': product_data['salt_100g'],
            'sodium_100g': product_data['sodium_100g'],
            'sugars_100g': product_data['sugars_100g'],
        }

        # Handle ingredient_origins using one-hot encoding
        if 'ingredient_origins' in product_data:
            ingredient_origins = product_data['ingredient_origins']
            mlb = MultiLabelBinarizer()
            ingredient_origins_encoded = mlb.fit_transform([ingredient_origins.keys()])
            for i, origin in enumerate(mlb.classes_):
                product_features[f'origin_{origin}'] = ingredient_origins_encoded[0][i]

        features.append(product_features)

        if 'ecoscore_grade' in product_data:
            labels.append(product_data['ecoscore_grade'])
        else:
            labels.append(None)  # or handle the missing value in an appropriate way

    return features, labels


def train_model(features, labels):
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(max_iter=500)  # Increase max_iter value to 500
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')

    return model, vectorizer, f1_macro


# Preprocess the training data
train_data = preprocess_data('train_products.json')
train_features, train_labels = extract_features(train_data)

# Train the MLPClassifier model
model, vectorizer, f1_macro = train_model(train_features, train_labels)

# Preprocess the test data
test_data = preprocess_data('test_products.json')
test_features, _ = extract_features(test_data)

# Generate predictions using the MLPClassifier model
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
