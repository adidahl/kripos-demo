# Data Science Platform Demo

A local data science platform with JupyterHub, MLflow, and MinIO.

## Components

- **JupyterHub**: Collaborative notebooks for data science (port 8000)
- **MLflow**: Experiment tracking and model registry (port 5001)
- **MinIO**: S3-compatible object storage (ports 9000/9001)
- **Shared Data Volume**: Mounted across services for seamless data access

## Usage Instructions

### Starting the Platform

```bash
cd ds-platform-demo
docker-compose up -d
```

### Accessing the Services

- **JupyterHub**: http://localhost:8000
  - Login with:
    - Username: `adi`
    - Password: `password`

- **MLflow**: http://localhost:5001
  - Connect from notebooks:
    ```python
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5001")
    ```

- **MinIO Console**: http://localhost:9001
  - Login with:
    - Access Key: `minioadmin`
    - Secret Key: `minioadmin`

### Shared Data

All services can access the shared data volume:
- JupyterHub: `/data`
- MLflow: `/mlruns`
- MinIO: `/data`

## Stopping the Platform

```bash
docker-compose down
```

## Examples

### Using MinIO from Jupyter

```python
import boto3
import pandas as pd

s3_client = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

# Upload data
s3_client.upload_file('data.csv', 'mybucket', 'data.csv')

# Load data
obj = s3_client.get_object(Bucket='mybucket', Key='data.csv')
df = pd.read_csv(obj['Body'])
```

### Tracking Experiments with MLflow

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set tracking server
mlflow.set_tracking_uri("http://mlflow:5000")

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
```

## Food Identification Demo

This example shows how to build a simple food image classifier using your data science platform.

### Step 1: Create a Jupyter Notebook

1. Log in to JupyterHub at http://localhost:8000
2. Create a new Python 3 notebook
3. Install required packages:

```python
!pip install tensorflow keras pillow boto3 mlflow
```

### Step 2: Set Up MinIO for Image Storage

```python
import boto3
from botocore.client import Config

# Create a MinIO client
s3_client = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Create a bucket for food images
bucket_name = 'food-images'
try:
    s3_client.create_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' created successfully")
except s3_client.exceptions.BucketAlreadyOwnedByYou:
    print(f"Bucket '{bucket_name}' already exists")
```

### Step 3: Download Food Images Dataset

```python
import os
import urllib.request
import tarfile
import shutil

# Create a directory for the dataset
!mkdir -p /tmp/food-data

# Download a simple food dataset (Food-101 subset)
# For demo purposes, we'll use a small subset of 3 classes
print("Downloading food images...")
classes = ['pizza', 'sushi', 'hamburger']
images_per_class = 100  # Limit for demo purposes

for food_class in classes:
    os.makedirs(f'/tmp/food-data/{food_class}', exist_ok=True)
    
    # You can use any image dataset, this is just an example
    # In a real demo, you might want to have the images pre-downloaded
    for i in range(1, images_per_class + 1):
        try:
            url = f'https://source-of-food-images/{food_class}/{i}.jpg'  # Replace with real source
            urllib.request.urlretrieve(url, f'/tmp/food-data/{food_class}/{i}.jpg')
            if i % 20 == 0:
                print(f"Downloaded {i} {food_class} images")
        except:
            print(f"Failed to download image {i} for {food_class}")
```

### Step 4: Upload Images to MinIO

```python
import glob

# Upload the images to MinIO
for food_class in classes:
    images = glob.glob(f'/tmp/food-data/{food_class}/*.jpg')
    
    for idx, image_path in enumerate(images):
        image_name = os.path.basename(image_path)
        s3_key = f'{food_class}/{image_name}'
        
        s3_client.upload_file(image_path, bucket_name, s3_key)
        
        if idx % 20 == 0:
            print(f"Uploaded {idx} {food_class} images to MinIO")
    
    print(f"Completed uploading {food_class} images")
```

### Step 5: Build a CNN Model for Food Classification

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Set up MLflow tracking
import mlflow
import mlflow.keras

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("food-classification")

# Download images from MinIO for training
for food_class in classes:
    os.makedirs(f'/tmp/train/{food_class}', exist_ok=True)
    os.makedirs(f'/tmp/val/{food_class}', exist_ok=True)
    
    # List objects in the bucket for this class
    objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f'{food_class}/')
    
    # Split images 80/20 for training/validation
    total_images = len(objects.get('Contents', []))
    train_count = int(total_images * 0.8)
    
    for idx, obj in enumerate(objects.get('Contents', [])):
        key = obj['Key']
        filename = os.path.basename(key)
        
        # Skip non-image files
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Determine if this goes to train or validation
        if idx < train_count:
            local_path = f'/tmp/train/{food_class}/{filename}'
        else:
            local_path = f'/tmp/val/{food_class}/{filename}'
            
        # Download the image
        s3_client.download_file(bucket_name, key, local_path)
    
    print(f"Downloaded {food_class} images for training and validation")

# Set up data generators
img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/tmp/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    '/tmp/val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model using transfer learning with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Start an MLflow run for tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "MobileNetV2")
    mlflow.log_param("img_height", img_height)
    mlflow.log_param("img_width", img_width)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("classes", classes)
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    # Log metrics
    for epoch in range(10):
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
    
    # Log model
    mlflow.keras.log_model(model, "model")
    
    # Save model to MinIO
    model.save('/tmp/food_classifier_model.h5')
    s3_client.upload_file('/tmp/food_classifier_model.h5', bucket_name, 'models/food_classifier_model.h5')
    print("Model saved to MinIO")
```

### Step 6: Test the Model with New Images

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Function to predict food class
def predict_food(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence

# Test with new images
test_images = [
    '/tmp/test_pizza.jpg',
    '/tmp/test_sushi.jpg',
    '/tmp/test_hamburger.jpg'
]

# You can upload test images through the notebook or download from MinIO
for test_image in test_images:
    food_class, confidence = predict_food(test_image)
    print(f"Image: {test_image}")
    print(f"Prediction: {food_class} with {confidence:.2f} confidence")
    print("---")
```

### Step 7: View Experiment Results in MLflow

1. Open MLflow at http://localhost:5001
2. Click on the "food-classification" experiment
3. Explore the runs, metrics, and parameters
4. Compare different model versions or hyperparameters

### Step 8: Deploy the Model

```python
import mlflow.pyfunc

# Load the model from MLflow
model_uri = "runs:/<run_id>/model"  # Replace <run_id> with actual run ID from MLflow
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Create a simple prediction service
def predict_food_service(image_path):
    # Preprocess the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = loaded_model.predict(img_array)
    predicted_class = classes[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

# This function could be used in a Flask app or FastAPI service
```

This demo showcases how to:
1. Store and manage food images in MinIO
2. Build a transfer learning model in JupyterHub
3. Track experiments with MLflow
4. Deploy the model for predictions

Note: In a real demonstration, you might want to use a pre-downloaded dataset or a smaller subset of Food-101 to avoid downloading too many images during the demo. 