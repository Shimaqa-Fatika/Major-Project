#Importing Libraries
import os
import glob
import h5py
import pandas as pd
from PIL import Image
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import mean_absolute_error, r2_score

#Loading the model and weights
def load_model(model_path, weights_path):
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

#Image normalization
def create_img(path):
    im = Image.open(path).convert('RGB')
    im = np.array(im)
    im = im / 255.0
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
    im = np.expand_dims(im, axis=0)
    return im

#Loading the grount truth values for images, performing predictions using the model and storing predicted values
def process_images(image_paths, model):
    names, y_true, y_pred = [], [], []
    for image in image_paths:
        names.append(image)
        gt_file = image.replace('.jpg', '.h5').replace('images', 'ground_truth')
        with h5py.File(gt_file, 'r') as gt:
            groundtruth = np.asarray(gt['density'])
            true_count = np.sum(groundtruth)
        y_true.append(true_count)
        img = create_img(image)
        predicted_count = np.sum(model.predict(img))
        y_pred.append(predicted_count)
    return pd.DataFrame({'name': names, 'y_pred': y_pred, 'y_true': y_true})

# Paths and models setup
root = '/Users/apple/Desktop/CSRnet-master/'
models_info = {
    'A': {'test_path': os.path.join(root, 'ShanghaiTech/part_A_final/test_data', 'images'),
          'model_path': '/Users/apple/Desktop/CSRnet-master/models/Model.json', 'weights_path': '/Users/apple/Desktop/CSRnet-master/weights/model_A_weights.h5'},
    'B': {'test_path': os.path.join(root, 'ShanghaiTech/part_B_final/test_data', 'images'),
          'model_path': '/Users/apple/Desktop/CSRnet-master/models/Model.json', 'weights_path': '/Users/apple/Desktop/CSRnet-master/weights/model_B_weights.h5'}
}

#Storing all image paths into a single dataframe
all_data = pd.DataFrame()


for part in ['A', 'B']:
    info = models_info[part]
    model = load_model(info['model_path'], info['weights_path'])
    img_paths = glob.glob(os.path.join(info['test_path'], '*.jpg'))
    data = process_images(img_paths, model)
    data.to_csv(f'CSV/{part}_on_{part}_test.csv', sep=',')
    all_data = pd.concat([all_data, data])

# Calculate metrics
y_true = all_data['y_true']
y_pred = all_data['y_pred']
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Combined MAE: {mae}")
print(f"Combined R2 Score: {r2}")
