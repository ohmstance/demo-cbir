import cv2
import torch
import io
import os
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from torchvision.models import mobilenet_v3_small
from PIL import Image

from tqdm.auto import tqdm
from utils import image_show

# Silence user warning by using a non-interactive backend
matplotlib.use('agg')

def update_features(db_path="features.pickle", img_dir="images"):
    """Update features in database according to images in image directory
    """
    
    # Get image filenames
    filenames = [fn for fn in os.listdir("images") if os.path.isfile(f"images/{fn}")]

    # Creates a list of dict in the format of ...
    # [{'filename': <file name of image>, 'image': <numpy array of pixel values>}, ...]
    images = []
    for filename in filenames:
        image_path = f"images/{filename}"
        with Image.open(image_path) as img:
            img = np.array(img, dtype=bool)
        images.append({'filename': filename, 'image': img})
        
    for index, image in enumerate(images):
        image = image['image']
        image = image.astype(np.uint8) # Convert to grayscale

        # Morphologically open image
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

        image = image.astype(bool) # Convert to binary
        image = Image.fromarray(image) # Convert to PIL object

        # Resize to 224 x 224
        image = image.resize((224, 224), resample=Image.Resampling.LANCZOS)

        # Convert to RGB for input into CNN
        image = image.convert('RGB')

        # Add gaussian blur
        image = gaussian_blur(image, (13, 13))

        image = np.array(image) # Convert back to numpy array
        images[index]['image'] = image

    # Convert to tensor and normalize according to mean and standard deviation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.8, 0.9, 0.9), (0.58, 0.5, 0.5))
    ])

    model = mobilenet_v3_small(weights="DEFAULT")
    model.classifier = torch.nn.Identity() # model.classifier[:1] # torch.nn.Identity()
    model.eval()
    model = torch.jit.script(model) # JIT optimize model
    model = torch.jit.optimize_for_inference(model) # Fuse batch norm + cnn layers
        
    features = {}
    with torch.inference_mode():
        for img in tqdm(images):
            filename = img['filename']
            image = img['image']

            # Tensorize, normalize, and add batch dimension
            image = transform(image)
            image = image.unsqueeze(0)

            # Extract features
            feature = model(image)

            # Remove batch dimension and convert features to numpy array
            feature = feature.squeeze(0)
            feature = feature.detach().cpu().numpy()

            features.update({filename: feature})

    with open(db_path, 'wb') as file:
        pickle.dump(features, file)
    
def get_similar(image, db_path="features.pickle", img_dir="images"):
    """Returns a list dict of of similar images with filenames
    """
    image = image.astype(np.uint8)
    
    # Morphologically open image
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

    image = image.astype(bool) # Convert to binary
    image = Image.fromarray(image) # Convert to PIL object

    # Resize to 224 x 224
    image = image.resize((224, 224), resample=Image.Resampling.LANCZOS)

    # Convert to RGB for input into CNN
    image = image.convert('RGB')

    # Add gaussian blur
    image = gaussian_blur(image, (13, 13))

    image = np.array(image) # Convert back to numpy array

    # Convert to tensor and normalize according to mean and standard deviation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.8, 0.9, 0.9), (0.58, 0.5, 0.5))
    ])
    
    model = mobilenet_v3_small(weights="DEFAULT")
    model.classifier = torch.nn.Identity() # model.classifier[:1] # torch.nn.Identity()
    model.eval()
    model = torch.jit.script(model) # JIT optimize model
    model = torch.jit.optimize_for_inference(model) # Fuse batch norm + cnn layers
    
    with torch.inference_mode():
        # Tensorize, normalize and add batch dimension
        image = transform(image)
        image = image.unsqueeze(0)

        # Extract features
        feature = model(image)

        # Remove batch dimension and convert features to numpy array
        feature = feature.squeeze(0)
        feature = feature.detach().cpu().numpy()

    out_feature = feature
    
    # Load features of images in database
    with open(db_path, 'rb') as file:
        features = pickle.load(file)

    # Calculate Euclidean difference between input feature and features in database
    dist_index = []
    for filename, feature in features.items():
        dist = np.linalg.norm(out_feature-feature)
        dist_index.append({'filename': filename, 'distance': dist})

    # Sort by distance in ascending order
    dist_index = sorted(dist_index, key = lambda x: x['distance'])
    
    filenames = [dist['filename'] for dist in dist_index]
    distances = [dist['distance'] for dist in dist_index]
    images = []

    for dist in dist_index[0: 20]:
        with Image.open(f"{img_dir}/{dist['filename']}") as img:
            img = np.array(img, dtype=bool)
            images.append(img)
            
    ret = []        
    for filename, distance, image in zip(filenames, distances, images):
        im = {
            'filename': filename,
            'image': image,
            'distance': distance
        }
        ret.append(im)
        
    return ret

def list_classes(db_path="features.pickle"):
    with open(db_path, 'rb') as file:
        features = pickle.load(file)
        
    classes = [filename.split('-')[0] for filename in features.keys()]
    classes = sorted(list(set(classes)))
    
    return classes

def evaluate_database(db_path="features.pickle"):
    with open(db_path, 'rb') as file:
        features = pickle.load(file)
    
    precisions = []
    recalls = []
    tps, fps, fns, tns = [], [], [], []

    for filename, feature in features.items():
        # Calculate Euclidean difference between input feature and features in database
        dist_index = []
        for filename_db, feature_db in features.items():
            dist = np.linalg.norm(feature-feature_db)
            dist_index.append({'filename': filename_db, 'distance': dist})

        # Sort by distance in ascending order
        dist_index = sorted(dist_index, key = lambda x: x['distance'])

        # Obtain class string from input feature
        img_class = filename.split('-')[0]

        # Obtain class string from retrieved images' filename
        ret_classes = [dist['filename'].split('-')[0] for dist in dist_index[:20]]

        # Number of images that should be retrieved and was
        tp = sum([img_class == ret_class for ret_class in ret_classes])

        # Number of images that shouldn't be retrieved but was
        fp = len(ret_classes) - tp

        # Number of images that should be retrieved but wasn't
        fn = sum([True for fn in features.keys() if img_class in fn]) - tp

        # Number of images that shouldn't be retrieved and wasn't
        tn = sum([True for fn in features.keys() if img_class not in fn]) - fp

        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        # precisions.append(tp / (tp + fp))
        # recalls.append(tp / (tp + fn))

    tps, fps, fns, tns = sum(tps), sum(fps), sum(fns), sum(tns)
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    
    return precision, recall

def evaluate_search(in_class, retrieved, db_path="features.pickle"):
    with open(db_path, 'rb') as file:
        features = pickle.load(file)
    
    # Obtain class string from image path
    img_class = in_class

    # Obtain class string from retrieved images' filename
    ret_classes = [r['filename'].split('-')[0] for r in retrieved[:20]]

    # Number of images that should be retrieved and was
    tp = sum([img_class == ret_class for ret_class in ret_classes])

    # Number of images that shouldn't be retrieved but was
    fp = len(ret_classes) - tp

    # Number of images that should be retrieved but wasn't
    fn = sum([True for fn in features.keys() if img_class in fn]) - tp

    # Number of images that shouldn't be retrieved and wasn't
    tn = sum([True for fn in features.keys() if img_class not in fn]) - fp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return precision, recall

def plot_pr_graph(in_class, retrieved, db_path="features.pickle"):
    # Refer to this stackoverflow answer regarding plotting Precision-Recall curve for CBIR
    # https://stackoverflow.com/questions/25799107/how-do-i-plot-precision-recall-graphs-for-content-based-image-retrieval-in-matla

    # Obtains what class the image is
    img_class = in_class

    # Makes a list of sorted distances for all 400 images in database
    distance_list = [dist['distance'] for dist in retrieved]

    # A list of indexes of where the image class is located in distance_list
    relevant_indexes = [i+1 for i, dist in enumerate(retrieved) if img_class in dist['filename']]

    # [1, 2, 3, 4, 5, ..., N] where N is the total num of relevant image class in database
    thresholds = list(range(1, len(relevant_indexes)+1))

    # Gets precision value if for every x number of images of relevant class are retrieved
    # Precision = <relevant images retrieved> / (<relevant images retrieved> + <irrelevant images retrieved>)
    precision = np.array(thresholds) / np.array(relevant_indexes)

    # Creates multiple recall thresholds for x-axis where for x num of images retrieved, with y recall, what is the precision
    # Recaall = <relevant images retrieved> / <relevant images in database>
    recall = np.array(thresholds) / len(thresholds)

    plt.plot(recall, precision, 'b.-')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title(f'Precision-Recall Graph')
    plt.axis([0, 1.05, 0, 1.05]) # Set x-axis and y-axis range
    
    mem_file = io.BytesIO()
    plt.savefig(mem_file)
    plt.close()
    
    return mem_file