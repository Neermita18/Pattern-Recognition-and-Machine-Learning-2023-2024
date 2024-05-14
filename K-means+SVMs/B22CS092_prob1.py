import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
import pandas as pd
from PIL import Image

image_path = 'test.png'
image = Image.open(image_path)
image_np = np.array(image)
print(image_np.shape) #512x512x3

image_reshaped = image_np.reshape(-1, 3)
print(image_reshaped.shape)

    # Shuffle the pixels
image_reshaped_sample = shuffle(image_reshaped, random_state=0)[:image_reshaped.shape[0]]
#used later

# def computeCentroid(imager):
#     df= pd.DataFrame(imager)
#     # print(df)
#     c1= df[0].mean()
#     c2= df[1].mean()
#     c3= df[2].mean()
#     centroid= [c1, c2, c3]
#     sum1= df[0].sum()
#     print(sum1)
#     print(f"Centroid is {centroid}")
#     c= np.mean(image_reshaped, axis=0)
#     # print(f"Centroid from another function {c}")
#     return centroid
 
def computeCentroid(imager):   
    return np.mean(imager, axis=0)
    
computeCentroid(image_reshaped)
print(image_reshaped.size)


def add_spatial_features(image_np, scale=0.1):
    print(image_np.shape)
    m, n, _ = image_np.shape
    X, Y = np.meshgrid(range(n), range(m))
    features = np.c_[image_np.reshape(-1, 3), scale * X.flatten(), scale * Y.flatten()]
    return features


def mykmeans(X, k, threshold=1e-4):
    print(X.shape)
    
    randcentroids = X[np.random.choice(X.shape[0], k, replace=False)]
    # print(randcentroids)
    
    for _ in range(1000):
        #find centroid closest to each data point
        labels= pairwise_distances_argmin(X, randcentroids)
        
        # Update randcentroids
        new_randcentroids = np.array([computeCentroid(X[labels == i]) for i in range(k)])
        #thresholding?? 
        # Check for convergence
        if np.allclose(randcentroids, new_randcentroids, atol= threshold):
            break
        
        randcentroids = new_randcentroids
    
    return randcentroids, labels
   

print(image_reshaped.shape)
spacial_features = add_spatial_features(image_np)
# centroidsf, labels= mykmeans(image_reshaped,3, threshold=1e-4)
centroidsf, labels= mykmeans(spacial_features,3, threshold=1e-4)

print(centroidsf)
print(labels)
# labels = pairwise_distances_argmin(image_reshaped, centroidsf)

#Replace pixel values with centroids
compressed_pixels = centroidsf[labels][:, :3]

print(compressed_pixels.shape)

print(compressed_pixels)
# compressed_pixels = np.array([centroidsf[label] for label in labels])


reconstructed_image = compressed_pixels.reshape(image_np.shape)
compressed_image = reconstructed_image.astype(np.uint8)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].set_title("Compressed Image")
ax[1].imshow(compressed_image )
    
ax[1].axis('off')

plt.show()




kmeans = KMeans(n_clusters=3, random_state=42).fit(image_reshaped_sample)
ls = kmeans.predict(image_reshaped)
cs = kmeans.cluster_centers_

    # Replace each pixel with its corresponding centroid color
ci= cs[ls]

    # Reshape the compressed image back to its original shape
ci = ci.reshape(image_np.shape)

    # Convert the compressed image array to uint8 data type
ci= ci.astype(np.uint8)

    # Display the original and compressed images
fig, ax = plt.subplots(1, 2, figsize=(12, 1))
ax[0].imshow(compressed_image)
ax[0].set_title('Compressed Image from scratch({} colors)'.format(6))
ax[0].axis('off')

ax[1].imshow(ci)
ax[1].set_title('Compressed Image using sklearn kmeans ({} colors)'.format(6))
ax[1].axis('off')

plt.show()