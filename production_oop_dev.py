import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
from keras import Model
from keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pathlib
from sklearn.metrics.pairwise import linear_kernel


class fashion_recommendations:
    """ Production class for recommendations of fashion from similarity """

    def __init__(self, img_path, df_embeddings, styles_path):
        self.img_path = img_path
        self.df_embeddings = df_embeddings
        self.styles_path = styles_path

    # Helper functions
    def get_styles_df(self):
        """ Load a dataframe contains styles details and images """
        styles_df = pd.read_csv(
            self.styles_path, nrows=6000, error_bad_lines=False
        )  # Read 6000 product and drop bad lines
        styles_df["image"] = styles_df.apply(
            lambda x: str(x["id"]) + ".jpg", axis=1
        )  # Make image column contains (id.jpg)
        return styles_df

    def load_model(self):
        """ Load our model """
        vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=(100, 100, 3))
        vgg16.trainable = False
        vgg16_model = keras.Sequential([vgg16, GlobalMaxPooling2D()])
        return vgg16_model

    def predict(self, model, img_path):
        """ Load and preprocess image then make prediction """
        # Reshape
        img = image.load_img(self.img_path, target_size=(100, 100))
        # img to Array
        img = image.img_to_array(img)
        # Expand Dim (1, w, h)
        img = np.expand_dims(img, axis=0)
        # Pre process Input
        img = preprocess_input(img)
        return model.predict(img)

    def get_similarity(self):
        """ Get similarity of custom image """
        model = self.load_model()
        df_embeddings = self.df_embeddings
        sample_image = self.predict(model, self.img_path)
        df_sample_image = pd.DataFrame(sample_image)
        sample_similarity = linear_kernel(df_sample_image, df_embeddings)
        return sample_similarity

    def normalize_sim(self):
        """ Normalize similarity results """
        similarity = self.get_similarity()
        x_min = similarity.min(axis=1)
        x_max = similarity.max(axis=1)
        norm = (similarity - x_min) / (x_max - x_min)[:, np.newaxis]
        return norm

    def get_recommendations(self):
        """ Get recommended images """
        similarity = self.normalize_sim()
        df = self.get_styles_df()
        # Get the pairwsie similarity scores of all clothes with that one (index, value)
        sim_scores = list(enumerate(similarity[0]))

        # Sort the clothes based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 5 most similar clothes
        sim_scores = sim_scores[0:5]
        print(sim_scores)
        # Get the clothes indices
        cloth_indices = [i[0] for i in sim_scores]

        # Return the top 5 most similar products
        return df["image"].iloc[cloth_indices]

    def print_recommendations(self):
        """ Print the top 5 most similar products"""
        recommendation = self.get_recommendations()
        recommendation_list = recommendation.to_list()
        # recommended images
        plt.figure(figsize=(20, 20))
        j = 0
        for i in recommendation_list:
            plt.subplot(6, 10, j + 1)
            cloth_img = mpimg.imread("../input/fashion-product-images-dataset/fashion-dataset/" + "images/" + i)
            plt.imshow(cloth_img)
            plt.axis("off")
            j += 1
        plt.title("Recommended images", loc="left")
        plt.subplots_adjust(wspace=-0.5, hspace=1)
        plt.show()
        return


img_path = "../input/fashion-product-images-dataset/fashion-dataset/images/10037.jpg"
df_embeddings_csv = "path"  # Type embeddings csv file path
styles_path = "../input/fashion-product-images-dataset/fashion-dataset/styles.csv"
obj = fashion_recommendations(img_path, df_embeddings_csv, styles_path)
obj.print_recommendations()