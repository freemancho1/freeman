import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from ai.algorithms.gan import SimpleGAN
from utils.support_tf import LogLevelManager as llm


class NumberGenerator:
    
    def __init__(self):
        llm.set(2)
        self.model = []
        (self.data, self.label), (_, _) = tf.keras.datasets.mnist.load_data()
        self.epochs = 10000
        self.batch_size = 128
        self.latent_z_dim = 100
        self.num_display_log = 5
        
    def train(self, digit):
        train_x = self.data[np.where(self.label == digit)]
        gan = SimpleGAN(train_x, 
                        epochs=self.epochs, batch_size=self.batch_size, 
                        latent_z_dim=self.latent_z_dim, num_display_log=self.num_display_log,
                        tf_log_level=2)
        gan.fit()
        gan.display_last_generation_images()
        return gan
            
    def fit(self):
        start = datetime.now()
        print(f"훈련 시작: {start}")
        for i in range(10):
            print(f"숫자 {i} 훈련")
            self.model.append(self.train(i))
        end = datetime.now()
        print(f"훈련 종료: {end}, 전체 진행시간: {end-start}")
            
    def generator(self, *digits):
        gen_images = []
        for digit in digits:
            gen_images.append(self.model[digit].create_number())
            
        grid_rows = 1
        grid_cols = len(gen_images)
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_rows, grid_cols), sharey=True, sharex=True)
        display_idx = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                axs[i,j].imshow(gen_images[display_idx, :, :, 0], cmap="gray")
                axs[i,j].axis("off")
                display_idx += 1