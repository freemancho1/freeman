import pickle
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow.keras.datasets.mnist import load_data

from freeman.utils.support_tf import LogLevelManager as llm
from freeman.ai.algorithms.gan.simple_gan import SimpleGAN


class NumberGenerator:
    
    def __init__(self, epochs=20000, batch_size=128, latent_z_dim=100, num_display_log=10, tf_log_level=2):
        llm.set(tf_log_level)
        
        self.epochs = epochs
        self.digit_of_epochs = len(str(self.epochs))
        self.batch_size = batch_size
        self.latent_z_dim = latent_z_dim
        self.num_display_log = num_display_log
        self.tf_log_level = tf_log_level
        
        self.model = []
        (self.data, self.label), (_, _) = load_data()
        self.model_path_pk = "/home/freeman/projects/data/models/gan_digit/"
        self.model_path_tf = "/home/freeman/projects/data/models/gan_digit_tf/"
        
    def train(self, digit):
        train_x = self.data[np.where(self.label == digit)]
        gan = SimpleGAN(train_x, 
                        epochs=self.epochs, batch_size=self.batch_size, 
                        latent_z_dim=self.latent_z_dim, num_display_log=self.num_display_log,
                        tf_log_level=self.tf_log_level)
        gan.fit()
        gan.display_last_generation_images()
        return gan    

    def fit_all(self):
        start = datetime.now()
        print(f"훈련 시작: {start}")
        for i in range(10):
            print(f"숫자 {i} 훈련")
            model = self.train(i)
            with open(f"/home/freeman/projects/data/models/gan_digit/model{i}.pkl", "wb") as f:
                pickle.dump(model, f)
            self.model.append(model)
        end = datetime.now()
        with open(f"/home/freeman/projects/data/models/gan_digit/model_all.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f"훈련 종료: {end}, 전체 진행시간: {end-start}")
        
    def fit(self, digit):
        print(f"숫자 {digit} 훈련")
        model = self.train(digit)
        with open(f"/home/freeman/projects/data/models/gan_digit/model{digit}.pkl", "wb") as f:
            pickle.dump(model, f)
        self.model.append(model)
        
    def _generator(self, *digits):
        gen_images = []
        for digit in digits:
            gen_images.append(self.model[digit].create_number())

        grid_rows = 1
        grid_cols = len(gen_images)
        _, axs = plt.subplots(grid_rows, grid_cols, figsize=(10, 10), sharey=True, sharex=True)
        display_idx = 0
        for i in range(grid_cols):
            axs[i].imshow(gen_images[display_idx], cmap="gray")
            axs[i].axis("off")
            display_idx += 1
            
    def generator(self, *digits):
        if self.model == []:
            llm.set(self.tf_log_level)
            for i in range(10):
                with open(f"/home/freeman/projects/data/models/gan_digit/model{i}.pkl", "rb") as f:
                    load_model = pickle.load(f)
                    self.model.append(load_model)
        self._generator(*digits)