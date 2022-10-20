import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from datetime import datetime
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten

from freeman.utils.date import eta
from freeman.utils.support_tf import LogLevelManager as llm


class NumberGAN:
    
    def __init__(self, data, epochs=10000, batch_size=128, latent_z_dim=50, num_display_log=10, tf_log_level=2):
        llm.set(tf_log_level)
        
        self.data = None
        self.data_size = None 
        self.input_shape = None
        self.input_size = None
        self.preprocessing_data(data)
        
        self.epochs = epochs
        self.digit_of_epochs = len(str(self.epochs))
        self.batch_size = batch_size
        self.latent_z_dim = latent_z_dim
        self.num_display_log = num_display_log
        self.epochs_display_log = self.epochs // self.num_display_log
        
        self.is_training = False
        self.generator = None
        
    
    def preprocessing_data(self, data):
        data_dim = data.ndim
        if data_dim not in [3, 4]:
            raise TypeError(f"3 또는 4차원의 입력 데이터가 필요합니다. 입력된 데이터는 {data_dim} 차원 입니다.")
        self.data = data / 127.5 - 1
        if data_dim == 3:
            self.data = np.expand_dims(self.data, axis=3)
        self.data_size = len(self.data)
        self.input_shape = self.data.shape[1:]
        self.input_size = np.prod(self.input_shape)
        
    def build_generator(self):
        model = Sequential([
            Dense(units=128, input_dim=self.latent_z_dim),
            LeakyReLU(alpha=0.01),
            Dense(units=self.input_size, activation="tanh"),
            Reshape(target_shape=self.input_shape)
        ])
        return model
    
    def build_discriminator(self):
        model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(units=128),
            LeakyReLU(alpha=0.01),
            Dense(units=1, activation="sigmoid") # 진짜/가짜 판별
        ])
        return model
    
    def build_gan(self, generator, discriminator):
        discriminator.trainable = False
        model = Sequential([
            generator,
            discriminator
        ])
        return model
    
    def fit(self):
        start_train = datetime.now()
        print(f"훈련시작: {start_train}\n"
              f"데이터: Shape={self.data.shape}, Epochs={self.epochs}, BatchSize={self.batch_size}")
        
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        discriminator.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        gan = self.build_gan(generator, discriminator)
        gan.compile(optimizer=Adam(), loss="binary_crossentropy")
        
        real_label, fake_label = np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))
                
        start_loop = datetime.now()
        for epoch in range(self.epochs):
            train_x, train_y = self.get_discriminator_train_data(generator, real_label, fake_label)
            d_loss, d_acc = discriminator.train_on_batch(train_x, train_y)
            train_x, train_y = self.get_latent_data(), real_label
            g_loss = gan.train_on_batch(train_x, train_y)
            
            if (epoch+1) % self.epochs_display_log == 0:
                end_loop = datetime.now()
                num_curr_loop = (epoch+1) // self.epochs_display_log
                print(f"{epoch+1: >6}({num_curr_loop: >2}/{self.num_display_log}), "
                      f"[D loss: {d_loss:8.6f}, accuracy: {d_acc:8.6f}], "
                      f"[G loss: {d_loss:8.6f}], "
                      f"Time[Curr: {end_loop-start_loop}, Total: {end_loop-start_train}, "
                      f"ETA: {eta(self.num_display_log-num_curr_loop, end_loop-start_loop)}]")
                start_loop = end_loop
        
        end_train = datetime.now()
        print(f"훈련 종료 - Time: {end_train}, "
              f"전체 소요시간: {end_train-start_train}")
        
        self.is_training = True
        self.generator = generator
            
    def get_discriminator_train_data(self, generator, real_label, fake_label):
        real_idx = np.random.randint(0, self.data_size, self.batch_size)
        real_images = self.data[real_idx]
        latent_data = self.get_latent_data()
        fake_images = generator.predict(latent_data, verbose=0)
        train_x = np.vstack([real_images, fake_images])
        train_y = np.vstack([real_label, fake_label])
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        return train_x[shuffle_idx], train_y[shuffle_idx]
    
    def get_latent_data(self):
        return np.random.normal(0, 1, (self.batch_size, self.latent_z_dim))
    
    def display_last_generation_images(self, grid_rows=4, grid_cols=4):
        if not self.is_training:
            raise RuntimeError("GAN 모델이 훈련되지 않았습니다.")
        
        gen_latent_noise = np.random.normal(0, 1, (grid_rows * grid_cols, self.latent_z_dim))
        gen_images = self.generator.predict(gen_latent_noise, verbose=0)
        gen_images = gen_images * .5 + .5
        
        _, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_rows, grid_cols), sharey=True, sharex=True)
        display_idx = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                axs[i,j].imshow(gen_images[display_idx, :, :, 0], cmap="gray")
                axs[i,j].axis("off")
                display_idx += 1
                
    def create_number(self):
        if not self.is_training:
            raise RuntimeError("GAN 모델이 훈련되지 않았습니다.")
        
        latent_noise = np.random.normal(0, 1, (1, self.latent_z_dim))
        gen_image = self.generator.predict(latent_noise, verbose=0)
        gen_image = (gen_image + 1) * 127.5
        return gen_image[0]
    
    
class NumberGenerator:
    
    def __init__(self, epochs=100, batch_size=128, latent_z_dim=50, num_display_log=10, tf_log_level=2):
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
        gan = NumberGAN(train_x, 
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
        