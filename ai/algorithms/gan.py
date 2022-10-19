import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 

from utils.support_tf import LogLevelManager as llm
from utils.date import eta


class SimpleGAN:
    
    def __init__(self, data, epochs=10000, batch_size=128, latent_z_dim=50, num_display_log=10, tf_log_level=2):
        SimpleGAN.init_env(tf_log_level)
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
        self.data = data / (data.max() / 2) - 1
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
        
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_rows, grid_cols), sharey=True, sharex=True)
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
        return gen_image
                
    @staticmethod
    def init_env(tf_log_level):
        llm.set(tf_log_level)
        