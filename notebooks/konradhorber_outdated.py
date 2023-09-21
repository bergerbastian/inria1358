# class TIFImageGenerator(tf.keras.utils.Sequence):
#     def __init__(self, image_dir, mask_dir, batch_size, img_size):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.image_filenames = os.listdir(image_dir)

#     def __len__(self):
#         return len(self.image_filenames) // self.batch_size

#     def __getitem__(self, index):
#         batch_files = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
#         X = []  # 3 channels for RGB
#         # y = np.zeros((self.batch_size, *self.img_size, 1))  # 1 channel for grayscale
        
#         first_iteration = True
#         for file in batch_files:
#             tensor = tf.io.read_file(f'{self.image_dir}/{file}')
#             tensor = tfio.experimental.image.decode_tiff(tensor)
#             tensor = tf.pad(tensor, [[60,60], [60,60], [0,0]], mode='CONSTANT', constant_values=-1)
#             image_tensor = tf.expand_dims(tensor, axis=0)
#             if first_iteration:
#                 X = image_tensor
#                 first_iteration = False
#             else:
#                 X = tf.concat([X, image_tensor], axis=0)
            
#             # X.append(tensor)

#             # img_path = os.path.join(self.image_dir, file)
#             # mask_path = os.path.join(self.mask_dir, file)  # Assuming masks have the same filenames
            
#             # # Load images in RGB and masks in grayscale
#             # img = Image.open(img_path).convert('RGB').resize(self.img_size)
#             # mask = Image.open(mask_path).convert('L').resize(self.img_size)
            
#             # X[i] = np.array(img) / 255.0  # Normalize to [0, 1]
#             # y[i] = np.expand_dims(np.array(mask), axis=-1) / 255.0
#         X = tf.image.extract_patches(
#         X,
#         sizes = [1, 256, 256, 1],
#         strides = [1, 256, 256, 1],
#         rates = [1, 1, 1, 1],
#         padding = 'VALID'
#         )
#         X = tf.reshape(
#             X,
#             shape=(2000, 256, 256, 4)
#         )
#         X = X/255

#         return X


#     BATCH_SIZE = 5
# IMG_SIZE = 0
# images_path = '/home/konrad.horber/code/bergerbastian/inria1358/raw_data/aerial_images_inria1358/AerialImageDataset/train/images'
# ref_path = 0
# train_gen = TIFImageGenerator(images_path, ref_path, BATCH_SIZE, IMG_SIZE)

# variable = tf.image.extract_patches(
#     variable,
#     sizes = [1, 256, 256, 1],
#     strides = [1, 256, 256, 1],
#     rates = [1, 1, 1, 1],
#     padding = 'VALID'
# )
# variable = tf.reshape(
#     variable,
#     shape=(2000, 256, 256, 4)
# )
# variable = variable/255

# dirpath='/home/konrad.horber/code/bergerbastian/inria1358/raw_data/aerial_images_inria1358/AerialImageDataset/train/images'
# tensor_batch = []
# count = 0

# for i in os.listdir(dirpath)
#     for filename in os.listdir(dirpath):
#         # if count >= 10:
#         #     break
#         tiff = tf.io.read_file(f'{dirpath}/{filename}')
#         tensor = tfio.experimental.image.decode_tiff(tiff)
#         chicago1_tensor_pad = tf.pad(tensor, [[60,60], [60,60], [0,0]], mode='CONSTANT', constant_values=-1)
#         tensor_batch.append(tensor)
#         count += 1

#         chicago1_tensor_pad_batch_extract = tf.image.extract_patches(
#     chicago1_tensor_pad_batch,
#     sizes = [1, 256, 256, 1],
#     strides = [1, 256, 256, 1],
#     rates = [1, 1, 1, 1],
#     padding = 'VALID'
# )

# plt.figure(figsize=(10, 10))
# for imgs in chicago1_tensor_pad_batch_extract:
#     count = 0
#     for r in range(4):
#         for c in range(4):
#             ax = plt.subplot(4, 4, count+1)
#             plt.imshow(tf.reshape(imgs[r,c],shape=(256,256,4)).numpy().astype("uint8"))
#             count += 1