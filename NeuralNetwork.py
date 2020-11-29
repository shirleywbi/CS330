from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import utils
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

"""
Each time we do `model.add` we are adding on one of those transformations
Scaling does not change # of features but OHE increases it
With a neural net, you specify the number of features after each transformation.
In the below, it goes from 1 to 10 to 15 to 1.
"""
model = Sequential()
model.add(Dense(10, input_dim=1, activation='tanh')) # Transformation 1
model.add(Dense(15, activation='tanh')) # transformation 2
model.add(Dense(1, activation='linear')) # linear regression
model.compile(loss='mean_squared_error', optimizer="adam")


# Using Pre-Trained networks
resnet = ResNet50(weights='imagenet') # ResNet50 is the architecture; ImageNet is the dataset

img_path = 'img/gelbart-michael-adam.jpg'
img = load_img(img_path, target_size=(224, 224))
plt.imshow(img)

x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = resnet.predict(x)
decode_predictions(preds, top=3)[0] # Outputs top 3 predictions


# Transfer Learning
data = pd.read_csv('data/dog-breed-labels.csv')
data = data[:2000]
data['image_path'] = data.apply( lambda row: (os.path.join("data/dog-breed-train", row["id"] + ".jpg") ), axis=1)
data.head()


target_labels = data['breed']
total_classes = len(set(target_labels))
print("number of dog breeds:", total_classes)

images = np.array([img_to_array(
                    load_img(img, target_size=(256,256))
                    ) for img in data['image_path'].values.tolist()])
images = images.astype('float32')/255.0
X_train, X_test, y_train, y_test = train_test_split(images, target_labels, 
                                                    stratify=np.array(target_labels),
                                                    random_state=42)

base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
feature_extractor = Model(inputs=base_inception.input, outputs=GlobalAveragePooling2D()(base_inception.output))
Z_train = feature_extractor.predict(X_train) # we think of this as .transform()
Z_test = feature_extractor.predict(X_test)

lr = LogisticRegression(multi_class="multinomial", solver="lbfgs")
lr.fit(Z_train, y_train)
lr.score(Z_train, y_train)
lr.score(Z_test, y_test)
