import urllib.request
import zipfile
import click
##import imageio
##from skimage import io
from sklearn.linear_model import LogisticRegression
from skimage import io
from PIL import Image
from skimage import color, exposure, transform
import numpy, os
import glob, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas
import pickle
from matplotlib import interactive
############################333
import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



@click.command()
@click.option('-m',default='',help='choose the model: LRscikit or LRtensor or LeNet')
@click.option('-d',default='',help="Path where is the DataSet")
@click.argument('command')
def data(command,m,d):
        if command == 'download':
                click.echo('Loading Dataset ---')
                response = urllib.request.urlretrieve("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip", "dataset.zip")
	
                with zipfile.ZipFile('dataset.zip',"r") as z:
                        z.extractall("E:\German-Traffic-Signs-Detector\images")
        elif command=="infer":
                if m=="LRscikit" and d!='':
                        inferenceModel_scikit(d)
                elif m=="LRtensor" and d!='':
                        imgs,labels=readImageAndProccess(d)
                        imgs = np.array(imgs, dtype='float32')
                        X = np.reshape(imgs, (imgs.shape[0], -1))
                        
                        inferenceModel_tensor(d,X,labels)
                elif m=="LeNet" and d!='':
                        imgs,labels=readImageAndProcces2(d)
                        imgs_nparray = np.array(imgs, dtype='float32')
                        X = np.reshape(imgs_nparray, (imgs_nparray.shape[0], -1))
                        image_size=32
                        num_channels=3
                        dataset = X.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
                        inferenceModel_lenet(d,dataset,labels)
                        
                
                
                        
        elif command=="train":
                if m=="LRscikit" and d!='':
                        click.echo("Training...")
                        imgs,labels=readImageAndProccess(d)
                        print(train_LRscikit(imgs,labels))
                        ##print(train_LRscikit)
                elif m=="LRtensor" and d!='':
                        click.echo("Training...")
                        imgs,labels=readImageAndProccess(d)
                        print(train_LRtensor(imgs,labels))
                elif m=="LeNet" and d!='':
                        click.echo("Training...")
                        imgs,labels=readImageAndProcces2(d)
                        print(Lenet_train(imgs,labels))
                        
                        
                        
                
        elif command=="test":
                if m=="LRscikit" and d!='':
                        click.echo("Testing...")
                        imgs,labels=readImageAndProccess(d)
                        print(test_LRscikit(imgs,labels))
                elif m=="LRtensor" and d!='':
                        click.echo("Testing...")
                        imgs,labels=readImageAndProccess(d)
                        print(test_LRtensor(imgs,labels))
                elif m=="LeNet" and d!='':
                        click.echo("Testing...")
                        imgs,labels=readImageAndProcces2(d)
                        print(Lenet_test(imgs,labels))
                        
       
                
        else:
                click.echo("Missing arguments")


def preprocess_img(img):
    IMG_SIZE = 35
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE),mode='constant')
    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])

## root_dir :  Need to be a PATH with /(slash) not \ (backslash) example: C:/Users/Usuario/Documents/German-Traffic-Signs-Detector/images/FullIJCNN2013
def readImageAndProccess(root_dir):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    ##print(all_img_paths)
    ##np.random.shuffle(all_img_paths)
    ##print(all_img_paths)
    for img_path in all_img_paths:
        ##print(img_path)
        img_path2=img_path.replace("\\","/")
        ##print(img_path2) io.imread                          
        img = preprocess_img(Image.open(img_path))
        label = get_class(img_path2)
        imgs.append(img)
        labels.append(label)
    ##print(labels) 

    
    
    ##print(X2.shape)
    ##print(nX2.shape)
    # Make one hot targets
    ##Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    ##print(len(labels))
    return imgs,labels

def readImageAndProcces2(root_dir):
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    for img_path in all_img_paths:
        img_path2=img_path.replace("\\","/")                     
        img = preprocess32(Image.open(img_path))
        label = get_class(img_path2)
        imgs.append(img)
        labels.append(label)
    return imgs,labels

def preprocess32(img):
    IMG_SIZE = 32
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return img
        
def saveModel(model):
    filename = 'LRscikitmodel.sav'
    pickle.dump(model, open("models/model1/saved/"+filename, 'wb'))
    
def train_LRscikit(imgs,labels):
    X2 = np.array(imgs, dtype='float32')
    nX2 = np.reshape(X2, (X2.shape[0], -1))
    x_train, x_test, y_train, y_test = train_test_split(nX2,labels,test_size=0.2,random_state=0)
    logisticRgre = LogisticRegression()
    logisticRgre.fit(x_train,y_train)    
    saveModel(logisticRgre)
    s = logisticRgre.score(x_test,y_test)
    accuracy = s * 100
    
    return accuracy

def test_LRscikit(imgs,labels):
    X2 = np.array(imgs, dtype='float32')
    nX2 = np.reshape(X2, (X2.shape[0], -1))
    x_train, x_test, y_train, y_test = train_test_split(nX2,labels,test_size=0.2,random_state=0)
    filename = 'LRscikitmodel.sav'
    loaded_model = pickle.load(open("models/model1/saved/"+filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    accuracy = result * 100
    
    return accuracy




def inferenceModel_scikit(root_dir):
        filename = 'LRscikitmodel.sav'
        loaded_model = pickle.load(open("models/model1/saved/"+filename, 'rb'))

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        totalOfimages=len(all_img_paths)
        
        cont=1
        for img_path in all_img_paths:
                ##print(len(img_path))
                img_path2=img_path.replace("\\","/")
                img = Image.open(img_path)
                img=inference_preprocess_img(img)     
                image_label= loaded_model.predict(img.reshape(1,-1))
                
                
                plt.figure(cont)
                plt.title("Prediction Label: "+"0"+str(image_label[0]), fontsize = 12)
                imgplot = plt.imshow(img)
                if cont < totalOfimages:
                        interactive(True)
                else:
                        interactive(False)
                plt.show()
                cont=cont+1
                
                
                plt.show()



def inference_preprocess_img(img):
    IMG_SIZE = 35
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return img



def train_LRtensor(imgs,labels):
        imgs = np.array(imgs, dtype='float32')
        X = np.reshape(imgs, (imgs.shape[0], -1))
        
        #Onehot_encode Ylist
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_Ylist = onehot_encoder.fit_transform(integer_encoded)

        x_train, x_test, y_train, y_test = train_test_split(X,onehot_Ylist , test_size=0.20, random_state=0)
        ntrain=len(x_train)
        ntest=len(x_test)
        dim= x_train.shape[1]
        nclass=y_train.shape[1]
        # Parameters of Logistic Regression
        learning_rate   = 0.001
        training_epochs = 1000
        batch_size      = 10
        #create placeholders for features and labels
        X_ = tf.placeholder(tf.float32, [None, dim], name="x")
        Y_ = tf.placeholder(tf.float32, [None, nclass],name="y")
        #Set model weights
        W = tf.Variable(tf.zeros([dim, nclass]), name="weights")
        b = tf.Variable(tf.zeros([nclass]), name="bias")
        #  predict Y from X and w, b
        prediction = tf.matmul(X_, W) + b
        model = tf.nn.softmax(tf.matmul(X_, W) + b,name="model")
        # define loss function
        # use softmax cross entropy with logits as the loss function
        # compute mean cross entropy, softmax is applied internally
        cost_function = -tf.reduce_sum(Y_*tf.log(prediction)) 
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2( logits=prediction, labels=Y_)
        loss = tf.reduce_mean(entropy)
        #define training op

        optimizer =tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                num_batch = int(ntrain/batch_size)
                # Loop over all batches
                for i in range(num_batch): 
                    randidx = np.random.randint(ntrain, size=batch_size)
                    batch_xs = x_train[randidx, :]
                    batch_ys = y_train[randidx, :]
                    # Fit training using batch data
                    sess.run(optimizer, feed_dict={X_: batch_xs, Y_: batch_ys})
                    # Compute average loss
                    avg_cost += sess.run(cost_function, feed_dict={X_: batch_xs, Y_: batch_ys})/num_batch
            # test the model
            predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_, 1),name="predictions")
            accuracy = tf.reduce_mean(tf.cast(predictions, "float"),name="accuracy")
            Accuracy = accuracy.eval({X_: x_test, Y_: y_test})
            # Save the variables to disk.
            save_path = saver.save(sess, "models/model2/saved/LRtensor")
            
            ##print ("Accuracy:", accuracy.eval({X_: x_test, Y_: y_test}))
            sess.close()
        
        return Accuracy*100

def test_LRtensor(imgs,labels):
        imgs = np.array(imgs, dtype='float32')
        X = np.reshape(imgs, (imgs.shape[0], -1))
        #Onehot_encode Ylist
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_Ylist = onehot_encoder.fit_transform(integer_encoded)
        #####
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model2/saved/LRtensor.meta')
            loader.restore(sess, 'models/model2/saved/LRtensor')
            accuracy = inference_graph.get_tensor_by_name('accuracy:0')
            X_  = inference_graph.get_tensor_by_name('x:0')
            Y_  = inference_graph.get_tensor_by_name('y:0')
            Accuracy= accuracy.eval({X_: X , Y_: onehot_Ylist})
            sess.close()
        
        return Accuracy*100

def inferenceModel_tensor(root_dir,x_test,y_test):
        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        totalOfimages=len(all_img_paths)
        
        cont=1
        for img_path in all_img_paths:
                ##print(len(img_path))
                img_path2=img_path.replace("\\","/")
                img = Image.open(img_path)
                img=preprocess_img(img)
                x_partial=[x_test[cont-1]]
                #x_partial = np.array(img, dtype='float32')
                prediction=predict(x_partial)
                
                
                
                plt.figure(cont)
                plt.title("Prediction Label: "+"0"+str(prediction[0]), fontsize = 12)
                imgplot = plt.imshow(img)
                if cont < totalOfimages:
                        interactive(True)
                else:
                        interactive(False)
                plt.show()
                cont=cont+1
        
        
        
def predict(x_test):
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model2/saved/LRtensor.meta')
            loader.restore(sess, 'models/model2/saved/LRtensor')
            model  = inference_graph.get_tensor_by_name('model:0')
            X_  = inference_graph.get_tensor_by_name('x:0')
            prediction = tf.argmax(model, 1)
            prediction=prediction.eval(feed_dict={X_: x_test})
        return prediction



def Lenet_train(imgs,labels):

        image_size=32
        num_channels=3
        num_labels=43
        imgs_nparray = np.array(imgs, dtype='float32')
        X = np.reshape(imgs_nparray, (imgs_nparray.shape[0], -1))
        # One hot
        NUM_CLASSES = 43
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
        ##Rechape X array 
        dataset = X.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        ##Split the data
        x_train, x_test, y_train, y_test = train_test_split(dataset,Y, test_size=0.20, random_state=0)
        #Training set (970, 32, 32, 3) (970, 43)
        #Test set (243, 32, 32, 3) (243, 43)
        EPOCHS = 50
        BATCH_SIZE = 128

        X_ =  tf.placeholder (tf.float32, (None, 32, 32, 3), name="x")
        Y_ = tf.placeholder(tf.float32, [None, 43], name="y")
        rate = 0.001
        # training pipeline that uses the model
        logits=LeNet(X_)
        #Model for prediction
        model = tf.nn.softmax(logits,name="model")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)
        #Model evaluation
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(x_train)
    
    
            for i in range(EPOCHS):
                x_train, y_train = shuffle(x_train, y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={X_: batch_x, Y_: batch_y})
                    
                #validation_accuracy = evaluate(x_test, y_test)
            Accuracy=accuracy_operation.eval({X_: x_test, Y_: y_test})
            save_path = saver.save(sess, "models/model3/saved/LeNet")
            sess.close()
            return Accuracy*100

def Lenet_test(imgs,labels):
        image_size=32
        num_channels=3
        imgs_nparray = np.array(imgs, dtype='float32')
        X = np.reshape(imgs_nparray, (imgs_nparray.shape[0], -1))
        # One hot
        NUM_CLASSES = 43
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
        ##Rechape X array 
        X = X.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model3/saved/LeNet.meta')
            loader.restore(sess, 'models/model3/saved/LeNet')
            X_  = inference_graph.get_tensor_by_name('x:0')
            Y_  = inference_graph.get_tensor_by_name('y:0')
            accuracy = inference_graph.get_tensor_by_name('accuracy:0')
            Accuracy= accuracy.eval({X_: X , Y_: Y})
            sess.close()
        
        return Accuracy*100

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,3,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,43), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits


def inferenceModel_lenet(root_dir,x_test,y_test):
        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        totalOfimages=len(all_img_paths)
        
        cont=1
        for img_path in all_img_paths:
                ##print(len(img_path))
                img_path2=img_path.replace("\\","/")
                img = Image.open(img_path)
                img=preprocess_img(img)
                x_partial=[x_test[cont-1]]
                #x_partial = np.array(img, dtype='float32')
                prediction=predictlenet(x_partial)
                
                
                
                plt.figure(cont)
                plt.title("Prediction Label: "+"0"+str(prediction[0]) + " Real label: "+ str(y_test[cont-1]), fontsize = 12)
                imgplot = plt.imshow(img)
                if cont < totalOfimages:
                        interactive(True)
                else:
                        interactive(False)
                plt.show()
                cont=cont+1

def predictlenet(x_test):
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model3/saved/LeNet.meta')
            loader.restore(sess, 'models/model3/saved/LeNet')
            model  = inference_graph.get_tensor_by_name('model:0')
            X_  = inference_graph.get_tensor_by_name('x:0')
            prediction = tf.argmax(model, 1)
            prediction=prediction.eval(feed_dict={X_: x_test})
            return prediction

if __name__ == '__main__':
	data()
	
