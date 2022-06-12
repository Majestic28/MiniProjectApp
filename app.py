import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras import layers
from u2net_util import U2NetPrediction,RescaleT
from keras.preprocessing import image
from torchvision import transforms
import os
import time

height = 384
width = 384

cardamomClasses = ['Blight1000', 'Healthy_1000', 'Phylosticta_LS_1000']
grapeClasses = ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy']

prediction_label = ''

# @st.cache(allow_output_mutation=True)
def samplePrediction(model,img,classes,height,width):
  resized = cv2.resize(img,(height,width))
  image_value = np.expand_dims(resized, axis=0)
  prediction_scores = model.predict(image_value)
  prediction_label = classes[int(prediction_scores.argmax(axis=1))]
  st.write("Prediction: " + prediction_label)

@st.cache(allow_output_mutation=True)
def getefficientnet_v2_s_model(classes):  
  global height,width,batchSize
  height = 384
  width = 384

  model_name = 'efficientnet_v2_s'

  EfficientNetV2S=tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    weights=None,
    include_top=True,
    input_shape=None,
    pooling=None,
    classes = len(classes)
    )

  model_efficientnetv2s_base = tf.keras.Sequential([
    layers.Input(shape=(height, width, 3)),
    EfficientNetV2S,
    ])
  model = tf.keras.models.clone_model(model_efficientnetv2s_base)

  return model_name,model

@st.cache(allow_output_mutation=True)
def find_best_weight(model_name,dataset_name):
  rootdir = './trained_models_weight/'+model_name+'/'+dataset_name+'/'
  bestmodelWeight = os.listdir(rootdir)
  best = None
  bestValue = 100
  for string in bestmodelWeight:
    temp = float(string.split('-')[1].replace('loss','').replace('.hdf5',''))
    if temp<bestValue:
      best = string
      bestValue = temp
  return rootdir + best

@st.cache(allow_output_mutation=True)
def semanticSegmenterOutput():
    semanticSegmenter = U2NetPrediction()
    semanticSegmenter.transform = transforms.Compose([RescaleT(320),transforms.ToTensor()])
    return semanticSegmenter

@st.cache(allow_output_mutation=True)
def cardamomloadModel():
    #cardamom #v2s
    model_name,model = getefficientnet_v2_s_model(cardamomClasses)
    model.load_weights(find_best_weight('efficientnetv2-s','cardamom_dataset'))
    return model_name,model

@st.cache(allow_output_mutation=True)
def grapeloadModel():
    #cardamom #v2s
    model_name,model = getefficientnet_v2_s_model(grapeClasses)
    model.load_weights(find_best_weight('efficientnetv2-s','grape_dataset'))
    return model_name,model

semanticSegmenter = semanticSegmenterOutput()

@st.cache(allow_output_mutation=True)
def loading():
    with st.spinner("Loading..."):
        time.sleep(1)

project = st.sidebar.radio("Dashboard",["Predictors","About","U^2Net_Output"])

if project == "About":
    with st.spinner("Loading..."):
        time.sleep(1)
    navig = st.sidebar.radio("About",["Contributors","The APP"])
    if navig == "Contributors":
        st.title("Contributors of the app..")
        
        st.header("1. Gokul R")
        g1,g2 = st.columns([1,3])        
        g1.subheader("Gmail: gokulrajakalappan@gmail.com")
        st.header("2. Vishnu Nandakumar")
        v1,v2 = st.columns([1,3])        
        v1.subheader("Gmail: universalvishnu2001@gmail.com")
        st.header("3. Gowatam rao GS")
        gg1,gg2 = st.columns([1,3])
        gg1.subheader("Gmail: gowtam.rao@gmail.com")

if project == "U^2Net_Output":
    loading()
    st.title("U^2 Net Output")
    uploaded_file = st.file_uploader("Choose a image file",type=['jpg','jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        st.write("  Uploaded Image")
        st.image(image, channels="RGB") # Printing original image
        
        height_temp,width_temp = image.shape[:2]
        masked = semanticSegmenter.semanticSegmentation(image = image,apply_mask=True)
        
        st.write("Masked Image")
        st.image(masked, channels="RGB") # Printing 
        
        S = semanticSegmenter.S
        img_show = st.columns(len(S))
        index = 0
        for i in S:
            i = semanticSegmenter.normPRED(i[:,0,:,:]).squeeze().cpu().data.numpy() * 255
            i = cv2.resize(i,(width_temp,height_temp),interpolation = cv2.INTER_AREA)
            i = i.astype(np.uint8)
            i = cv2.resize(i,(width//2,height//2))
            i = cv2.cvtColor(i, cv2. COLOR_GRAY2RGB)
            img_show[index].image(i)
            index+=1



    

if project == "Predictors":
    loading()
    navig = st.sidebar.radio("Available Disease Predictor",["Cardamom Disease Predictor","Grape Disease Predictor"])
    predictorClasses = cardamomClasses


    if navig == "Cardamom Disease Predictor":
        st.title("Cardamom Disease Predictor")
        predictorClasses = cardamomClasses
        model_name,model = cardamomloadModel()

    if navig == "Grape Disease Predictor":
        st.title("Grape Disease Predictor")
        predictorClasses = grapeClasses
        model_name,model = grapeloadModel()

    uploaded_file = st.file_uploader("Choose a image file",type=['jpg','jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        height_temp,width_temp = image.shape[:2]
        masked = semanticSegmenter.semanticSegmentation(image = image,apply_mask=True)
        st.image(masked, channels="RGB")
        S = semanticSegmenter.S

        Genrate_pred,clear,bl = st.columns(3)
        gen = Genrate_pred.button("Generate Prediction")    
        # clr = clear.button("Clear",disabled=True)
        clr = False
        if gen:
            clr = clear.button("Clear",disabled=False)
            progress = st.progress(0) # intialize with 0
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            masked = cv2.resize(masked,(width_temp,height_temp))
            output = samplePrediction(model,masked,predictorClasses,height,width)
        if clr:
            prediction_label = ''
