import sys, random

import tensorflow as tf

import numpy as np
import cv2

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QFileDialog
from design import Ui_MainWindow


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        
        #Presets.
        self.setWindowIcon(QtGui.QIcon('assets\icon.png'))

        self.ui.pushButton.setIcon(QIcon('assets/random-photo.png'))
        self.ui.pushButton.setIconSize(QtCore.QSize(24,24))
        self.ui.pushButton_2.setIcon(QIcon('assets/add-photo.png'))
        self.ui.pushButton_2.setIconSize(QtCore.QSize(24,24))
        self.ui.pushButton_4.setCheckable(True)
        
        self.ui.label_3.setPixmap(QPixmap("assets/no-image.png"))
        self.ui.label_7.setPixmap(QPixmap("assets/no-image.png"))
        self.ui.label_10.setPixmap(QPixmap("assets/model-info.png"))

        self.ui.label_2.setVisible(False)
        self.ui.label_10.setVisible(False)
        self.ui.checkBox.setEnabled(False)

        
        (x_train, y_train), (self.x_test, y_test) = tf.keras.datasets.mnist.load_data()
       
        self.model = tf.keras.models.load_model("model\mnist_model.h5")
        
        self.set_test=-1
        self.image_test=""
        self.gray_image=""

        self.ui.pushButton.clicked.connect(self.test_testset)
        self.ui.pushButton_2.clicked.connect(self.test_image)
        self.ui.pushButton_3.clicked.connect(self.model_prediction)
        self.ui.pushButton_4.clicked.connect(self.model_info)
        self.ui.checkBox.clicked.connect(self.digit_zoom)
        
    #Model test from test set.
    def test_testset(self):
       
        self.ui.label_7.setPixmap(QPixmap("assets/no-image.png"))
        
        if(self.ui.checkBox.isChecked()):
            self.ui.checkBox.setChecked(False)
        
        if(self.ui.checkBox.isEnabled()):
            self.ui.checkBox.setEnabled(False)

        if(self.ui.label_2.isVisible()):
            self.ui.label_2.setVisible(False)
        
        #Randam number selection (9999>=n>=0).
        selected_index=random.randint(0, 9999)
        
        selectted_testset_image=self.x_test[selected_index]
        selectted_testset_image = QImage(selectted_testset_image, selectted_testset_image.shape[1], selectted_testset_image.shape[0], QImage.Format_Grayscale8)
        
        self.set_test=selected_index
        
        self.show_original_image(selectted_testset_image)  

    #Model test from selected image.
    def test_image(self):
        
        if(self.ui.checkBox.isChecked()):
            self.ui.checkBox.setChecked(False)
        
        if(self.ui.checkBox.isEnabled()==False):
            self.ui.checkBox.setEnabled(True)

        if(self.ui.label_2.isVisible()):
            self.ui.label_2.setVisible(False)
       
        
        selectted_image, _ = QFileDialog.getOpenFileName(self, "Choose an image", "", "Image formats (*.png *.jpg *.jpeg)")
        
        if(selectted_image!=""):
            
            self.set_test=-1    

            self.show_original_image(selectted_image)
            self.image_oto_gray(selectted_image)
        else:
            self.ui.label_2.setVisible(True)
            self.ui.checkBox.setEnabled(False)
    
    
    def show_original_image(self, image):
        if(self.ui.pushButton_3.isEnabled()==False):
            self.ui.pushButton_3.setEnabled(True)
        
        self.ui.label_3.setPixmap(QPixmap(image))
    
    
    def show_processed_image(self, image):
        self.ui.label_7.setPixmap(QPixmap.fromImage(image))
        
   
    def image_oto_gray(self, image):
        self.gray_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        
        #Applying pixel change for image background. 
        if(self.gray_image[0,0]>=127):
            self.gray_image=255-self.gray_image
        
        
        self.image_test=self.gray_image
        
        self.show_processed_image(QImage(self.gray_image.data, self.gray_image.shape[1], self.gray_image.shape[0], self.gray_image.strides[0], QImage.Format_Grayscale8))

   
    def digit_zoom(self):
        if self.ui.checkBox.isChecked():
            image=self.gray_image
            
            image=255-image
            
            blur = cv2.GaussianBlur(image, (5,5), 0)
            
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            x,y,w,h = cv2.boundingRect(thresh)
            
            cropped_image = image[y:y+h, x:x+w]
            cropped_image=255-cropped_image
            
            h, w = cropped_image.shape[0:2]
            bottom_right_color=cropped_image[h-1, w-1]
            
            #Adding a border to cropped image.
            cropped_image=cv2.copyMakeBorder(cropped_image, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT, value=[int(bottom_right_color), int(bottom_right_color), int(bottom_right_color)])
            
            self.image_test=cropped_image
            
            self.show_processed_image(QImage(cropped_image.data, cropped_image.shape[1], cropped_image.shape[0], cropped_image.strides[0], QImage.Format_Grayscale8))
        
        else:
            self.image_test=self.gray_image
            self.show_processed_image(QImage(self.gray_image.data, self.gray_image.shape[1], self.gray_image.shape[0], self.gray_image.strides[0], QImage.Format_Grayscale8))
    
     
    def model_info(self):
        if self.ui.pushButton_4.isChecked():
            self.ui.label_10.setVisible(True)
        else:
            self.ui.label_10.setVisible(False)

   
    def model_prediction(self):
        if(self.set_test!=-1):
            image=self.x_test[int(self.set_test)]
        
        else:
            image = cv2.resize(self.image_test, (28, 28))
            
        #Test image reshapeing and normalizing.
        image = image.reshape(1, 28, 28, 1)
        image= image/255
        
        prediction = self.model.predict(image)
        
        #Predicted digit showing.
        self.ui.label_4.setText(str(np.argmax(prediction)))
        
        #Prediction rate showing.
        self.ui.label_5.setText(str("%.2f" %(max(prediction[0])*100))+" %")


def window():
    window=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(window.exec_())

window()