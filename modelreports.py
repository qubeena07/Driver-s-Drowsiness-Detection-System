from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

#loading of the data assets by data pre-processing methods
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=False,batch_size=28,target_size=(32,32),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)


BS= 28
TS=(32,32)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/test',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS

model = load_model('drow_model.h5')


#plotting of confusion matrix
Y_pred = model.predict(valid_batch, SPE+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(valid_batch.classes, y_pred))
print('Classification Report')
target_names = ['Closed', 'Open']
print(classification_report(valid_batch.classes, y_pred, target_names=target_names))
cm = confusion_matrix(valid_batch.classes, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#plotting of receiving operating characteristics(ROC)
y_pred_keras = model.predict(valid_batch,SPE+1)
y_pred = np.argmax(y_pred_keras, axis=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(valid_batch.classes, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()