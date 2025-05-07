from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

main = Tk()
main.title("Early ADHD Detection in Children Using Machine Learning")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test, scaler, svm_cls
global dataset
proto_File = "Models/pose_deploy_linevec.prototxt"
weights_File = "Models/pose_iter_440000.caffemodel"
n_Points = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
in_Width = 368
in_Height = 368
threshold = 0.1
POSE_NAMES = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
              "RAnkle", "LHip", "LKnee", "LAnkle", "Chest", "Background"]

net = cv2.dnn.readNetFromCaffe(proto_File, weights_File)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

def uploadDataset(): 
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="ADHDDataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))

def processDataset():
    global dataset, X, Y, scaler
    text.delete('1.0', END)
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    indices = np.arange(X.shape[0]) #shuffling dataset values
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Dataset Processing, Shuffling & Normalization Completed\n\n")
    text.insert(END,"Normalized Dataset Values = "+str(X))

def splitDataset():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in Dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each record : "+str(X.shape[1])+"\n")
    text.insert(END,"Dataset Train & Test Split\n")
    text.insert(END,"80% dataset size used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset size used to test algorithms : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    label = ['Normal', 'ADHD Disease']
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = label, yticklabels = label, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(label)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def trainSVM():
    global svm_cls
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    svm_cls = svm.SVC(kernel="rbf", C = 12, probability=True, gamma="auto")
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

def predictADHD(testData):
    values = []
    testData = np.asarray(testData)
    values.append(testData)
    testData = np.asarray(values)
    testData = scaler.transform(testData)
    predict = svm_cls.predict(testData)
    return int(predict[0])

def detectDisease(frame):
    global net
    frame_Width = frame.shape[1]
    frame_Height = frame.shape[0]
    img = np.zeros((frame_Height,frame_Width,3), dtype=np.uint8)
    inp_Blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_Width, in_Height), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp_Blob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    points = []
    testData = []
    for i in range(n_Points):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frame_Width * point[0]) / W
        y = (frame_Height * point[1]) / H
        if prob > threshold :
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
            testData.append(x)
            testData.append(y)
        else :
            points.append(None)
            testData.append(0)
            testData.append(0)
    predict = predictADHD(testData)        
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        print(str(pair[0])+" "+str(pair[1])+" "+str(partA)+" "+str(partB))
        if points[partA] and points[partB]:
            cv2.line(img, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    return frame, img, predict

def imageDetect():
    text.delete('1.0', END)
    global scaler, svm_cls
    filename = filedialog.askopenfilename(initialdir="images")
    frame = cv2.imread(filename)
    frame, img, predict = detectDisease(frame)
    frame = cv2.resize(frame, (400, 400))
    img = cv2.resize(img, (400, 400))
    label = ['Normal', 'ADHD Disease']
    print(predict)
    cv2.putText(frame, 'Predicted As : '+label[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow("Pose Estimated Image", frame)
    cv2.imshow("Pose Image", img)
    cv2.waitKey(0)

def videoDetect():
    text.delete('1.0', END)
    global scaler, svm_cls
    filename = filedialog.askopenfilename(initialdir="videos")
    normal_count = 0
    abnormal_count = 0
    video = cv2.VideoCapture(filename)
    count = 0
    while(True):
        ret, frame = video.read()
        if ret == True:
            filename = "temp.png"
            frame, img, predict = detectDisease(frame)
            if predict == 0:
                normal_count += 1
            else:
                abnormal_count + 1                
            cv2.imshow("Estimated Pose", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            count = count + 1
            if count > 20:
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    if normal_count >= abnormal_count:
        text.insert(END,"Pose in Video Predicted as : NORMAL\n")
    else:
        text.insert(END,"Pose in Video Predicted as : ADHD Disease\n")
                

font = ('times', 15, 'bold')
title = Label(main, text='Children ADHD Disease Detection using Pose Extimation Technique')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload ADHD Pose Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

splitButton = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitButton.place(x=20,y=200)
splitButton.config(font=ff)

svmButton = Button(main, text="Train SVM Algorithm", command=trainSVM)
svmButton.place(x=20,y=250)
svmButton.config(font=ff)

imageDetectionButton = Button(main, text="Disease Detection from Test Image", command=imageDetect)
imageDetectionButton.place(x=20,y=300)
imageDetectionButton.config(font=ff)

videoDetectionButton = Button(main, text="Disease Detection from Video", command=videoDetect)
videoDetectionButton.place(x=20,y=350)
videoDetectionButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
