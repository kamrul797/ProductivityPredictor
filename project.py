from tkinter import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ord_enc = OrdinalEncoder()
df = pd.read_csv("data.csv")

headers = ["Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced", "Prod"]
df["Care-taker_WhileWorking"] = ord_enc.fit_transform(df[["Care-taker_WhileWorking"]])
df["SP_Supp"] = ord_enc.fit_transform(df[["SP_Supp"]])
df["Using_DCC"] = ord_enc.fit_transform(df[["Using_DCC"]])
df["ComfyLeave_ChildSick"] = ord_enc.fit_transform(df[["ComfyLeave_ChildSick"]])
df["MatPat_Leave"] = ord_enc.fit_transform(df[["MatPat_Leave"]])
df["Flex_Whours"] = ord_enc.fit_transform(df[["Flex_Whours"]])
df["Eval_WorkProduced"] = ord_enc.fit_transform(df[["Eval_WorkProduced"]])
df["Prod"] = ord_enc.fit_transform(df[["Prod"]])
#print(df.info())
#print(df.head())
features = ["Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced"]
labels = ["Prod"]
X = df[features]
y = df[labels]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.05)
print(X_test)
#.............................................................................DECISION TREE>>>>>>
#dtc = DecisionTreeClassifier(max_depth=5)
#dtc.fit(X_train,y_train)
#pred = dtc.predict(X_test)

l1=['Yes','No']
#List of Productivity Feature is listed in list of two.
prod=['Increased','Decreased']
l2=[]
for i in range(0,len(features)):
    l2.append(0)
    
def DecisionTree():
    from sklearn import tree
    dtc = tree.DecisionTreeClassifier() 
    dtc = dtc.fit(X,y)
    from sklearn.metrics import accuracy_score
    y_pred=dtc.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    #print(accuracy_score(y_test, y_pred,normalize=False))
    pfeatures = [Feature1.get(),Feature2.get(),Feature3.get(),Feature4.get(),Feature5.get(),Feature6.get(),Feature7.get()]
    #print(pfeatures)
    
    for z in range(0,len(pfeatures)):
        if(pfeatures[z]=='Yes'):
            l2[z]=1
        else:
            l2[z]=0

    inputtest = [l2]
    print(inputtest)
    predict = dtc.predict(inputtest)
    predicted=predict[0] 
    print(predicted)

    if(predicted==1.0):
        t1.delete("1.0", END)
        t1.insert(END, "Productivity Will Increase")
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Productivity Will Decrease")        

#.............................................................................

#accuracy = accuracy_score(y_test,pred)
#matrix = confusion_matrix(y_test, pred)

#print("Decision Tree : ", accuracy) 
#print("Decision Tree : ", matrix) 

# GUI stuff..............................................................................
        
root = Tk()
root.wm_title("Working Productivity Predictor")
root.configure(background='white')
Feature1 = StringVar()
Feature1.set("Choose Option")
Feature2 = StringVar()
Feature2.set("Choose Option")
Feature3 = StringVar()
Feature3.set("Choose Option")
Feature4 = StringVar()
Feature4.set("Choose Option")
Feature5 = StringVar()
Feature5.set("Choose Option")
Feature6 = StringVar()
Feature6.set("Choose Option")
Feature7 = StringVar()
Feature7.set("Choose Option")
Name = StringVar()

S1Lb = Label(root, text="Do you have care taker for your children while working?", fg="black")
S1Lb.config(font=("poppins",10,"bold"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)
S2Lb = Label(root, text="Do you have supportive spouse?", fg="black")
S2Lb.config(font=("poppins",10,"bold"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)
S3Lb = Label(root, text="Are you using day care center for your child?", fg="black")
S3Lb.config(font=("poppins",10,"bold"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)
S4Lb = Label(root, text="Are you comfortable taking leave while your child is sick?", fg="black")
S4Lb.config(font=("poppins",10,"bold"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)
S5Lb = Label(root, text="Does your company provide maternity/paternity leave?", fg="black")
S5Lb.config(font=("poppins",10,"bold"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)
S6Lb = Label(root, text="Do you have flexible working hours?", fg="black")
S6Lb.config(font=("poppins",10,"bold"))
S6Lb.grid(row=12, column=0, pady=10, sticky=W)
S7Lb = Label(root, text="Is your performance evaluation based on the work you produced?", fg="black")
S7Lb.config(font=("poppins",10,"bold"))
S7Lb.grid(row=13, column=0, pady=10, sticky=W)

lrLb = Label(root, text="Productivity Prediction:", fg="black")
lrLb.config(font=("poppins",10,"bold"))
lrLb.grid(row=15, column=0, pady=10,sticky=W)
OPTIONS = sorted(l1)

S1 = OptionMenu(root, Feature1,*OPTIONS)
S1.grid(row=7, column=1)
S2 = OptionMenu(root, Feature2,*OPTIONS)
S2.grid(row=8, column=1)
S3 = OptionMenu(root, Feature3,*OPTIONS)
S3.grid(row=9, column=1)
S4 = OptionMenu(root, Feature4,*OPTIONS)
S4.grid(row=10, column=1)
S5 = OptionMenu(root, Feature5,*OPTIONS)
S5.grid(row=11, column=1)
S6 = OptionMenu(root, Feature6,*OPTIONS)
S6.grid(row=12, column=1)
S7 = OptionMenu(root, Feature7,*OPTIONS)
S7.grid(row=13, column=1)

dst = Button(root, text="Predict", command=DecisionTree,bg="green",fg="white")
dst.config(font=("poppins",12,"bold"))
dst.grid(row=10, column=3,padx=10)
t1 = Text(root, height=1, width=25,bg="white",fg="black")
t1.config(font=("poppins",10,"bold"))
t1.grid(row=15, column=1, padx=10)

root.mainloop()