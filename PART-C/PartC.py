import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
def errorfunc(x_data,pred_theta,y_data):
    temp=np.dot(x_data,pred_theta)-y_data;
    temp=temp**2;
    return (float)(temp.sum()/(2.0*(len(x_data))));
       
def perf_measure(x_data,pred_theta,y_data,):
    temp=np.dot(x_data,pred_theta)-y_data;
    temp=temp**2;
    pred_theta=pred_theta**2;
    temp1= (float)(temp.sum()/((len(x_data))));
    return  math.sqrt(temp1);


def get_linear_features(data):
    rows,columns=data.shape;
    train_size= (int)(0.8*(rows));
    test_size=rows-train_size;
    featureScaling(data);
    train_data=data[0:train_size];
    test_data=data[train_size:];
    train_xData=train_data.iloc[:,0:columns-1].copy();
    train_xData['x0']=1;
    train_yData=train_data.iloc[:,columns-1:];
    test_xData=test_data.iloc[:,0:columns-1].copy();
    test_xData['x0']=1;
    test_yData=test_data.iloc[:,columns-1:];
    return train_xData,train_yData,test_xData,test_yData,columns;

def get_quadratic_features(data):
    rows,columns=data.shape;
    train_size= (int)(0.8*(rows));
    test_size=rows-train_size;
    for i in range(0,columns-1):
            buf="x%d_square"%(i+1);
            data[buf]=data.iloc[:,i].copy();
            data[buf]=data[buf]**2;
    
    featureScaling(data);
    train_data=data[0:train_size];
    test_data=data[train_size:];
    train_yData=train_data['price'].copy();
    train_yData=train_yData.values.reshape((train_size,1));
    test_yData=test_data['price'].copy();
    test_yData=test_yData.values.reshape((test_size,1));
    del train_data['price'];
    del test_data['price'];
    train_xData=train_data.iloc[:,0:].copy();
    train_xData['x0']=1;
    print(train_yData.shape);
    test_xData=test_data.iloc[:,0:].copy();
    test_xData['x0']=1;
   
    return train_xData,train_yData,test_xData,test_yData,2*columns-1;


def get_cubic_features(data):
    rows,columns=data.shape;
    train_size= (int)(0.8*(rows));
    test_size=rows-train_size;
    for i in range(0,columns-1):
            buf="x%d_cube"%(i+1);
            data[buf]=data.iloc[:,i].copy();
            data[buf]=data[buf]**3;
    
    featureScaling(data);
    train_data=data[0:train_size];
    test_data=data[train_size:];
    train_yData=train_data['price'].copy();
    train_yData=train_yData.values.reshape((train_size,1));
    test_yData=test_data['price'].copy();
    test_yData=test_yData.values.reshape((test_size,1));
    del train_data['price'];
    del test_data['price'];
    train_xData=train_data.iloc[:,0:].copy();
    train_xData['x0']=1;
    print(train_yData.shape);
    test_xData=test_data.iloc[:,0:].copy();
    test_xData['x0']=1;
   
    return train_xData,train_yData,test_xData,test_yData,2*columns-1;


def featureScaling(data):
        for col in data.columns:
            mean=(data[col].mean());
            den=((data[col].max()-data[col].min())*1.0);
            data[col]=(data[col]-mean)/data[col].std();
            
def train_model(x_data,theta,y_data,alpha):
    for i in range(0,2000):
        #preverror=errorfunc(x_data,theta,y_data);
        temp=np.dot(x_data,theta)-y_data;
        temp=np.dot(np.transpose(x_data),temp);
        rows,cols=temp.shape;
        # print(i);
        oldtheta0=theta[rows-1];
        theta= (theta)-(((1.0*alpha)/len(x_data))*temp);
    return theta;
    
def get_results(columns,train_xData,train_yData,test_xData,test_yData,std1):
    alpha_val=np.arange(0.01,0.2,0.01);
    real_perferror_val=[];
    
    for i in alpha_val:
        theta=np.random.rand(columns,1);
        theta=train_model(train_xData,theta,train_yData,i);
        print(theta);
        perf_error=perf_measure(test_xData,theta,test_yData);
        real_perferror=(perf_error*std1);
        real_perferror_val.append(real_perferror);
        print("Performance Error  for alpha = "+ str(i)+" : "+str(real_perferror));
    return alpha_val,real_perferror_val;

def main():
    data=pd.read_csv('kc_house_data.csv');

    data1=data.copy();
    data2=data.copy();
    data3=data.copy();
    std1=data['price'].std();
    plt.xlabel("alpha values");
    plt.ylabel("Performance error");
    plt.title("Performance error vs alpha");
    
    train_xData,train_yData,test_xData,test_yData,columns=get_linear_features(data1);
    alpha_val,real_perferror_val=get_results(columns,train_xData,train_yData,test_xData,test_yData,std1);
    plt.plot(alpha_val,real_perferror_val,marker='x',label="linear");

    train_xData,train_yData,test_xData,test_yData,columns=get_quadratic_features(data2);
    alpha_val,real_perferror_val=get_results(columns,train_xData,train_yData,test_xData,test_yData,std1);
    plt.plot(alpha_val,real_perferror_val,marker='+',label="Quadratic");   

    train_xData,train_yData,test_xData,test_yData,columns=get_cubic_features(data3);
    alpha_val,real_perferror_val=get_results(columns,train_xData,train_yData,test_xData,test_yData,std1);
    plt.plot(alpha_val,real_perferror_val,marker='*',label="cubic");    
    

    plt.legend(loc="best");
    plt.savefig("different_features_combo.jpg");
    
if __name__=="__main__":
    main();
    
    
