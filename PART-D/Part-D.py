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




def featureScaling(data):
        for col in data.columns:
            mean=(data[col].mean());
            den=((data[col].max()-data[col].min())*1.0);
            data[col]=(data[col]-mean)/data[col].std();
            
def train_model(x_data,theta,y_data,alpha):
    for i in range(0,1000):

        temp=np.dot(x_data,theta)-y_data;
        temp=np.dot(np.transpose(x_data),temp);
        rows,cols=temp.shape;
        # print(i);
        oldtheta0=theta[rows-1];
        theta= (theta)-(((1.0*alpha)/len(x_data))*temp);
    return theta;


def train_model2(x_data,theta,y_data,alpha):
    for i in range(0,1000):
        #preverror=errorfunc(x_data,theta,y_data);
        temp=np.dot(x_data,theta)-y_data;
        temp=abs(temp)/temp;
        temp=np.dot(np.transpose(x_data),temp);
        rows,cols=temp.shape;
        # print(i);
        oldtheta0=theta[rows-1];
        theta= (theta)-(((1.0*alpha)/len(x_data))*temp);
    return theta;


def train_model3(x_data,theta,y_data,alpha):
    for i in range(0,1000):
        temp=(np.dot(x_data,theta)-y_data).copy();
        temp=temp**2;
        temp=np.dot(np.transpose(x_data),temp);
        theta= (theta)-(((1.0*alpha)/len(x_data))*temp);
        # print(theta);
    return theta;


def get_results1(columns,train_xData,train_yData,test_xData,test_yData,std1):
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


def get_results2(columns,train_xData,train_yData,test_xData,test_yData,std1):
    alpha_val=np.arange(0.01,0.2,0.02);
    real_perferror_val=[];
    
    for i in alpha_val:
        theta=np.random.rand(columns,1);
        theta=train_model2(train_xData,theta,train_yData,i);
        print(theta);
        perf_error=perf_measure(test_xData,theta,test_yData);
        real_perferror=(perf_error*std1);
        real_perferror_val.append(real_perferror);
        print("Performance Error  for alpha = "+ str(i)+" : "+str(real_perferror));
    return alpha_val,real_perferror_val;




def get_results3(columns,train_xData,train_yData,test_xData,test_yData,std1):
    alpha_val=np.arange(3e-5,4e-5,1e-6);
    real_perferror_val=[];
    
    for i in alpha_val:
        theta=np.random.rand(columns,1);
        theta=train_model3(train_xData,theta,train_yData,i);
        print(theta);
        perf_error=perf_measure(test_xData,theta,test_yData);
        real_perferror=(perf_error*std1);
        real_perferror_val.append(real_perferror);
        print("Performance Error  for alpha = "+ str(i)+" : "+str(real_perferror));
    return alpha_val,real_perferror_val;


def main():
    data=pd.read_csv('kc_house_data.csv');
    data1=data.copy();
    std1=data['price'].std();
    plt.xlabel("alpha values");
    plt.ylabel("Performance error");
    plt.title("Performance error vs alpha using different cost functions");
    
    plt.figure(1)
    train_xData,train_yData,test_xData,test_yData,columns=get_linear_features(data);
    alpha_val,real_perferror_val=get_results1(columns,train_xData,train_yData,test_xData,test_yData,std1);
    plt.plot(alpha_val,real_perferror_val,marker='x',label="mean_squared");
    plt.legend(loc="best");
    plt.savefig("mean_squared.jpg");
    plt.figure(2);
    alpha_val,real_perferror_val=get_results2(columns,train_xData,train_yData,test_xData,test_yData,std1);
    plt.plot(alpha_val,real_perferror_val,marker='x',label="mean_absolute");
    plt.legend(loc="best");
    plt.savefig("mean_absolute.jpg");
    
    plt.figure(3);
    alpha_val,real_perferror_val=get_results3(columns,train_xData,train_yData,test_xData,test_yData,std1);
    plt.plot(alpha_val,real_perferror_val,marker='x',label="mean cubic");
    plt.legend(loc="best");
    plt.savefig("cubic_cost_func.jpg");


    
    
if __name__=="__main__":
    main();
    
    
