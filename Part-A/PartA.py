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

def featureScaling(data):
        for col in data.columns:
            mean=(data[col].mean());
            den=((data[col].max()-data[col].min())*1.0);
            data[col]=(data[col]-mean)/data[col].std();
            
def train_model(x_data,theta,y_data,alpha,lambda1):
    for i in range(0,1000):
        #preverror=errorfunc(x_data,theta,y_data);
        temp=np.dot(x_data,theta)-y_data;
        temp=np.dot(np.transpose(x_data),temp);
        rows,cols=temp.shape;
        # print(i);
        oldtheta0=theta[rows-1];
        theta= (theta*(1- ((alpha*lambda1)/len(x_data))))-(((1.0*alpha)/len(x_data))*temp);
        theta[rows-1]=theta[rows-1]+(oldtheta0*(((alpha*lambda1)/len(x_data))));
        # print(theta);
        #currerror=errorfunc(x_data,theta,y_data);
    return theta;
    
def main():
    data=pd.read_csv('kc_house_data.csv');
    std1=data['price'].std();
    mean1=data['price'].mean();
    rows,columns=data.shape;
    train_size= (int)(0.8*(rows));
    test_size=rows-train_size;
    featureScaling(data);
    train_data=data[0:train_size];
    test_data=data[train_size:];
    
    # theta=np.ones((columns,1));
    train_xData=train_data.iloc[:,0:columns-1].copy();
    train_xData['x0']=1;
    train_yData=train_data.iloc[:,columns-1:];
    test_xData=test_data.iloc[:,0:columns-1].copy();
    test_xData['x0']=1;
    test_yData=test_data.iloc[:,columns-1:];
    # print(train_xData);
    lambda_val=np.arange(0.0,5.0,0.5);
    real_perferror_val=[];

    for i in lambda_val:
        theta=np.random.rand(columns,1);
        theta=train_model(train_xData,theta,train_yData,0.05,i);
        print(theta);
        perf_error=perf_measure(test_xData,theta,test_yData);
        real_perferror=(perf_error*std1);
        real_perferror_val.append(real_perferror);
        print("Performance Error  for lamda = "+ str(i)+" : "+str(real_perferror));
        

    plt.xlabel("lamda values");
    plt.ylabel("Performance error");
    plt.title("Performance error vs lambda");
    plt.plot(lambda_val,real_perferror_val,marker='+');
    plt.xlim(0,5.0,0.5);
    plt.savefig("performance_error_plot.jpg");
    
if __name__=="__main__":
    main();
    
    
