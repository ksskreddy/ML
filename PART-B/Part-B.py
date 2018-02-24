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
            
def train_model1(x_data,theta,y_data,alpha,iter):
    for i in range(0,iter):
        temp=np.dot(x_data,theta)-y_data;
        temp=np.dot(np.transpose(x_data),temp);
        theta= (theta)-(((1.0*alpha)/len(x_data))*temp);
    return theta;
    

def train_model2(x_data,theta,y_data,iter):
    for i in range(0,iter):
        temp = np.dot(x_data,theta)-y_data;
        temp = np.dot(np.transpose(x_data),temp);
        Hessian = np.dot(np.transpose(x_data),x_data);
        Hessian = np.linalg.inv(Hessian);
        theta = theta-np.dot(Hessian,temp);
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
    theta=np.random.rand(columns,1);
    train_xData=train_data.iloc[:,0:columns-1].copy();
    train_xData['x0']=1;
    train_yData=train_data.iloc[:,columns-1:];
    test_xData=test_data.iloc[:,0:columns-1].copy();
    test_xData['x0']=1;
    test_yData=test_data.iloc[:,columns-1:];

    iter_val=np.arange(0,30,1);
    real_perferror_val1=[];
    real_perferror_val2=[];
    theta_temp=theta.copy();
    for i in iter_val:
        theta2=train_model2(train_xData,theta_temp,train_yData,i);
        perf_error=perf_measure(test_xData,theta2,test_yData);
        real_perferror1=(perf_error*std1);
        real_perferror_val2.append(real_perferror1);
        theta1=train_model1(train_xData,theta,train_yData,0.05,i);
        perf_error=perf_measure(test_xData,theta1,test_yData);
        real_perferror2=(perf_error*std1);
        real_perferror_val1.append(real_perferror2);

    print("Perf error in GD : "+str(real_perferror2));
    print("Perf error in IRLS : "+str(real_perferror1));

    plt.xlabel("No of iterations");
    plt.ylabel("Performance error in 1000's");
    plt.title("Performance error vs Iterations");
    plt.plot(iter_val,real_perferror_val1,marker='+',label="Grad Descent");
    plt.plot(iter_val,real_perferror_val2,marker='*',label="IRLS");
    plt.legend(loc="best");
    plt.savefig("performance_error_plot.jpg");
  
if __name__=="__main__":
    main();
    
    

