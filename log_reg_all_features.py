import pandas as pd
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

###################
# data processing #
###################
# read the data from the file and transform it to the same
# data as in the jupyter notebook
data_frame = pd.read_csv("train.csv", encoding="utf-8")
df = data_frame.drop(['id'],axis=1)

y = df.label.values
X = df.drop(['label'],axis=1).values

# split data
y_split = np.array_split(y,6)
y_train = np.concatenate((y_split[3],y_split[5],y_split[2]))
y_val = np.concatenate((y_split[0],y_split[4]))
y_test = y_split[1]
print("y_train shape: \t", y_train.shape)
print("y_val shape: \t", y_val.shape)
print("y_test shape: \t", y_test.shape)


X_split = np.array_split(X,6)
X_train = np.concatenate((X_split[3],X_split[5],X_split[2]))
X_val = np.concatenate((X_split[0],X_split[4]))
X_test = X_split[1]
print("X_train shape: \t", X_train.shape)
print("X_val shape: \t", X_val.shape)
print("X_test shape: \t", X_test.shape)

# save the mean and the standard deviation of every feature
feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)

###################
# feature scaling #
###################
X_train_scaled = (X_train - feature_mean)/(feature_std)
X_train = X_train_scaled
X_val_scaled = (X_val - feature_mean)/(feature_std)
X_val = X_val_scaled
X_test_scaled = (X_test - feature_mean)/(feature_std)
X_test = X_test_scaled


################################
# START OF LOGISTIC REGRESSION #
################################
# logistic function #
#####################
def logistic_function(x):
    return 1/(1 + np.exp(-x))


#######################
# logistic hypothesis #
#######################
def logistic_hypothesis(theta):
    return lambda X: logistic_function((np.concatenate((np.ones((len(X),1)),X),axis=1)).dot(theta))
    




#########################
# stable cross entropy  #
#########################
def batched_cross_entropy(z, y):
	res = np.zeros(len(z))
	mu = np.max([np.zeros(len(z)),-z],axis=0)
	r1 = y * (mu + np.log(np.exp(-mu)+np.exp(-z-mu)))  
	mu = np.max([np.zeros(len(z)), z],axis=0)
	r2 = (1-y) * (mu + np.log(np.exp(-mu)+np.exp(z-mu)))
	res = r1 + r2
	return res



def mean_batch_cross_entropy_costs(X,y,theta,b_cross_entropy,lambda_reg):
	m = len(X)
	z = (np.concatenate((np.ones((len(X),1)),X),axis=1)).dot(theta)
	return lambda theta: 1./(m) * (np.sum(b_cross_entropy(z,y)) + (lambda_reg/2)*np.sum(theta**2))




###########################
# update gradient descent #
###########################
def compute_new_theta(X, y, theta, learning_rate, hypothesis, lambda_reg):
    m = len(X)
    x_ = np.concatenate((np.ones([m,1]), X), axis=1)
    theta =  theta - learning_rate * (1./(m) * (hypothesis(theta)(X)-y).dot(x_) + (lambda_reg/m) * theta)
    return theta


####################
# gradient descent #
####################
def gradient_descent(X, y, theta, learning_rate, num_iters, lambda_reg):
	history_cost = np.zeros(num_iters)
	history_theta = np.zeros([num_iters,len(theta)])
	m = len(X)
	
	for i in range(num_iters):
		costs = mean_batch_cross_entropy_costs(X,y,theta,batched_cross_entropy,lambda_reg)
		history_theta[i] = theta
		history_cost[i] = costs(theta)
		theta = compute_new_theta(X,y,theta,learning_rate,logistic_hypothesis,lambda_reg)
	return history_cost, history_theta

########
# plot #
########
def plot_progress(costs):
	fig = plt.figure(figsize=(10,7))
	ax = fig.add_subplot(111)
	ax.plot(np.array(range(len(costs))), costs)
	ax.set_xlabel("Iterations")
	ax.set_ylabel("costs")
	ax.set_title("Evaluation")
	plt.show()	



#############
# accurcacy #
#############
def accuracy(final_theta,X_val,y_val,threshold):
    correct = 0
    length = len(X_val)
    prediction = (logistic_hypothesis(final_theta)(X_val) > threshold)
    correct = prediction == y_val
    my_accuracy = (np.sum(correct) / length)*100
    print ('LR Accuracy %: \t', my_accuracy)
    


#######################
# out of sample error #
#######################
# use the validation data
def out_of_sample_error(y_preds, y):
	return ((y_preds - y) ** 2).mean()



####################
# confusion matrix #
####################
def print_confusion_matrix(final_theta, logistic_hypothesis, X_val_scaled, y_val, threshold):
	print("Confusion matrix for threshold = \t" , threshold)
	y_preds = logistic_hypothesis(final_theta)(X_val_scaled)
	# change y_preds from floats to 1 if >= threshold else to 0
	y_preds_final = np.zeros(len(y_preds))
	for i in range(len(y_preds)):
		if y_preds[i] >= threshold:
			y_preds_final[i] = 1
		else:
			y_preds_final[i] = 0
	cm = confusion_matrix(y_val,y_preds_final)
	print(cm)
	print("-------------------------------------------------")
	return cm, y_preds_final, y_preds



#############################
# accuracy precision recall #
#############################
def acc_pre_rec(cm):
	# true positive
	tp = cm[0][0]
	# true negative 
	tn = cm[1][1]
	# false negative
	fn = cm[1][0]
	# false postive
	fp = cm[0][1]
	accuracy = (tp + tn)/(tp + fp + tn + fn)
	precision = (tp)/(tp + fp)
	recall = (tp)/(tp + fn)
	print("True positive \t", tp)
	print("True negative \t", tn)
	print("False negative \t", fn)
	print("False postive \t", fp)
	print("Accuracy % \t", accuracy*100)
	print("Precision % \t", precision*100)
	print("Recall (Sensitivity) % \t", recall*100)



#######
# ROC #
#######
# use sklearn to plot the ROC curve
# true positive rate on the x-axis
# false positive rate on the y-axis
def plot_my_roc(y_val, y_preds):
	fpr, tpr, threshold = metrics.roc_curve(y_val,y_preds)
	roc_auc = metrics.auc(fpr,tpr)
	plt.figure(figsize=(10,7))
	plt.plot(fpr, tpr, label="ROC curve (area = %0.4f) " % roc_auc)
	plt.plot([0,1],[0,1], "r--")
	plt.xlim([-0.01,1.01])
	plt.ylim([-0.01, 1.01])
	plt.xlabel("False Positive Rate = Fallout")
	plt.ylabel("True Positive Rate = Sensitivity = Recall")
	plt.title("Receiver operating characteristics")
	plt.legend(loc="lower right")
	plt.show()




###########################################################
# main method
# thetas = initial thetas
# alpha = learning rate
# num_iters = number of iterations
# lambda_reg = regularisation
# threshold
# cm_print = True -> prints confusion matrix for different thresholds, False -> otherwise
def test_for_mult_values(theta,alpha,num_iters,lambda_reg,treshold,cm_print):
	print("Learning Rate:\t", alpha)
	print("Number of iterations:\t", num_iters)
	print("Lambda Regression:\t", lambda_reg)
	print("Threshold:\t", treshold)
	print("Thetas:\t", theta)
	history_cost, history_theta = gradient_descent(X_train, y_train, theta, alpha, num_iters, lambda_reg)
	plot_progress(history_cost)
	print("Costs before the training:\t", history_cost[0])
	print("Costs after the training:\t", history_cost[-1])
	final_theta = history_theta[-1]
	print("Final Thetas:\t", final_theta)
	oos_error = out_of_sample_error(logistic_hypothesis(theta)(X_val),y_val)
	error = 1/np.sqrt(len(X_val))
	print("Out of sample Error\t", oos_error)
	print("Validation Error: " + str(oos_error) + " +- " + str(error))
	print("Accuracy with test data")
	accuracy(final_theta,X_test,y_test,treshold)
	print("-------------------------------------------------")
	cm, y_preds_final, y_preds = print_confusion_matrix(final_theta, logistic_hypothesis, X_test, y_test,treshold)
	acc_pre_rec(cm)
	plot_my_roc(y_test, y_preds)
	# test cm for different thresholds
	if cm_print:
		for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
			cm_, y_preds_final_, y_preds_ = print_confusion_matrix(final_theta, logistic_hypothesis, X_val_scaled, y_val,i)
	return final_theta
#############################################################

########### TESTING WITH DIFFERENT PARAMETERS ###############
thetas_1 = np.array([0, 2, -1.5, 4, 0, 0, 1, -3.5, 2.2, 4.3, 
			2, 0.6, 0.4, -3, 3, 3, 6, 3, 0, 2, 
			-1, -3, 0.5, 3, 1, 3, 5,0.1,2,4,0])  


thetas_2 = np.array([1, 4, -2.5, 4, 1, 5, 4, -1.5, 1.2, 6.5, 
			7, 0.6, 0.9, -3, 3, 4, 2, 6, -3, 4, 
			-3, -1, 0.5, 2, 1, 2, 7,1,1,0,1])  

thetas_3 = np.zeros(31)

#theta = np.random.randn(31)
#test_for_mult_values(theta,0.01,10000,0.1)
#theta = np.random.randn(31)
#test_for_mult_values(theta,0.1,10000,0.001)
#theta = np.random.randn(31)
#test_for_mult_values(theta,0.1,1000,0.0000001)

print("------------------------------------------------------------------")
# best combination between low out of sample error and high accuracy
print("Best combination between low out of sample error and high accuracy")
print("------------------------------------------------------------------")
#final_theta_1 = test_for_mult_values(thetas_1,0.1,5000,0.01,0.5,False)

test_for_mult_values(thetas_1,0.1,10000,0.5,0.5,False)
#test_for_mult_values(thetas_2,0.1,15000,0.001,0.5,False)

#test_for_mult_values(thetas_2,0.1,15000,0.01,0.5,False)


# better accuracy but higher out of sample error
#final_theta_2 = test_for_mult_values(thetas_3,0.01,5000,0.001,0.5,False)


################################################################
