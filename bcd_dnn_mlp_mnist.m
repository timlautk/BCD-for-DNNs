%% Block Proximal Point (BPP) Algorithm for Training DNNs (3-layer MLP)
clear all
close all
clc

addpath Algorithms Tools

rng('default');
seed = 20;
rng(seed);

% read in MNIST dataset into Matlab format if not exist
if exist('mnist.mat', 'file')
    mnist = load('mnist.mat');
else
    disp('reading in MNIST dataset into Matlab format')
    addpath mnist-matlab
    convertMNIST
    mnist = load('mnist.mat');
end

% train data and labels
[x_d1,x_d2,x_d3] = size(mnist.training.images);
x_train = reshape(mnist.training.images,x_d1*x_d2,x_d3); % train data
% x_train = gpuArray(x_train);
y_train = mnist.training.labels; % labels
% y_train = gpuArray(y_train);

%% Extract Classes 
num_classes = 10; % choose the first num_class classes in the MNIST dataset for training
X = [y_train';x_train];
[~,col] = find(X(1,:) < num_classes);
X = X(:,col);
[~,N] = size(X);
X = X(:,randperm(N)); % shuffle the dataset
x_train = X(2:end,:);
y_train = X(1,:)';
clear X

y_one_hot = ind2vec((y_train'+1));
[K,N] = size(y_one_hot);
[d,~] = size(x_train);

%% Test data
% read in test data and labels
[x_test_d1,x_test_d2,x_test_d3] = size(mnist.test.images);
x_test = reshape(mnist.test.images,x_test_d1*x_test_d2,x_test_d3); % test data
y_test = mnist.test.labels; % labels

X_test = [y_test';x_test];
[~, col_test] = find(X_test(1,:) < num_classes);
X_test = X_test(:,col_test);
[~,N_test] = size(X_test);
X_test = X_test(:,randperm(N_test,N_test));
x_test = X_test(2:end,:);
y_test = X_test(1,:)';
clear X_test

y_test_one_hot = ind2vec((y_test'+1));
[~,N_test] = size(y_test_one_hot);

%% Visual data samples
% figure;
% for i = 1:100
%     subplot(10,10,i)
%     img{i} = reshape(x_train(:,i),x_d1,x_d2);
%     imshow(img{i})
% end
% 
% close all

%% Main Algorithm 1 (Proximal Point)
% Initialization of parameters 
d0 = d; d1 = 2048; d2 = 2048; d3 = 2048; d4 = K; % Layers: input + 3 hidden + output
W1 = 0.01*randn(d1,d0); b1 = 0.1*ones(d1,1); 
% W1 = zeros(d1,d0); b1 = zeros(d1,1);

W2 = 0.01*randn(d2,d1); b2 = 0.1*ones(d2,1); 
% W2 = zeros(d2,d1); b2 = zeros(d2,1);

W3 = 0.01*randn(d3,d2); b3 = 0.1*ones(d3,1); 
% W3 = zeros(d3,d2); b3 = zeros(d3,1); 

% V = 0.01*sprand(d4,d3,0.1); c = zeros(d4,1); 
V = 0.01*randn(d4,d3); c = 0.1*ones(d4,1); 
% V = zeros(d4,d3); c = zeros(d4,1);


indicator = 1; % 0 = sign; 1 = ReLU; 2 = tanh; 3 = sigmoid

% a1 = zeros(d1,N); a2 = zeros(d2,N); a3 = zeros(d3,N);
switch indicator
    case 0 % sign (binary)
        a1 = sign(W1*x_train+b1); a2 = sign(W2*a1+b2); a3 = sign(W3*a2+b3);  
	case 1 % ReLU
        a1 = max(0,W1*x_train+b1); a2 = max(0,W2*a1+b2); a3 = max(0,W3*a2+b3);  
	case 2 % tanh
        a1 = tanh_proj(W1*x_train+b1); a2 = tanh_proj(W2*a1+b2); a3 = tanh_proj(W3*a2+b3);
	case 3 % sigmoid
	a1 = sigmoid_proj(W1*x_train+b1); a2 = sigmoid_proj(W2*a1+b2); a3 = sigmoid_proj(W3*a2+b3);
end
a1star = zeros(d1,N); a2star = zeros(d2,N); a3star = zeros(d3,N); 
u1 = zeros(d1,N); u2 = zeros(d2,N); u3 = zeros(d3,N); 

lambda = 0;
gamma = 0.1; gamma1 = gamma; gamma2 = gamma; gamma3 = gamma; gamma4 = 0.1;
% alpha1 = 10; 
alpha1 = 1e-3; 
alpha = 1e-2;
alpha2 = alpha; alpha3 = alpha; alpha4 = alpha; 
alpha5 = alpha; alpha6 = alpha; alpha7 = alpha; 
alpha8 = alpha; alpha9 = alpha; alpha10 = alpha; 

beta = 0.9;
beta1 = beta; beta2 = beta; beta3 = beta; beta4 = beta; 
beta5 = beta; beta6 = beta; beta7 = beta; 
beta8 = beta; beta9 = beta; beta10 = beta; 

t = 0.1;

s = 10; % number of mini-batches
% niter = input('Number of iterations: ');
niter = 30;
loss1 = zeros(niter,1);
loss2 = zeros(niter,1);
accuracy_train = zeros(niter,1);
accuracy_test = zeros(niter,1);
time1 = zeros(niter,1);

% Iterations
for k = 1:niter
    tic
    
    % Forward propagation
%     switch indicator
%     case 0 % sign (binary)
%         a1 = sign(W1*x_train+b1); a2 = sign(W2*a1+b2); a3 = sign(W3*a2+b3);  
% 	case 1 % ReLU
%         a1 = max(0,W1*x_train+b1); a2 = max(0,W2*a1+b2); a3 = max(0,W3*a2+b3);  
% 	case 2 % tanh
%         a1 = tanh_proj(W1*x_train+b1); a2 = tanh_proj(W2*a1+b2); a3 = tanh_proj(W3*a2+b3);
% 	case 3 % sigmoid
%         a1 = sigmoid_proj(W1*x_train+b1); a2 = sigmoid_proj(W2*a1+b2); a3 = sigmoid_proj(W3*a2+b3);
%     end

    
    % update stepsize
%     alpha1 = (1+k)/alpha1; alpha2 = (1+k)/alpha2; alpha3 = (1+k)/alpha3; alpha4 = (1+k)/alpha4; alpha5 = (1+k)/alpha5; 
%     alpha6 = (1+k)/alpha6; alpha7 = (1+k)/alpha7; alpha8 = (1+k)/alpha8; alpha9 = (1+k)/alpha9; alpha10 = (1+k)/alpha10; 

    % update V and c (output/loss layer)
    [Vstar,cstar] = updateWb(y_one_hot,a3,V,c,alpha1,gamma4,lambda);
    % adaptive momentum and update
	[V,c,beta1] = AdaptiveVb_4(lambda,gamma4,y_one_hot,a3,V,Vstar,c,cstar,beta1,t);
%     [V,c,beta1] = AdaptiveVb_3(lambda,y_one_hot,a3,V,Vstar,c,cstar,beta1,t);
    
    % update a3
    a3star = updatea_2(a2,a3,y_one_hot,W3,V,b3,c,u3,zeros(d4,1),alpha2,gamma3,gamma4,indicator);
    % adaptive momentum and update
    [a3,beta2] = Adaptivea1_3(gamma3,gamma4,y_one_hot,a2,a3,a3star,W3,V,b3+u3,c,beta2,t);
    
    % update u3
    u3 = a3-W3*a2-b3;

    % update W3 and b3 (3rd layer)
    [W3star,b3star] = updateWb_2(a3,a2,u3,W3,b3,alpha3,gamma3,lambda);
    % adaptive momentum and update
    [W3,b3,beta3] = AdaptiveWb1_3(a2,a3,W3,W3star,b3,b3star,beta3,t);
    
    
    % update a2
    a2star = updatea_2(a1,a2,a3,W2,W3,b2,b3,u2,u3,alpha4,gamma2,gamma3,indicator);
    % adaptive momentum and update
    [a2,beta4] = Adaptivea1_3(gamma2,gamma3,a3,a1,a2,a2star,W2,W3,b2+u2,b3,beta4,t);
    
    % update u2
    u2 = a2-W2*a1-b2;

    
    % update W2 and b2 (2nd layer)
    [W2star,b2star] = updateWb_2(a2,a1,u2,W2,b2,alpha5,gamma2,lambda);
    % adaptive momentum and update
    [W2,b2,beta5] = AdaptiveWb1_3(a1,a2,W2,W2star,b2,b2star,beta5,t);
    
    
    % update a1
    a1star = updatea_2(x_train,a1,a2,W1,W2,b1,b2,u1,u2,alpha6,gamma1,gamma2,indicator);
    % adaptive momentum and update
    [a1,beta6] = Adaptivea1_3(gamma1,gamma2,a2,x_train,a1,a1star,W1,W2,b1+u1,b2,beta6,t);
    
    % update u1
    u1 = a1-W1*x_train-b1;
    
    % update W1 and b1 (1st layer)
    [W1star,b1star] = updateWb_2(a1,x_train,u1,W1,b1,alpha7,gamma1,lambda);
    % adaptive momentum and update
    [W1,b1,beta7] = AdaptiveWb1_3(x_train,a1,W1,W1star,b1,b1star,beta7,t);
     
    % Training accuracy
    switch indicator
    case 0 % sign
    	a1_train = sign(W1*x_train+b1);
        a2_train = sign(W2*a1_train+b2);
        a3_train = sign(W3*a2_train+b3);
    case 1 % ReLU
        a1_train = max(0,W1*x_train+b1);
        a2_train = max(0,W2*a1_train+b2);
        a3_train = max(0,W3*a2_train+b3);
%         a3_train = max(0,W3*a2_train+b3);
    case 2 % tanh
        a1_train = tanh_proj(W1*x_train+b1);
        a2_train = tanh_proj(W2*a1_train+b2);
        a3_train = tanh_proj(W3*a2_train+b3);
    case 3 % sigmoid
        a1_train = sigmoid_proj(W1*x_train+b1);
        a2_train = sigmoid_proj(W2*a1_train+b2);
        a3_train = sigmoid_proj(W3*a2_train+b3);
    end
    [~,pred] = max(V*a3_train+c,[],1);
    
    % Test accuracy
    switch indicator
        case 0 % sign
        a1_test = sign(W1*x_test+b1);
        a2_test = sign(W2*a1_test+b2);
        a3_test = sign(W3*a2_test+b3);
        case 1 % ReLU
        a1_test = max(0,W1*x_test+b1); 
        a2_test = max(0,W2*a1_test+b2); 
        a3_test = max(0,W3*a2_test+b3);
%         a4_test = max(0,W4*a3_test+b4);
        case 2 % tanh
        a1_test = tanh_proj(W1*x_test+b1); 
        a2_test = tanh_proj(W2*a1_test+b2); 
        a3_test = tanh_proj(W3*a2_test+b3);
%         a4_test = tanh_proj(W4*a3_test+b4);
        case 3 % sigmoid
        a1_test = sigmoid_proj(W1*x_test+b1); 
        a2_test = sigmoid_proj(W2*a1_test+b2); 
        a3_test = sigmoid_proj(W3*a2_test+b3);
%         a4_test = sigmoid_proj(W4*a3_test+b4);
    end
    [~,pred_test] = max(V*a3_test+c,[],1);
    
    time1(k) = toc;
    loss1(k) = gamma4/2*norm(V*a3+c-y_one_hot,'fro')^2;
%     loss1(k) = cross_entropy(y_one_hot,a1,V,c)+lambda*norm(V,'fro')^2;
    loss2(k) = loss1(k)+gamma1/2*norm(W1*x_train+b1-a1+u1,'fro')^2+gamma2/2*norm(W2*a1+b2-a2+u2,'fro')^2+gamma3/2*norm(W3*a2+b3-a3+u3,'fro')^2+lambda*(norm(W1,'fro')^2+norm(W2,'fro')^2+norm(W3,'fro')^2+norm(V,'fro')^2);
%     loss1(k) = cross_entropy(y_one_hot,a1,W2,b2)+gamma1/2*norm(W1*x_train+b1-a1,'fro')^2;
    accuracy_train(k) = sum(pred'-1 == y_train)/N;
    accuracy_test(k) = sum(pred_test'-1 == y_test)/N_test;
    fprintf('epoch: %d, squared loss: %f, total loss: %f, training accuracy: %f, validation accuracy: %f, time: %f\n',k,loss1(k),loss2(k),accuracy_train(k),accuracy_test(k),time1(k));
  
end


fprintf('squared error: %f\n',loss1(k))
fprintf('sum of inter-layer loss: %f\n',loss2(k)-loss1(k))
%disp(full(cross_entropy(y_one_hot,a2,V,c)))


%% Plots
figure;
graph1 = semilogy(1:niter,loss1,1:niter,loss2);
set(graph1,'LineWidth',1.5);
legend('Squared loss','Total loss');
ylabel('Loss')
xlabel('Epochs')
title('Three-layer MLP')

figure;
graph2 = semilogy(1:niter,accuracy_train,1:niter,accuracy_test);
set(graph2,'LineWidth',1.5);
legend('Training accuracy','Validation accuracy','Location','southeast');
ylabel('Accuracy')
xlabel('Epochs')
title('Three-layer MLP')

%% Training error
switch indicator
    case 1 % ReLU
        a1_train = max(0,W1*x_train+b1);
        a2_train = max(0,W2*a1_train+b2);
        a3_train = max(0,W3*a2_train+b3);
    case 2 % tanh
        a1_train = tanh_proj(W1*x_train+b1);
        a2_train = tanh_proj(W2*a1_train+b2);
        a3_train = tanh_proj(W3*a2_train+b3);
    case 3 % sigmoid
        a1_train = sigmoid_proj(W1*x_train+b1);
        a2_train = sigmoid_proj(W2*a1_train+b2);
        a3_train = sigmoid_proj(W3*a2_train+b3);
end

[~,pred] = max(V*a3_train+c,[],1);
pred_one_hot = ind2vec(pred);
accuracy_final = sum(pred'-1 == y_train)/N;
fprintf('Training accuracy using output layer: %f\n',accuracy_final);
% error = full(norm(pred_one_hot-y_one_hot,'fro')^2/(2*N));
% fprintf('Training MSE using output layer: %f\n',error);

%% Test errors
switch indicator
    case 1 % ReLU
        a1_test = max(0,W1*x_test+b1); 
        a2_test = max(0,W2*a1_test+b2); 
        a3_test = max(0,W3*a2_test+b3); 
    case 2 % tanh
        a1_test = tanh_proj(W1*x_test+b1); 
        a2_test = tanh_proj(W2*a1_test+b2); 
        a3_test = tanh_proj(W3*a2_test+b3);
    case 3 % sigmoid
        a1_test = sigmoid_proj(W1*x_test+b1); 
        a2_test = sigmoid_proj(W2*a1_test+b2); 
        a3_test = sigmoid_proj(W3*a2_test+b3);
end


[~,pred_test] = max(V*a3_test+c,[],1);
pred_test_one_hot = ind2vec(pred_test);
accuracy_test_final = sum(pred_test'-1 == y_test)/N_test;
fprintf('Test accuracy using output layer: %f\n',accuracy_test_final);
% error_test = full(norm(pred_test_one_hot-y_test_one_hot,'fro')^2/(2*N_test));
% fprintf('Test MSE using output layer: %f\n',error_test);

%% Linear SVM for train errors
% rng(seed); % For reproducibility
% SVMModel = fitcecoc(a3_train,y_train,'ObservationsIn','columns');
% L = resubLoss(SVMModel,'LossFun','classiferror');
% % fprintf('Training error classified with SVM: %f\n',L);
% fprintf('Training accuracy classified with SVM: %f\n',1-L);

%% SVM test error
% predictedLabels = predict(SVMModel,a3_test,'ObservationsIn','columns');
% accuracy = sum(predictedLabels==y_test)/numel(predictedLabels);
% fprintf('Test accuracy classified with SVM: %f\n',accuracy);


%% Toolbox training

% layers = [imageInputLayer([28 28 1]);
%           fullyConnectedLayer(100);
%           reluLayer();
%           fullyConnectedLayer(K);
%           softmaxLayer();
%           classificationLayer()];
%       
%       
% options = trainingOptions('sgdm','ExecutionEnvironment','gpu','MaxEpochs',50,'InitialLearnRate',0.01);
% 
% rng(20)
% net = trainNetwork(reshape(x_train,28,28,1,N),categorical(y_train),layers,options);
% 
% % Test accuracy
% YTest = classify(net,reshape(x_test,28,28,1,N_test));
% TTest = categorical(y_test);
% accuracy1 = sum(YTest == TTest)/numel(TTest);  
% fprintf('Test accuracy with backprop: %f\n',accuracy1);

%% Feature extraction + SVM + Test accuracy
% trainFeatures = activations(net,reshape(x_train,28,28,1,N),3);
% svm = fitcecoc(trainFeatures,categorical(y_train));
% L2 = resubLoss(svm,'LossFun','classiferror');
% fprintf('Training error using backprop classified with SVM: %f\n',L2);
% fprintf('Training accuracy using backprop classified with SVM: %f\n',1-L2);
% 
% testFeatures = activations(net,reshape(x_test,28,28,1,N_test),3);
% testPredictions = predict(svm,testFeatures);
% accuracy2 = sum(categorical(y_test) == testPredictions)/numel(categorical(y_test));
% fprintf('Test accuracy using backprop classified with SVM: %f\n',accuracy2);

