% Feature Fix Depth Change 'Gini')
clear all; clc; close all

%Gini
x_axis_depth = [3, 10, 30, 40, 50];
val_acc_depth = [0.44729270206643995, 0.8543029034789432, 0.8535181794402302, 0.8579649489929375 , 0.8603191211090766];
train_acc_depth = [0.4612236738698049, 0.9637075509675752, 1, 1, 1];

x_axis_feature = [2, 4, 6, 8, 10, 12];
val_acc_feature = [0.7206382422181533, 0.7899555323044729, 0.8260528380852733, 0.8286685848809835 , 0.8587496730316505, 0.8543029034789432];
train_acc_feature = [0.8537765808808017 ,0.9244758021744186, 0.9427198311939733, 0.9480851328767752, 0.9582127352415544, 0.9637075509675752];

%Entropy 

x_axis_depth_2 = [3, 10, 30, 40, 50];
val_acc_depth_2 = [0.508239602406487, 0.8618885691865027, 0.8535181794402302, 0.8629348679047868  , 0.8616269945069317];
train_acc_depth_2 = [0.5203514762181372 ,0.9818844666688289, 1, 1, 1];

x_axis_feature_2 = [2, 4, 6, 8, 10, 12];
val_acc_feature_2 = [0.7138373005493068, 0.8004185194873136, 0.8401778707821083, 0.8257912634057023 , 0.8456709390530996, 0.8618885691865027];
train_acc_feature_2 = [0.8483592297475482 ,0.947425502445505, 0.9736507521059176, 0.9711605447072837, 0.9782246351214298, 0.9818844666688289];


figure(1)
subplot(2,2,1)
plot(x_axis_depth,train_acc_depth,'b',x_axis_depth,val_acc_depth,'r');
title('Fixed Feature = 12, Depth vs Accuracies, Gini')
xlabel('Depth')
ylabel('Acc')
legend('Train Acc','Val Acc')
grid on
subplot(2,2,3)
plot(x_axis_feature,train_acc_feature,'b',x_axis_feature,val_acc_feature,'r');
title('Fixed Depth = 10, Feature vs Accuracies, Gini')
xlabel('Feature')
ylabel('Acc')
legend('Train Acc','Val Acc')
grid on

subplot(2,2,2)
plot(x_axis_depth_2,train_acc_depth_2,'b',x_axis_depth_2,val_acc_depth_2,'r');
title('Fixed Depth = 10, Feature vs Accuracies, Entropy')
xlabel('Feature')
ylabel('Acc')
legend('Train Acc','Val Acc')
grid on

subplot(2,2,4)
plot(x_axis_feature,train_acc_feature_2,'b',x_axis_feature_2,val_acc_feature_2,'r');
title('Fixed Depth = 10, Feature vs Accuracies, Entropy')
xlabel('Feature')
ylabel('Acc')
legend('Train Acc','Val Acc')
grid on
