clear
clc


addpath /data/usr/matlab_path/npy-matlab-master/npy-matlab


pred = readNPY('pred_score1.npy');
label = readNPY('label_score1.npy');

[a,b] = corr(label, pred)

[p,s] = polyfit(label, pred, 1);
y_fit = polyval(p, label);

scatter(label, pred, 300, '.')
xlim([0,1])
ylim([0,1])
xticks(0:1:1)
yticks(0:1:1)

hold on
plot(label, y_fit, 'LineWidth',3)

[yfit,dy1] = polyconf(p,x,s);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear
clc


addpath /data/usr/matlab_path/npy-matlab-master/npy-matlab


pred = readNPY('pred_score6.npy');
label = readNPY('label_score6.npy');

x=label';
y=pred';

a=[x;y];
a1=a(1,:);
[a1,pos]=sort(a1);
a2=a(2,pos);    
[p,s]=polyfit(a1,a2,1);

y1= polyval(p,a1);

[yfit,dy1] = polyconf(p,x,s);

[yfit,dy] = polyconf(p,a1,s,'predopt','curve');
hold on

% fill([a1,fliplr(a1)],[yfit-dy,fliplr(yfit+dy)],[255/255 204/255 255/255],'EdgeColor','none');
% fill([a1,fliplr(a1)],[yfit-dy1,fliplr(yfit+dy1)],[255/255 204/255 255/255],'EdgeColor','none');
plot(a1,y1+dy1,'r--',a1,y1-dy1,'r--','LineWidth',3)
hold on 
plot(a1,y1,'r','linewidth',3)
hold on;
scatter(x,y,'fill');
xlim([0,1])
ylim([0,1])
xticks(0:1:1)
yticks(0:1:1)


%R2
r2 = 1 - (sum((y1 - a2).^2) / sum((a2 - mean(a2)).^2))








%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc


addpath /data/usr/matlab_path/npy-matlab-master/npy-matlab


pred = readNPY('pred_score4_v1.npy');
label = readNPY('label_score4_v1.npy');


a = pred(find(label~=0));
b = label(find(label~=0));

% a = a(71:240,1);
% b = b(71:240,1);

a = a(61:230,1);
b = b(61:230,1);
corr(a,b)


scatter(a,b)
label = b;
pred = a;


[a,b] = corr(label, pred)

p = polyfit(label, pred, 1);
y_fit = polyval(p, label);

scatter(label, pred, 300, '.')
xlim([0.2,1])
ylim([0,1])
xticks(0.2:0.8:1)
yticks(0:1:1)

hold on
plot(label, y_fit, 'LineWidth',3)
%%%%%%%%%%%%%%%%%%








clear
clc


addpath /data/usr/matlab_path/npy-matlab-master/npy-matlab


pred = readNPY('pred_score4_v1.npy');
label = readNPY('label_score4_v1.npy');


a = pred(find(label~=0));
b = label(find(label~=0));

% a = a(71:240,1);
% b = b(71:240,1);

a = a(61:230,1);
b = b(61:230,1);

label = b;
pred = a;

x=label';
y=pred';

a=[x;y];
a1=a(1,:);
[a1,pos]=sort(a1);
a2=a(2,pos);     
[p,s]=polyfit(a1,a2,1);

y1= polyval(p,a1);

[yfit,dy1] = polyconf(p,x,s);

[yfit,dy] = polyconf(p,a1,s,'predopt','curve');
hold on

% fill([a1,fliplr(a1)],[yfit-dy,fliplr(yfit+dy)],[255/255 204/255 255/255],'EdgeColor','none');
% fill([a1,fliplr(a1)],[yfit-dy1,fliplr(yfit+dy1)],[255/255 204/255 255/255],'EdgeColor','none');
plot(a1,y1+dy1,'r--',a1,y1-dy1,'r--','LineWidth',3)
hold on 
plot(a1,y1,'r','linewidth',3)
hold on;
scatter(x,y,'fill');
xlim([0.15,1])
ylim([0,1])
xticks(0:1:1)
yticks(0:1:1)



