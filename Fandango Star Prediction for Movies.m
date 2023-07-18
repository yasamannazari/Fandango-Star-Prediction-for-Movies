clc;
%converting a csv file to mat file
Dataset=readtable('fandango.csv');
%save('fandango.mat','Dataset','-mat');


%adding a new label column to the table
Dataset.label(Dataset.Fandango_Stars >= 3.5) = 1;
Dataset.label(Dataset.Fandango_Stars < 3.5) = 0;

%dropping some of the unimportant columns
Dataset(:,3:8) = [];
Dataset(:,"FILM") = [];
Dataset(:,"Fandango_Stars")=[];
T = table2array(Dataset);

%splitting data into train and test
d_tr = T(1:117,1:15);
d_te = T(118:146,1:15);

train_D = zeros(117,15);
m = zeros(1,117);
v = zeros(1,117);
for i = 1:117
    xi = d_tr(i,:);
    m(i) = mean(xi);
    v(i) = sqrt(var(xi));
    train_D(i,:) = (xi - m(i))/v(i);
end

test_D = zeros(29,15);
for i = 1:29
    xi = d_te(i,:);
    test_D(i,:) = (xi - m(i))/v(i);
end


%keeping the lable column for both train and test sets
ytr = T(1:117,16);
yte = T(118:146,16);
disp(ytr);
%training and testing dataset to use
tr = [train_D ytr];
te = [test_D yte];
%disp(te)
%defining the confusion matrix parameters
P = find(yte == 1);
N = find(yte == 0);
E = zeros(2,length(test_D));

%setting the initial point
w_h = zeros(17,1);

mu = 0.075;
K = 30;

%using the GD algorithm to minimize cost function
[Ws, f, k] = cg('f_LRBC', 'g_LRBC', zeros(16,1), train_D', ytr', 0.001);
% w1 = grad_desc('f_softmax', 'g_softmax', zeros(16,1), train_D', mu,K);

Dte = [d_te'; ones(1, 29)];

test_labels = yte';
displayResultGeneric(Dte, Ws, test_labels);
 
% %making the confusion matrix
% for i = 1: length(train_D)
%     xi= [train_D(i,:); ones(1, 15)];
%     yi = sign(xi.*w1');
%     eval(['ti = sign(w',num2str(i),'*xi);']);
%     idx = -yi/2 + 1.5;
%     E(idx,i) = 1;
% end
% 
% E1 = E(:,P);
% E2 = E(:,N);
% C1 = sum(E1,2);
% C2 = sum(E2,2);
% C = [C1,C2];

% %showing the confusion matrix
% disp('confusion matrix is: ');
% disp(C);

% %showing the accuracy
% acc = trace(C) / length(train_D);
% disp('The accuracy is = ');
% disp(acc);

%the softmax function 
function f = f_softmax(w,D,mu)
X = D(1:16,:);
y = D(18,:);
P = length(y);
f = 0;
for i = 1:P
xi = [X(:,i); 1];
fi = log(1+exp(-y(i)*(w'*xi)));
f = f + fi;
end
%adding the regularization term
f = f/P + 0.5*mu*norm(w)^2;
end

%gradient of softmax function
function g = g_softmax(w,D,mu)
X = D(1:16,:);
y = D(18,:);
P = length(y);
g = zeros(17,1);
for i = 1:P
    xi = [X(:,i); 1];
    ei = exp(y(i)*(w'*xi));
    gi = -y(i)*xi/(1+ei);
    g = g + gi;
end
g = g/P + mu*w;

end

function a = bt_lsearch2019(x,d,fname,gname,p1,p2)
rho = 0.1;
gma = 0.5;
x = x(:);
d = d(:);
a = 1;
xw = x + a*d;
parameterstring ='';
if nargin == 5
   if ischar(p1)
      eval([p1 ';']);
   else
      parameterstring = ',p1';
   end
end
if nargin == 6
   if ischar(p1)
      eval([p1 ';']);
   else
      parameterstring = ',p1';
   end
   if ischar(p2)
      eval([p2 ';']);
   else
      parameterstring = ',p1,p2';
   end
end
eval(['f0 = ' fname '(x' parameterstring ');']);
eval(['g0 = ' gname '(x' parameterstring ');']);
eval(['f1 = ' fname '(xw' parameterstring ');']);
t0 = rho*(g0'*d);
f2 = f0 + a*t0;
er = f1 - f2;
while er > 0
     a = gma*a;
     xw = x + a*d;
     eval(['f1 = ' fname '(xw' parameterstring ');']);
     f2 = f0 + a*t0;
     er = f1 - f2;
end
if a < 1e-5
   a = min([1e-5, 0.1/norm(d)]); 
end 
end

%gradient descent algorithm
function xs = grad_desc(fname,gname,x0,D, mu, K)
format compact
format long
k = 1;
xk = x0;
gk = feval(gname, xk, D, mu);
dk = -gk;
ak = bt_lsearch2019(xk,dk, fname,gname,D, mu);
adk = ak*dk;
while k <= K
xk = xk + adk;
gk = feval(gname,xk, D, mu);
dk = -gk;
ak = bt_lsearch2019(xk,dk, fname,gname, D, mu);
adk = ak*dk;
k = k + 1;
end
xs = xk + adk;
end



function f = f_LRBC(w,X)
    P = size(X,2);
    f = sum(log(1+exp(-X'*w)))/P;
end

function g = g_LRBC(w,X)
    P = size(X,2);
    q1 = exp(X'*w);
    q = 1./(1+q1);
    g = -(X*q)/P;
end

function displayResultGeneric(Data_Matrix, Ws, ytest)
    C = zeros(2,2);
    values = sign(Ws' * Data_Matrix);
    
    for value = 1 : length(values)
        if(ytest(value) == 1 && values(value) == 1)
            C(1, 1) = C(1, 1) + 1;
        elseif(ytest(value)== 1 && values(value) == -1)
            C(2, 1) = C(2, 1) + 1;
        elseif(ytest(value) == -1 && values(value) == 1)
            C(1, 2) = C(1, 2) + 1;
        else
            C(2, 2) = C(2, 2) + 1;
        end       
    end

    disp(C)
    accuracy = sum(diag(C))/(sum(C, 'all')) * 100;
    fprintf("\n\nAccuracy: %f\n", accuracy);
end
