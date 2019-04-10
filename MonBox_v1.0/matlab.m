close all;
clc
 
data1=load('w1.mat');
data2=load('w2_new.mat');
 
data1=cell2mat(struct2cell(data1));
data2=cell2mat(struct2cell(data2));
 
row2_335=data2(335,:);
 
row2_336=data2(336,:);
row2_416=data2(416,:);
...row21=data2(1,:);
...rows = vertcat(row11,row21);
 
 
...mydata = cell(1, numfiles);
...row1_1=data1(1,:);
 
...plot(row1_2)
 
    numfiles = 5;
    array = [];
 
for k=1:numfiles
    
    data_row=data1(k,:);
    mean_row = mean(data_row);
    array(k,1) = mean_row;
    
    for j=1:numfiles
        
        data=data2(j,:);
        m=mean(data);
        array(j+numfiles,1)=m; 
    end
end
for k=1:numfiles
    
    data=data1(k,:);
    m=std(data);
    array(k,2)=m;
    
    for j=1:numfiles
        
        data=data2(j,:);
        m=std(data);
        array(j+numfiles,2)=m; 
    end
end
    
for k=1:numfiles
    
    data=data1(k,:);
    m=var(data);
    array(k,3)=m;
    
    for j=1:numfiles
        
        data=data2(j,:);
        m=var(data);
        array(j+numfiles,3)=m; 
    end
end
   
 
...ver=array(:);
...csvwrite('dataa1.csv', array)
 
    
   
%{
 
for k=1:numfiles
    data=data1(k,:);
 
    ...h=figure;  
    subplot(500,1,k);
    spectrogram(data, 6300, [], 200, 100, 'yaxis');
    ...set(h,'visible','off');
    set(gca, 'Ylim', [0,6])
    colorbar('hide')
    axis off
    colormap jet
    ...saveas(h,sprintf('w2_%d.png',k));
 
    
 
    data=data+1;
end
 
  subplot of the graphs
    ...data_t=data2(1,:);
    ...subplot(2,1,2);
    ...spectrogram(data_t, 10000, [], 30, 100, 'yaxis');  colorbar;
    ...mydata{k}=spectrogram(row2, 2999, [], 110, 100, 'yaxis'); set (gca, 'Ylim', [0,10]); colorbar;
    ...mydata{k} = imread(jpegFiles(k).name);
    ...  m=data2(1,:);
    ...spectrogram(m, 9000, [], 90,120, 'yaxis');  colorbar;
   to plot the individual graph of a sample
    ... data=data1(1,:);
    ...  plot(data);
%}
  %to fade away the graph     
    ...set(h,'visible','off');
  %logarithm of the data (magnification)
    ...h=figure;      
    ...spectrogram(data, 9000, [], 160, 100, 'yaxis');  colorbar;
    ...ax = gca;
    ...ax.YScale = 'log';
    ...saveas(h,sprintf('w2_%d.jpg',k));
  % controlling the window of view
    ...h=figure;      
    ...spectrogram(data, 240, [], 240, 100, 'yaxis');  
    ...set(gca, 'Ylim', [0,1.2])
    ...colorbar;
    
  % to graph all the w1 or w2 without axis, and exporting each sample as a seperate file
  
  ...for k=1:numfiles
    ...data=data2(k,:);
 
    ...h=figure;  
   
  ...  spectrogram(data, 6300, [], 200, 100, 'yaxis');
    ...set(h,'visible','off');
    ...set(gca, 'Ylim', [0,6])
    ...colorbar('hide')
    ...axis off
    ...colormap jet
    ...saveas(h,sprintf('w2_%d.png',k));
    ...data=data+1;
...end
 
% WHEN YOU WANT TO ADD TWO VECTORS VERTICALLY
 
    ...rows=vertcat(row1_1, row2_1);  
    ...   
% WHEN YOU WANT THREE COLUMNS WITH 500 ELEMENTS IN A ROW, 1000 IN TOTAL
%{
 for k=1:numfiles
    
    data=data1(k,:);
    m=mean(data);
    array(k,1)=m;
    
    for j=1:numfiles
        
        data=data2(j,:);
        m=mean(data);
        array(j+numfiles,1)=m; 
    end
    end
    for k=1:numfiles
    
    data=data1(k,:);
    m=std(data);
    array(k,2)=m;
    
    for j=1:numfiles
        
        data=data2(j,:);
        m=std(data);
        array(j+numfiles,2)=m; 
    end
    end
    
for k=1:numfiles
    
    data=data1(k,:);
    m=var(data);
    array(k,3)=m;
    
    for j=1:numfiles
        
        data=data2(j,:);
        m=var(data);
        array(j+numfiles,3)=m; 
    end
end
...ver=array(:);
csvwrite('dataa.csv', array)
%}
   
...weightedMeanA = sum(A.*B,N)./sum(row2_335)


