function lgraph = generate_network_v2(x, M, MM)
%
%
%

filter_width = x(1); 
filter_depth = x(2); res_loops = x(3); fn = x(4);
fw1 = filter_width;
fw2 = filter_width*2-1;
fd1 = filter_depth; %fd2 = fd1;
fd2 = floor(filter_depth/2)+1; if fd2<1; fd2 = 1; end
%fn = 8;
if fw1-2<2; pd1 = 2; else; pd1 = fw1-2; end
switch res_loops
    case 1
        layerz = [imageInputLayer([MM M 1], 'Name', 'Input')
        convolution2dLayer([fw2 fd1], fn, 'stride',2, 'Padding', 'same', 'Name', 'CL10') 
        batchNormalizationLayer('Name', 'BN0') 
        reluLayer('Name', 'Relu10') % use longer filters as fs is higher
        averagePooling2dLayer([pd1 fd1],'Stride', [2 2], 'Padding', 'same', 'Name', 'AV1')
        convolution2dLayer([fw1 fd2], fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL11')
        batchNormalizationLayer('Name', 'BN1') 
        reluLayer('Name', 'Relu11')
        convolution2dLayer([fw1 fd2], fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL12')
        batchNormalizationLayer('Name', 'BN2')         
        additionLayer(2,'Name', 'add1')
        reluLayer('Name', 'Relu12')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',2, 'Padding', 'same', 'Name', 'CL13')
        batchNormalizationLayer('Name', 'BN3')         
        reluLayer('Name', 'Relu13')
        convolution2dLayer([fw1 fd2], 2*fn, 'stride',1, 'Padding', 'same', 'Name', 'CL20')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN4') 
        additionLayer(2,'Name', 'add2')
        reluLayer('Name', 'Relu20')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL21')
        reluLayer('Name', 'Relu21')
        batchNormalizationLayer('Name', 'BN5')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL22')
        batchNormalizationLayer('Name', 'BN6')
        additionLayer(2,'Name', 'add3')
        reluLayer('Name', 'Relu22')
        averagePooling2dLayer([fw1-1 1],'Stride', 2, 'Padding', [1 0], 'Name', 'AV2')
  %      dropoutLayer(0.5, 'Name', 'DO1')
        fullyConnectedLayer(1, 'Name', 'FC1')
        regressionLayer('Name', 'Reg1')];
        lgraph = layerGraph(layerz);
        lgraph = connectLayers(lgraph,'AV1','add1/in2');
        skip1 = [convolution2dLayer(1, 2*fn,'Stride', 2, 'Name','skipConv1')
                 batchNormalizationLayer('Name','skipBN1')];
        lgraph = addLayers(lgraph,skip1);
        lgraph = connectLayers(lgraph,'Relu12','skipConv1');
        lgraph = connectLayers(lgraph,'skipBN1','add2/in2');
        lgraph = connectLayers(lgraph,'Relu20','add3/in2');
               
    case 2
        layerz = [imageInputLayer([MM M 1], 'Name', 'Input')
        convolution2dLayer([fw2 fd1], fn, 'stride',2, 'Padding', 'same', 'Name', 'CL10') 
        batchNormalizationLayer('Name', 'BN0') 
        reluLayer('Name', 'Relu10') % use longer filters as fs is higher
        averagePooling2dLayer([pd1 fd1],'Stride', [2 2], 'Padding', 'same', 'Name', 'AV1')
        convolution2dLayer([fw1 fd2], fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL11')
        batchNormalizationLayer('Name', 'BN1') 
        reluLayer('Name', 'Relu11')
        convolution2dLayer([fw1 fd2], fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL12')
        batchNormalizationLayer('Name', 'BN2')         
        additionLayer(2,'Name', 'add1')
        reluLayer('Name', 'Relu12')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',[2 1], 'Padding', 'same', 'Name', 'CL13')
        batchNormalizationLayer('Name', 'BN3')         
        reluLayer('Name', 'Relu13')
        convolution2dLayer([fw1 fd2], 2*fn, 'stride',1, 'Padding', 'same', 'Name', 'CL20')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN4') 
        additionLayer(2,'Name', 'add2')
        reluLayer('Name', 'Relu20')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL21')
        reluLayer('Name', 'Relu21')
        batchNormalizationLayer('Name', 'BN5')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL22')
        batchNormalizationLayer('Name', 'BN6')
        additionLayer(2,'Name', 'add3')
        reluLayer('Name', 'Relu22')
        convolution2dLayer([fw1 fd2], 4*fn, 'stride',[2 1], 'Padding', 'same', 'Name', 'CL23')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN7')
        reluLayer('Name', 'Relu23')
        convolution2dLayer([fw1 fd2], 4*fn, 'stride',1, 'Padding', 'same', 'Name', 'CL30')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN8')
        additionLayer(2,'Name', 'add4')
        reluLayer('Name', 'Relu30')
        convolution2dLayer([fw1 fd2], 4*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL31')
        batchNormalizationLayer('Name', 'BN9')
        reluLayer('Name', 'Relu31')
        convolution2dLayer([fw1 fd2], 4*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL32')
        batchNormalizationLayer('Name', 'BN10')
        additionLayer(2,'Name', 'add5')
        reluLayer('Name', 'Relu32')
        averagePooling2dLayer([fw1-1 1],'Stride', [2 1], 'Padding', [1 0], 'Name', 'AV2')
    %    dropoutLayer(0.35, 'Name', 'DO1')
        fullyConnectedLayer(1, 'Name', 'FC1')
        regressionLayer('Name', 'Reg1')];
        lgraph = layerGraph(layerz);
        lgraph = connectLayers(lgraph,'AV1','add1/in2');
        skip1 = [convolution2dLayer(1, 2*fn,'Stride', [2 1], 'Name','skipConv1')
                 batchNormalizationLayer('Name','skipBN1')];
        lgraph = addLayers(lgraph,skip1);
        lgraph = connectLayers(lgraph,'Relu12','skipConv1');
        lgraph = connectLayers(lgraph,'skipBN1','add2/in2');
        lgraph = connectLayers(lgraph,'Relu20','add3/in2');
        skip2 = [convolution2dLayer(1, 4*fn,'Stride', [2 1], 'Name','skipConv2')
                 batchNormalizationLayer('Name','skipBN2')];
        lgraph = addLayers(lgraph, skip2);
        lgraph = connectLayers(lgraph,'Relu22','skipConv2');
        lgraph = connectLayers(lgraph,'skipBN2','add4/in2');
        lgraph = connectLayers(lgraph,'Relu30','add5/in2');
                
    case 3
        layerz = [imageInputLayer([MM M 1], 'Name', 'Input')
        convolution2dLayer([fw2 fd1], fn, 'stride',2, 'Padding', 'same', 'Name', 'CL10') 
        batchNormalizationLayer('Name', 'BN0') 
        reluLayer('Name', 'Relu10') % use longer filters as fs is higher
        averagePooling2dLayer([pd1 fd1],'Stride', [2 2], 'Padding', 'same', 'Name', 'AV1')
        convolution2dLayer([fw1 fd2], fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL11')
        batchNormalizationLayer('Name', 'BN1') 
        reluLayer('Name', 'Relu11')
        convolution2dLayer([fw1 fd2], fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL12')
        batchNormalizationLayer('Name', 'BN2')         
        additionLayer(2,'Name', 'add1')
        reluLayer('Name', 'Relu12')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',[2 1], 'Padding', 'same', 'Name', 'CL13')
        batchNormalizationLayer('Name', 'BN3')         
        reluLayer('Name', 'Relu13')
        convolution2dLayer([fw1 fd2], 2*fn, 'stride',1, 'Padding', 'same', 'Name', 'CL20')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN4') 
        additionLayer(2,'Name', 'add2')
        reluLayer('Name', 'Relu20')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL21')
        reluLayer('Name', 'Relu21')
        batchNormalizationLayer('Name', 'BN5')
        convolution2dLayer([fw1 fd2], 2*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL22')
        batchNormalizationLayer('Name', 'BN6')
        additionLayer(2,'Name', 'add3')
        reluLayer('Name', 'Relu22')
        convolution2dLayer([fw1 fd2], 4*fn, 'stride',[2 1], 'Padding', 'same', 'Name', 'CL23')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN7')
        reluLayer('Name', 'Relu23')
        convolution2dLayer([fw1 fd2], 4*fn, 'stride',1, 'Padding', 'same', 'Name', 'CL30')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN8')
        additionLayer(2,'Name', 'add4')
        reluLayer('Name', 'Relu30')
        convolution2dLayer([fw1 fd2], 4*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL31')
        batchNormalizationLayer('Name', 'BN9')
        reluLayer('Name', 'Relu31')
        convolution2dLayer([fw1 fd2], 4*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL32')
        batchNormalizationLayer('Name', 'BN10')
        additionLayer(2,'Name', 'add5')
        reluLayer('Name', 'Relu32')
        convolution2dLayer([fw1 fd2], 8*fn, 'Stride',[2 1], 'Padding', 'same', 'Name', 'CL33')
        batchNormalizationLayer('Name', 'BN11')
        reluLayer('Name', 'Relu33')
        convolution2dLayer([fw1 fd2], 8*fn, 'stride',1, 'Padding', 'same', 'Name', 'CL40')                % use longer filters as fs is higher
        batchNormalizationLayer('Name', 'BN12')
        additionLayer(2,'Name', 'add6')
        reluLayer('Name', 'Relu40')
        convolution2dLayer([fw1 fd2], 8*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL41')
        batchNormalizationLayer('Name', 'BN13')
        reluLayer('Name', 'Relu41')
        convolution2dLayer([fw1 fd2], 8*fn, 'Stride',1, 'Padding', 'same', 'Name', 'CL42')
        batchNormalizationLayer('Name', 'BN14')
        additionLayer(2,'Name', 'add7')
        reluLayer('Name', 'Relu42')        
        averagePooling2dLayer([fw1-1 1],'Stride', 2, 'Padding', 'same', 'Name', 'AV2')
      %  dropoutLayer(0.25, 'Name', 'DO1')
        fullyConnectedLayer(1, 'Name', 'FC1')
        regressionLayer('Name', 'Reg1')];
        lgraph = layerGraph(layerz);
        lgraph = connectLayers(lgraph,'AV1','add1/in2');
        skip1 = [convolution2dLayer(1, 2*fn,'Stride', [2 1], 'Name','skipConv1')
                 batchNormalizationLayer('Name','skipBN1')];
        lgraph = addLayers(lgraph,skip1);
        lgraph = connectLayers(lgraph,'Relu12','skipConv1');
        lgraph = connectLayers(lgraph,'skipBN1','add2/in2');
        lgraph = connectLayers(lgraph,'Relu20','add3/in2');
        skip2 = [convolution2dLayer(1, 4*fn,'Stride', [2 1], 'Name','skipConv2')
                 batchNormalizationLayer('Name','skipBN2')];
        lgraph = addLayers(lgraph, skip2);
        lgraph = connectLayers(lgraph,'Relu22','skipConv2');
        lgraph = connectLayers(lgraph,'skipBN2','add4/in2');
        lgraph = connectLayers(lgraph,'Relu30','add5/in2');
        skip3 = [convolution2dLayer(1, 8*fn,'Stride', [2 1], 'Name','skipConv3')
                 batchNormalizationLayer('Name','skipBN3')];
        lgraph = addLayers(lgraph, skip3);
        lgraph = connectLayers(lgraph,'Relu32','skipConv3');
        lgraph = connectLayers(lgraph,'skipBN3','add6/in2');
        lgraph = connectLayers(lgraph,'Relu40','add7/in2');
        
    case 4
                
        lgraph = layerGraph();
        tempLayers = [imageInputLayer([MM M 1],"Name","input_1", "Normalization","rescale-symmetric")
                       convolution2dLayer([fw2 fw2], fn, 'stride', 2, 'Padding', 'same', 'Name', 'cnv_1') 
                       batchNormalizationLayer('Name', 'BN0') 
                       reluLayer('Name', 'Relu10') % use longer filters as fs is higher
                       averagePooling2dLayer([2 2],'Stride', [2 2], 'Padding', 'same', 'Name', 'AV1')];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],fn,"Name","conv2d_7","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_7","Epsilon",0.001)
            reluLayer("Name","activation_7_relu")
            convolution2dLayer([fw2 fw2],2*fn,"Name","conv2d_8","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_8","Epsilon",0.001)
            reluLayer("Name","activation_8_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],2*fn,"Name","conv2d_6","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_6","Epsilon",0.001)
            reluLayer("Name","activation_6_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],fn,"Name","conv2d_9","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_9","Epsilon",0.001)
            reluLayer("Name","activation_9_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_10","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_10","Epsilon",0.001)
            reluLayer("Name","activation_10_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_11","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_11","Epsilon",0.001)
            reluLayer("Name","activation_11_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [averagePooling2dLayer([fw1 fw1],"Name","average_pooling2d_1","Padding","same")
            convolution2dLayer([1 1],2*fn,"Name","conv2d_12","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_12","Epsilon",0.001)
            reluLayer("Name","activation_12_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(4,"Name","mixed0");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [globalAveragePooling2dLayer("Name","avg_pool")
            fullyConnectedLayer(1,"Name","predictions")
            regressionLayer("Name","ClassificationLayer_predictions")];
        lgraph = addLayers(lgraph,tempLayers);
        lgraph = connectLayers(lgraph,"AV1","conv2d_7");
        lgraph = connectLayers(lgraph,"AV1","conv2d_6");
        lgraph = connectLayers(lgraph,"AV1","conv2d_9");
        lgraph = connectLayers(lgraph,"AV1","average_pooling2d_1");
        lgraph = connectLayers(lgraph,"activation_6_relu","mixed0/in1");
        lgraph = connectLayers(lgraph,"activation_11_relu","mixed0/in3");
        lgraph = connectLayers(lgraph,"activation_8_relu","mixed0/in2");
        lgraph = connectLayers(lgraph,"activation_12_relu","mixed0/in4");
        lgraph = connectLayers(lgraph,"mixed0", "avg_pool");
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%NETWORK TYPE 2
    case 5
        lgraph = layerGraph();
        tempLayers = [imageInputLayer([MM M 1],"Name","input_1", "Normalization","rescale-symmetric")
               convolution2dLayer([fw2 fw2], fn, 'stride', 2, 'Padding', 'same', 'Name', 'cnv_1') 
               batchNormalizationLayer('Name', 'BN0') 
               reluLayer('Name', 'Relu10') % use longer filters as fs is higher
               averagePooling2dLayer([2 2],'Stride', [2 2], 'Padding', 'same', 'Name', 'AV1')];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1], fn*2,"Name","conv2d_78","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_78","Epsilon",0.001)
            reluLayer("Name","activation_78_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [averagePooling2dLayer([fw1 fw1],"Name","average_pooling2d_8","Padding","same")
            convolution2dLayer([1 1], fn,"Name","conv2d_85","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_85","Epsilon",0.001)
            reluLayer("Name","activation_85_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1], 2*fn,"Name","conv2d_81","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_81","Epsilon",0.001)
            reluLayer("Name","activation_81_relu")
            convolution2dLayer([fw1 fw1], 16,"Name","conv2d_82","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_82","Epsilon",0.001)
            reluLayer("Name","activation_82_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 fw1],2*fn,"Name","conv2d_79","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_79","Epsilon",0.001)
            reluLayer("Name","activation_79_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1], 2*fn,"Name","conv2d_77","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_77","Epsilon",0.001)
            reluLayer("Name","activation_77_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([fw1 1], 2*fn,"Name","conv2d_84","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_84","Epsilon",0.001)
            reluLayer("Name","activation_84_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 fw1], 2*fn,"Name","conv2d_83","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_83","Epsilon",0.001)
            reluLayer("Name","activation_83_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(2,"Name","concatenate_1");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([fw1 1],2*fn,"Name","conv2d_80","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_80","Epsilon",0.001)
            reluLayer("Name","activation_80_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(2,"Name","mixed9_0");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(4,"Name","mixed9");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [globalAveragePooling2dLayer("Name","avg_pool")
            fullyConnectedLayer(1,"Name","predictions")
            regressionLayer("Name","ClassificationLayer_predictions")];
        lgraph = addLayers(lgraph,tempLayers);
        lgraph = connectLayers(lgraph,"AV1","conv2d_78");
        lgraph = connectLayers(lgraph,"AV1","conv2d_81");
        lgraph = connectLayers(lgraph,"AV1","conv2d_77");
        lgraph = connectLayers(lgraph,"AV1","average_pooling2d_8");
        lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_79");
        lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_80");
        lgraph = connectLayers(lgraph,"activation_79_relu","mixed9_0/in1");
        lgraph = connectLayers(lgraph,"activation_80_relu","mixed9_0/in2");
        lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_84");
        lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_83");
        lgraph = connectLayers(lgraph,"activation_84_relu","concatenate_1/in2");
        lgraph = connectLayers(lgraph,"activation_83_relu","concatenate_1/in1");
        lgraph = connectLayers(lgraph,"mixed9_0","mixed9/in1");
        lgraph = connectLayers(lgraph,"concatenate_1","mixed9/in3");
        lgraph = connectLayers(lgraph,"activation_85_relu","mixed9/in2");
        lgraph = connectLayers(lgraph,"activation_77_relu","mixed9/in4");
        lgraph = connectLayers(lgraph,"mixed9", "avg_pool");

% NETWORK TYPE THREE

    case 6

        lgraph = layerGraph();
        tempLayers = [imageInputLayer([MM M 1],"Name","input_1", "Normalization","rescale-symmetric")
                       convolution2dLayer([fw2 fw2], fn, 'stride', 2, 'Padding', 'same', 'Name', 'cnv_1') 
                       batchNormalizationLayer('Name', 'BN0') 
                       reluLayer('Name', 'Relu10') % use longer filters as fs is higher
                       averagePooling2dLayer([2 2],'Stride', [2 2], 'Padding', 'same', 'Name', 'AV1')];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],2*fn,"Name","conv2d_73","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_73","Epsilon",0.001)
            reluLayer("Name","activation_73_relu")
            convolution2dLayer([1 fw2+2],2*fn,"Name","conv2d_74","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_74","Epsilon",0.001)
            reluLayer("Name","activation_74_relu")
            convolution2dLayer([fw2+2 1],2*fn,"Name","conv2d_75","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_75","Epsilon",0.001)
            reluLayer("Name","activation_75_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_76","BiasLearnRateFactor",0,"Stride",[2 2],"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_76","Epsilon",0.001)
            reluLayer("Name","activation_76_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],2*fn,"Name","conv2d_71","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_71","Epsilon",0.001)
            reluLayer("Name","activation_71_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_72","BiasLearnRateFactor",0,"Stride",[2 2],"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_72","Epsilon",0.001)
            reluLayer("Name","activation_72_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = averagePooling2dLayer([3 3],'Stride', [2 2], 'Padding', 'same', 'Name', 'average_pooling2d_4');
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(3,"Name","mixed8");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [globalAveragePooling2dLayer("Name","avg_pool")
            fullyConnectedLayer(1,"Name","predictions")
            %softmaxLayer("Name","predictions_softmax")
            regressionLayer("Name","ClassificationLayer_predictions")];
        lgraph = addLayers(lgraph,tempLayers);
        lgraph = connectLayers(lgraph,"AV1","conv2d_73");
        lgraph = connectLayers(lgraph,"AV1","conv2d_71");
        lgraph = connectLayers(lgraph,"AV1","average_pooling2d_4");
        lgraph = connectLayers(lgraph,"average_pooling2d_4","mixed8/in3");
        lgraph = connectLayers(lgraph,"activation_72_relu","mixed8/in1");
        lgraph = connectLayers(lgraph,"activation_76_relu","mixed8/in2");
        lgraph = connectLayers(lgraph,"mixed8", "avg_pool");     
        
    case 7
        
        lgraph = layerGraph();
        tempLayers = [imageInputLayer([MM M 1],"Name","input_1", "Normalization","rescale-symmetric")
                       convolution2dLayer([fw2 fw2], fn, 'stride', 2, 'Padding', 'same', 'Name', 'cnv_1') 
                       batchNormalizationLayer('Name', 'BN0') 
                       reluLayer('Name', 'Relu10') % use longer filters as fs is higher
                       averagePooling2dLayer([2 2],'Stride', [2 2], 'Padding', 'same', 'Name', 'AV1')];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],fn,"Name","conv2d_7","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_7","Epsilon",0.001)
            reluLayer("Name","activation_7_relu")
            convolution2dLayer([fw2 fw2],2*fn,"Name","conv2d_8","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_8","Epsilon",0.001)
            reluLayer("Name","activation_8_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],2*fn,"Name","conv2d_6","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_6","Epsilon",0.001)
            reluLayer("Name","activation_6_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],fn,"Name","conv2d_9","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_9","Epsilon",0.001)
            reluLayer("Name","activation_9_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_10","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_10","Epsilon",0.001)
            reluLayer("Name","activation_10_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_11","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_11","Epsilon",0.001)
            reluLayer("Name","activation_11_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [averagePooling2dLayer([fw1 fw1],"Name","average_pooling2d_1","Padding","same")
            convolution2dLayer([1 1],2*fn,"Name","conv2d_12","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_12","Epsilon",0.001)
            reluLayer("Name","activation_12_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(4,"Name","mixed0");
        lgraph = addLayers(lgraph,tempLayers);
                tempLayers = [convolution2dLayer([1 1],2*fn,"Name","conv2d_73","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_73","Epsilon",0.001)
            reluLayer("Name","activation_73_relu")
            convolution2dLayer([1 fw2+2],2*fn,"Name","conv2d_74","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_74","Epsilon",0.001)
            reluLayer("Name","activation_74_relu")
            convolution2dLayer([fw2+2 1],2*fn,"Name","conv2d_75","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_75","Epsilon",0.001)
            reluLayer("Name","activation_75_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_76","BiasLearnRateFactor",0,"Stride",[2 2],"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_76","Epsilon",0.001)
            reluLayer("Name","activation_76_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1],2*fn,"Name","conv2d_71","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_71","Epsilon",0.001)
            reluLayer("Name","activation_71_relu")
            convolution2dLayer([fw1 fw1],2*fn,"Name","conv2d_72","BiasLearnRateFactor",0,"Stride",[2 2],"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_72","Epsilon",0.001)
            reluLayer("Name","activation_72_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = averagePooling2dLayer([3 3],'Stride', [2 2], 'Padding', 'same', 'Name', 'average_pooling2d_4');
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(3,"Name","mixed8");
        lgraph = addLayers(lgraph,tempLayers);
       
         tempLayers = [convolution2dLayer([1 1], fn*2,"Name","conv2d_78","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_78","Epsilon",0.001)
            reluLayer("Name","activation_78_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [averagePooling2dLayer([fw1 fw1],"Name","average_pooling2d_8","Padding","same")
            convolution2dLayer([1 1], fn,"Name","conv2d_85","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_85","Epsilon",0.001)
            reluLayer("Name","activation_85_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1], 2*fn,"Name","conv2d_81","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_81","Epsilon",0.001)
            reluLayer("Name","activation_81_relu")
            convolution2dLayer([fw1 fw1], 16,"Name","conv2d_82","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_82","Epsilon",0.001)
            reluLayer("Name","activation_82_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 fw1],2*fn,"Name","conv2d_79","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_79","Epsilon",0.001)
            reluLayer("Name","activation_79_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 1], 2*fn,"Name","conv2d_77","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_77","Epsilon",0.001)
            reluLayer("Name","activation_77_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([fw1 1], 2*fn,"Name","conv2d_84","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_84","Epsilon",0.001)
            reluLayer("Name","activation_84_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([1 fw1], 2*fn,"Name","conv2d_83","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_83","Epsilon",0.001)
            reluLayer("Name","activation_83_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(2,"Name","concatenate_1");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [convolution2dLayer([fw1 1],2*fn,"Name","conv2d_80","BiasLearnRateFactor",0,"Padding","same")
            batchNormalizationLayer("Name","batch_normalization_80","Epsilon",0.001)
            reluLayer("Name","activation_80_relu")];
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(2,"Name","mixed9_0");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = depthConcatenationLayer(4,"Name","mixed9");
        lgraph = addLayers(lgraph,tempLayers);
        tempLayers = [globalAveragePooling2dLayer("Name","avg_pool")
            fullyConnectedLayer(1,"Name","predictions")
            regressionLayer("Name","ClassificationLayer_predictions")];
        lgraph = addLayers(lgraph,tempLayers);
               

        lgraph = connectLayers(lgraph,"AV1","conv2d_7");
        lgraph = connectLayers(lgraph,"AV1","conv2d_6");
        lgraph = connectLayers(lgraph,"AV1","conv2d_9");
        lgraph = connectLayers(lgraph,"AV1","average_pooling2d_1");
        lgraph = connectLayers(lgraph,"activation_6_relu","mixed0/in1");
        lgraph = connectLayers(lgraph,"activation_11_relu","mixed0/in3");
        lgraph = connectLayers(lgraph,"activation_8_relu","mixed0/in2");
        lgraph = connectLayers(lgraph,"activation_12_relu","mixed0/in4");
        

        lgraph = connectLayers(lgraph,"mixed0","conv2d_73");
        lgraph = connectLayers(lgraph,"mixed0","conv2d_71");
        lgraph = connectLayers(lgraph,"mixed0","average_pooling2d_4");
        lgraph = connectLayers(lgraph,"average_pooling2d_4","mixed8/in3");
        lgraph = connectLayers(lgraph,"activation_72_relu","mixed8/in1");
        lgraph = connectLayers(lgraph,"activation_76_relu","mixed8/in2");
        %lgraph = connectLayers(lgraph,"mixed8", "avg_pool");     
        
        lgraph = connectLayers(lgraph,"mixed8","conv2d_78");
        lgraph = connectLayers(lgraph,"mixed8","conv2d_81");
        lgraph = connectLayers(lgraph,"mixed8","conv2d_77");
        lgraph = connectLayers(lgraph,"mixed8","average_pooling2d_8");
        lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_79");
        lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_80");
        lgraph = connectLayers(lgraph,"activation_79_relu","mixed9_0/in1");
        lgraph = connectLayers(lgraph,"activation_80_relu","mixed9_0/in2");
        lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_84");
        lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_83");
        lgraph = connectLayers(lgraph,"activation_84_relu","concatenate_1/in2");
        lgraph = connectLayers(lgraph,"activation_83_relu","concatenate_1/in1");
        lgraph = connectLayers(lgraph,"mixed9_0","mixed9/in1");
        lgraph = connectLayers(lgraph,"concatenate_1","mixed9/in3");
        lgraph = connectLayers(lgraph,"activation_85_relu","mixed9/in2");
        lgraph = connectLayers(lgraph,"activation_77_relu","mixed9/in4");
        
        lgraph = connectLayers(lgraph,"mixed9", "avg_pool");     
       


end
