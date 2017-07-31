function train_vKITTI_Conv(model,predict,loss,w; lr=1e-5,epochs=100,batchsize=24,testInterval=1,
    trainVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    trainMode="odd",
    testMode="even"
    )
    @time begin
        w = map(KnetArray,w)
        m = map(zeros,w)
        v = map(zeros,w)
        beta1 = 0.9; beta2 = 0.999; adamT = 1; eps = 1e-8;
        printlnFlush("lr=$(lr)\tepochs=$(epochs)\tbatchsize=$(batchsize)\tmodel=Conv_$(model)\ttestInterval=$(testInterval)")
        printlnFlush(string("Number of parameters: ",sum(map(length,w))," (",Int(floor(sum(map(length,w))/1e6))," million)"))
        printlnFlush("Loading data...")
        @time xtrain,ytrain,colors = loadData_vKITTI(worlds,trainVariations,model,trainMode)
        @time xtest,ytest,colors = loadData_vKITTI(worlds,testVariations,model,testMode)
        printlnFlush("Training...")
        grdnt = gradloss(loss)
        trainbatchCount = Int64(ceil(size(ytrain,4)/batchsize))
        testbatchCount = Int64(ceil(size(ytest,4)/batchsize))
        maxacc = 0
        for epoch=1:epochs
            trainloss = testloss = trainncorrect = testncorrect = trainncount = testncount = 0
            for batch=1:trainbatchCount
                printUpdate(@sprintf("Batch: %5d/%-5d",batch,trainbatchCount))
                #Calculate current indices
                startIndex = 1+(batch-1)*batchsize
                endIndex = min(batch*batchsize,size(ytrain,4))
                #Minibatch
                batchx = KnetArray(xtrain[:,:,:,startIndex:endIndex])
                batchy = ytrain[:,:,:,startIndex:endIndex]
                g,l = grdnt(w,batchx,batchy)
                trainloss += l
                #Adam
                for i in 1:length(w)
                    m[i] = beta1*m[i] + (1-beta1)*g[i]
                    v[i] = beta2*v[i] + (1-beta2)*(g[i].*g[i])
                    mhat = m[i] ./ (1-beta1^adamT)
                    vhat = v[i] ./ (1-beta2^adamT)
                    w[i] = w[i] - (lr ./ (sqrt(vhat)+eps)).*mhat
                end
                adamT += 1
            end
            #Testing accuracy
            if epoch%testInterval == 0 || epoch == 1
                for batch=1:testbatchCount
                    printUpdate(@sprintf("Batch: %5d/%-5d",batch,testbatchCount))
                    #Calculate current indices
                    startIndex = 1+(batch-1)*batchsize
                    endIndex = min(batch*batchsize,size(ytest,4))
                    #Minibatch
                    batchx = KnetArray(xtest[:,:,:,startIndex:endIndex])
                    batchy = ytest[:,:,:,startIndex:endIndex]
                    #Accuracy
                    ypred = Array(predict(w,batchx))
                    testloss += pixelwiseSoftloss(ypred,batchy)
                    testAcc = pixelwiseAccuracy(ypred,batchy)
                    testncorrect += testAcc[1]
                    testncount += testAcc[2]
                end
            end
            testaccuracy = testncorrect/testncount*100
            if isnan(testaccuracy)
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f",epoch,trainloss/trainbatchCount))
            else
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f\tTest Loss: %0.10f\tTest Accuracy: %0.4f%%",
                    epoch,trainloss/trainbatchCount,testloss/testbatchCount,testaccuracy))
            end
            if testaccuracy > maxacc
                maxacc = testaccuracy
                saveNetwork(w,"vKITTI_Conv_$(model)")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end

function train_vKITTI_RNN(model,predict,loss,w; lr=1e-5,epochs=100,testInterval=1,
    trainVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    trainMode="odd",
    testMode="even"
    )
    @time begin
        D = 1
        w = map(KnetArray,w)
        m = map(zeros,w)
        v = map(zeros,w)
        h = KnetArray(zeros(Float32,28,28,512,1))
        beta1 = 0.9; beta2 = 0.999; adamT = 1; eps = 1e-8;
        printlnFlush("lr=$(lr)\tepochs=$(epochs)\tmodel=RNN_$(model)\ttestInterval=$(testInterval)")
        printlnFlush(string("Number of parameters: ",sum(map(length,w))," (",Int(floor(sum(map(length,w))/1e6))," million)"))
        printlnFlush("Loading data...")
        @time xtrain,ytrain,colors = loadData_vKITTI(worlds,trainVariations,model,trainMode)
        @time xtest,ytest,colors = loadData_vKITTI(worlds,testVariations,model,testMode)
        printlnFlush("Training...")
        grdnt = gradloss(loss)
        trainbatchCount = size(ytrain,4)
        testbatchCount = size(ytest,4)
        maxacc = 0
        for epoch=1:epochs
            trainloss = testloss = trainncorrect = testncorrect = trainncount = testncount = 0
            #Train
            h = zeros(h)
            for t=1:trainbatchCount
                printUpdate(@sprintf("Time: %5d/%-5d",t,trainbatchCount))
                g,l = grdnt(w,h,xtrain,ytrain,t,D)
                trainloss += l
                #Adam
                for i in 1:length(w)
                    m[i] = beta1*m[i] + (1-beta1)*g[i]
                    v[i] = beta2*v[i] + (1-beta2)*(g[i].*g[i])
                    mhat = m[i] ./ (1-beta1^adamT)
                    vhat = v[i] ./ (1-beta2^adamT)
                    w[i] = w[i] - (lr ./ (sqrt(vhat)+eps)).*mhat
                end
                adamT += 1
            end
            #Testing accuracy
            if epoch%testInterval == 0 || epoch == 1
                h = zeros(h)
                for t=1:testbatchCount
                    printUpdate(@sprintf("Time: %5d/%-5d",t,testbatchCount))
                    ypred = Array(predict(w,h,KnetArray(xtest[:,:,:,t:t])))
                    testloss += pixelwiseSoftloss(ypred,ytest[:,:,:,t:t])
                    testAcc = pixelwiseAccuracy(ypred,ytest[:,:,:,t:t])
                    testncorrect += testAcc[1]
                    testncount += testAcc[2]
                end
            end
            testaccuracy = testncorrect/testncount*100
            if isnan(testaccuracy)
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f",epoch,trainloss/trainbatchCount))
            else
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f\tTest Loss: %0.10f\tTest Accuracy: %0.4f%%",
                    epoch,trainloss/trainbatchCount,testloss/testbatchCount,testaccuracy))
            end
            if testaccuracy > maxacc
                maxacc = testaccuracy
                saveNetwork(w,"vKITTI_RNN_$(model)")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end

function train_vKITTI_LSTM(model,predict,loss,w; lr=1e-5,epochs=100,testInterval=1,
    trainVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    trainMode="odd",
    testMode="even"
    )
    @time begin
        D = 1
        w = map(KnetArray,w)
        m = map(zeros,w)
        v = map(zeros,w)
        h = KnetArray(zeros(Float32,28,28,512,1))
        c = KnetArray(zeros(Float32,28,28,512,1))
        beta1 = 0.9; beta2 = 0.999; adamT = 1; eps = 1e-8;
        printlnFlush("lr=$(lr)\tepochs=$(epochs)\tmodel=LSTM_$(model)\ttestInterval=$(testInterval)")
        printlnFlush(string("Number of parameters: ",sum(map(length,w))," (",Int(floor(sum(map(length,w))/1e6))," million)"))
        printlnFlush("Loading data...")
        @time xtrain,ytrain,colors = loadData_vKITTI(worlds,trainVariations,model,trainMode)
        @time xtest,ytest,colors = loadData_vKITTI(worlds,testVariations,model,testMode)
        printlnFlush("Training...")
        grdnt = gradloss(loss)
        trainbatchCount = size(ytrain,4)
        testbatchCount = size(ytest,4)
        maxacc = 0
        for epoch=1:epochs
            trainloss = testloss = trainncorrect = testncorrect = trainncount = testncount = 0
            #Train
            h = zeros(h)
            c = zeros(c)
            for t=1:trainbatchCount
                printUpdate(@sprintf("Time: %5d/%-5d",t,trainbatchCount))
                g,l = grdnt(w,h,c,xtrain,ytrain,t,D)
                trainloss += l
                #Adam
                for i in 1:length(w)
                    m[i] = beta1*m[i] + (1-beta1)*g[i]
                    v[i] = beta2*v[i] + (1-beta2)*(g[i].*g[i])
                    mhat = m[i] ./ (1-beta1^adamT)
                    vhat = v[i] ./ (1-beta2^adamT)
                    w[i] = w[i] - (lr ./ (sqrt(vhat)+eps)).*mhat
                end
                adamT += 1
            end
            #Testing accuracy
            if epoch%testInterval == 0 || epoch == 1
                h = zeros(h)
                c = zeros(c)
                for t=1:testbatchCount
                    printUpdate(@sprintf("Time: %5d/%-5d",t,testbatchCount))
                    ypred = Array(predict(w,h,c,KnetArray(xtest[:,:,:,t:t])))
                    testloss += pixelwiseSoftloss(ypred,ytest[:,:,:,t:t])
                    testAcc = pixelwiseAccuracy(ypred,ytest[:,:,:,t:t])
                    testncorrect += testAcc[1]
                    testncount += testAcc[2]
                end
            end
            testaccuracy = testncorrect/testncount*100
            if isnan(testaccuracy)
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f",epoch,trainloss/trainbatchCount))
            else
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f\tTest Loss: %0.10f\tTest Accuracy: %0.4f%%",
                    epoch,trainloss/trainbatchCount,testloss/testbatchCount,testaccuracy))
            end
            if testaccuracy > maxacc
                maxacc = testaccuracy
                saveNetwork(w,"vKITTI_LSTM_$(model)")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end

function train_vKITTI_GRU(model,predict,loss,w; lr=1e-5,epochs=100,testInterval=1,
    trainVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    trainMode="odd",
    testMode="even"
    )
    @time begin
        D = 1
        w = map(KnetArray,w)
        m = map(zeros,w)
        v = map(zeros,w)
        h = KnetArray(zeros(Float32,28,28,512,1))
        beta1 = 0.9; beta2 = 0.999; adamT = 1; eps = 1e-8;
        printlnFlush("lr=$(lr)\tepochs=$(epochs)\tmodel=GRU_$(model)\ttestInterval=$(testInterval)")
        printlnFlush(string("Number of parameters: ",sum(map(length,w))," (",Int(floor(sum(map(length,w))/1e6))," million)"))
        printlnFlush("Loading data...")
        @time xtrain,ytrain,colors = loadData_vKITTI(worlds,trainVariations,model,trainMode)
        @time xtest,ytest,colors = loadData_vKITTI(worlds,testVariations,model,testMode)
        printlnFlush("Training...")
        grdnt = gradloss(loss)
        trainbatchCount = size(ytrain,4)
        testbatchCount = size(ytest,4)
        maxacc = 0
        for epoch=1:epochs
            trainloss = testloss = trainncorrect = testncorrect = trainncount = testncount = 0
            #Train
            h = zeros(h)
            for t=1:trainbatchCount
                printUpdate(@sprintf("Time: %5d/%-5d",t,trainbatchCount))
                g,l = grdnt(w,h,xtrain,ytrain,t,D)
                trainloss += l
                #Adam
                for i in 1:length(w)
                    m[i] = beta1*m[i] + (1-beta1)*g[i]
                    v[i] = beta2*v[i] + (1-beta2)*(g[i].*g[i])
                    mhat = m[i] ./ (1-beta1^adamT)
                    vhat = v[i] ./ (1-beta2^adamT)
                    w[i] = w[i] - (lr ./ (sqrt(vhat)+eps)).*mhat
                end
                adamT += 1
            end
            #Testing accuracy
            if epoch%testInterval == 0 || epoch == 1
                h = zeros(h)
                for t=1:testbatchCount
                    printUpdate(@sprintf("Time: %5d/%-5d",t,testbatchCount))
                    ypred = Array(predict(w,h,KnetArray(xtest[:,:,:,t:t])))
                    testloss += pixelwiseSoftloss(ypred,ytest[:,:,:,t:t])
                    testAcc = pixelwiseAccuracy(ypred,ytest[:,:,:,t:t])
                    testncorrect += testAcc[1]
                    testncount += testAcc[2]
                end
            end
            testaccuracy = testncorrect/testncount*100
            if isnan(testaccuracy)
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f",epoch,trainloss/trainbatchCount))
            else
                printlnFlush(@sprintf("Epoch: %-6d\tTrain Loss: %0.10f\tTest Loss: %0.10f\tTest Accuracy: %0.4f%%",
                    epoch,trainloss/trainbatchCount,testloss/testbatchCount,testaccuracy))
            end
            if testaccuracy > maxacc
                maxacc = testaccuracy
                saveNetwork(w,"vKITTI_GRU_$(model)")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end


#E1 : Degrees
#E2 : Half Train/Test

#   Conv
#E1
function run_vKITTI_Conv_D_E1_train()
    train_vKITTI_Conv("D_VGG",vKITTI_Conv_D,vKITTI_Conv_D_loss,vKITTI_Conv_D_w(); lr=1e-5,epochs=100,batchsize=24,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_Conv_RGB_E1_train()
    train_vKITTI_Conv("RGB_VGG",vKITTI_Conv_RGB,vKITTI_Conv_RGB_loss,vKITTI_Conv_RGB_w(); lr=1e-5,epochs=100,batchsize=24,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_Conv_RGBD_E1_train()
    train_vKITTI_Conv("RGBD_VGG",vKITTI_Conv_RGBD,vKITTI_Conv_RGBD_loss,vKITTI_Conv_RGBD_w(); lr=1e-5,epochs=50,batchsize=16,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
#E2
function run_vKITTI_Conv_D_E2_train()
    train_vKITTI_Conv("D_VGG",vKITTI_Conv_D,vKITTI_Conv_D_loss,vKITTI_Conv_D_w(); lr=1e-5,epochs=100,batchsize=24,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        trainMode="firstHalf",
        testMode="secondHalf"
    )
end
function run_vKITTI_Conv_RGB_E2_train()
    train_vKITTI_Conv("RGB_VGG",vKITTI_Conv_RGB,vKITTI_Conv_RGB_loss,vKITTI_Conv_RGB_w(); lr=1e-5,epochs=100,batchsize=24,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        trainMode="firstHalf",
        testMode="secondHalf"
    )
end
function run_vKITTI_Conv_RGBD_E2_train()
    train_vKITTI_Conv("RGBD_VGG",vKITTI_Conv_RGBD,vKITTI_Conv_RGBD_loss,vKITTI_Conv_RGBD_w(); lr=1e-5,epochs=50,batchsize=16,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        trainMode="firstHalf",
        testMode="secondHalf"
    )
end

#   RNN
function run_vKITTI_RNN_D_E1_train()
    train_vKITTI_RNN("D_VGG",vKITTI_RNN_D,vKITTI_RNN_D_loss,vKITTI_RNN_D_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_RNN_RGB_E1_train()
    train_vKITTI_RNN("RGB_VGG",vKITTI_RNN_RGB,vKITTI_RNN_RGB_loss,vKITTI_RNN_RGB_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_RNN_RGBD_E1_train()
    train_vKITTI_RNN("RGBD_VGG",vKITTI_RNN_RGBD,vKITTI_RNN_RGBD_loss,vKITTI_RNN_RGBD_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end

#   LSTM
#E1
function run_vKITTI_LSTM_D_E1_train()
    train_vKITTI_LSTM("D_VGG",vKITTI_LSTM_D,vKITTI_LSTM_D_loss,vKITTI_LSTM_D_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_LSTM_RGB_E1_train()
    train_vKITTI_LSTM("RGB_VGG",vKITTI_LSTM_RGB,vKITTI_LSTM_RGB_loss,vKITTI_LSTM_RGB_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_LSTM_RGBD_E1_train()
    train_vKITTI_LSTM("RGBD_VGG",vKITTI_LSTM_RGBD,vKITTI_LSTM_RGBD_loss,vKITTI_LSTM_RGBD_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
#E2
function run_vKITTI_LSTM_D_E2_train()
    train_vKITTI_LSTM("D_VGG",vKITTI_LSTM_D,vKITTI_LSTM_D_loss,vKITTI_LSTM_D_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        trainMode="firstHalf",
        testMode="secondHalf"
    )
end
function run_vKITTI_LSTM_RGB_E2_train()
    train_vKITTI_LSTM("RGB_VGG",vKITTI_LSTM_RGB,vKITTI_LSTM_RGB_loss,vKITTI_LSTM_RGB_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        trainMode="firstHalf",
        testMode="secondHalf"
    )
end
function run_vKITTI_LSTM_RGBD_E2_train()
    train_vKITTI_LSTM("RGBD_VGG",vKITTI_LSTM_RGBD,vKITTI_LSTM_RGBD_loss,vKITTI_LSTM_RGBD_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        trainMode="firstHalf",
        testMode="secondHalf"
    )
end

#   GRU
function run_vKITTI_GRU_D_E1_train()
    train_vKITTI_GRU("D_VGG",vKITTI_GRU_D,vKITTI_GRU_D_loss,vKITTI_GRU_D_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_GRU_RGB_E1_train()
    train_vKITTI_GRU("RGB_VGG",vKITTI_GRU_RGB,vKITTI_GRU_RGB_loss,vKITTI_GRU_RGB_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end
function run_vKITTI_GRU_RGBD_E1_train()
    train_vKITTI_GRU("RGBD_VGG",vKITTI_GRU_RGBD,vKITTI_GRU_RGBD_loss,vKITTI_GRU_RGBD_w(); lr=1e-5,epochs=25,testInterval=1,
        trainVariations=["30-deg-left","30-deg-right","clone"],
        testVariations=["15-deg-left","15-deg-right"],
        trainMode="full",
        testMode="full"
    )
end

#   Conv
#E1
export run_vKITTI_Conv_D_E1_train
export run_vKITTI_Conv_RGB_E1_train
export run_vKITTI_Conv_RGBD_E1_train
#E2
export run_vKITTI_Conv_D_E2_train
export run_vKITTI_Conv_RGB_E2_train
export run_vKITTI_Conv_RGBD_E2_train
#   RNN
export run_vKITTI_RNN_D_E1_train
export run_vKITTI_RNN_RGB_E1_train
export run_vKITTI_RNN_RGBD_E1_train
#   LSTM
#E1
export run_vKITTI_LSTM_D_E1_train
export run_vKITTI_LSTM_RGB_E1_train
export run_vKITTI_LSTM_RGBD_E1_train
#E2
export run_vKITTI_LSTM_D_E2_train
export run_vKITTI_LSTM_RGB_E2_train
export run_vKITTI_LSTM_RGBD_E2_train
#   GRU
export run_vKITTI_GRU_D_E1_train
export run_vKITTI_GRU_RGB_E1_train
export run_vKITTI_GRU_RGBD_E1_train