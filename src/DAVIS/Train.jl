function train_DAVIS_Conv(; lr=1e-5,epochs=100,batchsize=24,testInterval=1)
    @time begin
        w = map(KnetArray,DAVIS_Conv_w())
        m = map(zeros,w)
        v = map(zeros,w)
        beta1 = 0.9; beta2 = 0.999; adamT = 1; eps = 1e-8;
        printlnFlush("lr=$(lr)\tepochs=$(epochs)\tbatchsize=$(batchsize)\tmodel=Conv\ttestInterval=$(testInterval)")
        printlnFlush(string("Number of parameters: ",sum(map(length,w))," (",Int(floor(sum(map(length,w))/1e6))," million)"))
        printlnFlush("Loading data...")
        @time xtrain,ytrain,xtest,ytest = loadData_DAVIS()
        printlnFlush("Training...")
        grdnt = gradloss(DAVIS_Conv_loss)
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
                    ypred = Array(DAVIS_Conv(w,batchx))
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
                saveNetwork(w,"DAVIS_Conv")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end

function train_DAVIS_LSTM(; lr=1e-5,epochs=100,testInterval=1)
    @time begin
        D = 1
        w = map(KnetArray,DAVIS_LSTM_w())
        m = map(zeros,w)
        v = map(zeros,w)
        h = KnetArray(zeros(Float32,28,28,512,1))
        c = KnetArray(zeros(Float32,28,28,512,1))
        beta1 = 0.9; beta2 = 0.999; adamT = 1; eps = 1e-8;
        printlnFlush("lr=$(lr)\tepochs=$(epochs)\tmodel=LSTM\ttestInterval=$(testInterval)")
        printlnFlush(string("Number of parameters: ",sum(map(length,w))," (",Int(floor(sum(map(length,w))/1e6))," million)"))
        printlnFlush("Loading data...")
        @time xtrain,ytrain,xtest,ytest = loadData_DAVIS()
        printlnFlush("Training...")
        grdnt = gradloss(DAVIS_LSTM_loss)
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
                    ypred = Array(DAVIS_LSTM(w,h,c,KnetArray(xtest[:,:,:,t:t])))
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
                saveNetwork(w,"DAVIS_LSTM")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end

function run_DAVIS_Conv_train()
    train_DAVIS_Conv(; lr=1e-5,epochs=100,batchsize=24,testInterval=1)
end

function run_DAVIS_LSTM_train()
    train_DAVIS_LSTM(; lr=1e-5,epochs=25,testInterval=1)
end

export run_DAVIS_Conv_train,run_DAVIS_LSTM_train