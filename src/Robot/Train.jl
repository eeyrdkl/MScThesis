function train_Robot_Conv(model,predict,loss,w; lr=1e-5,epochs=100,batchsize=24,testInterval=1)
    @time begin
        w = map(KnetArray,w)
        m = map(zeros,w)
        v = map(zeros,w)
        beta1 = 0.9; beta2 = 0.999; adamT = 1; eps = 1e-8;
        printlnFlush("lr=$(lr)\tepochs=$(epochs)\tbatchsize=$(batchsize)\tmodel=Conv_$(model)\ttestInterval=$(testInterval)")
        printlnFlush(string("Number of parameters: ",sum(map(length,w))," (",Int(floor(sum(map(length,w))/1e6))," million)"))
        printlnFlush("Loading data...")
        @time xtrain,ytrain = loadData_Robot(model,"train")
        @time xtest,ytest = loadData_Robot(model,"test")
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
                saveNetwork(w,"Robot_Conv_$(model)")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end

function train_Robot_LSTM(model,predict,loss,w; lr=1e-5,epochs=100,testInterval=1)
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
        @time xtrain,ytrain = loadData_Robot(model,"train")
        @time xtest,ytest = loadData_Robot(model,"test")
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
                saveNetwork(w,"Robot_LSTM_$(model)")
            end
            if epoch == epochs
                printlnFlush(@sprintf("Maximum Test Accuracy: %0.4f%%",maxacc))
            end
        end
    end
end

#   Conv
function run_Robot_Conv_D_train()
    train_Robot_Conv("D",Robot_Conv_D,Robot_Conv_D_loss,Robot_Conv_D_w(); lr=1e-5,epochs=100,batchsize=16,testInterval=1)
end
function run_Robot_Conv_RGB_train()
    train_Robot_Conv("RGB",Robot_Conv_RGB,Robot_Conv_RGB_loss,Robot_Conv_RGB_w(); lr=1e-5,epochs=100,batchsize=16,testInterval=1)
end
function run_Robot_Conv_RGBD_train()
    train_Robot_Conv("RGBD",Robot_Conv_RGBD,Robot_Conv_RGBD_loss,Robot_Conv_RGBD_w(); lr=1e-5,epochs=50,batchsize=8,testInterval=1)
end

#   LSTM
function run_Robot_LSTM_D_train()
    train_Robot_LSTM("D",Robot_LSTM_D,Robot_LSTM_D_loss,Robot_LSTM_D_w(); lr=1e-5,epochs=50,testInterval=1)
end
function run_Robot_LSTM_RGB_train()
    train_Robot_LSTM("RGB",Robot_LSTM_RGB,Robot_LSTM_RGB_loss,Robot_LSTM_RGB_w(); lr=1e-5,epochs=50,testInterval=1)
end
function run_Robot_LSTM_RGBD_train()
    train_Robot_LSTM("RGBD",Robot_LSTM_RGBD,Robot_LSTM_RGBD_loss,Robot_LSTM_RGBD_w(); lr=1e-5,epochs=50,testInterval=1)
end

#   Conv
export run_Robot_Conv_D_train
export run_Robot_Conv_RGB_train
export run_Robot_Conv_RGBD_train
#   LSTM
export run_Robot_LSTM_D_train
export run_Robot_LSTM_RGB_train
export run_Robot_LSTM_RGBD_train