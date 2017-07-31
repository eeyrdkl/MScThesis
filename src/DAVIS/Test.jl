function test_DAVIS_Conv(w; batchsize=32,saveImages=true)
    @time begin
        printlnFlush("Loading weights from $(w)")
        w = loadNetwork(w)
        colors = [Float32[0.0,0.0,0.0], Float32[1.0,1.0,1.0]]
        xtrain,ytrain,x,y = loadData_DAVIS()
        totalloss = totalcorrect = totalcount = imcounter = 0
        batchCount = Int64(ceil(size(y,4)/batchsize))
        for batch=1:batchCount
            printUpdate(@sprintf("Batch: %5d/%-5d",batch,batchCount))
            #Calculate current indices
            startIndex = 1+(batch-1)*batchsize
            endIndex = min(batch*batchsize,size(y,4))
            #Minibatch
            batchx = KnetArray(x[:,:,:,startIndex:endIndex])
            batchy = y[:,:,:,startIndex:endIndex]
            #Accuracy
            ypred = Array(DAVIS_Conv(w,batchx))
            totalloss += pixelwiseSoftloss(ypred,batchy)
            testAcc = pixelwiseAccuracy(ypred,batchy)
            totalcorrect += testAcc[1]
            totalcount += testAcc[2]
            #Image conversion
            if saveImages
                for im=1:size(ypred,4)
                    savePrediction(ypred[:,:,:,im],colors,outputPath(@sprintf("DAVIS/Conv/%05d.png",imcounter)))
                    imcounter += 1
                end
            end
        end
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss/batchCount,totalcorrect/totalcount*100))
    end
end

function test_DAVIS_LSTM(w; saveImages=true)
    @time begin
        printlnFlush("Loading weights from $(w)")
        @time w = loadNetwork(w)
        colors = [Float32[0.0,0.0,0.0], Float32[1.0,1.0,1.0]]
        totalloss = totalcorrect = totalcount = imcounter = 0
        xtrain,ytrain,x,y = loadData_DAVIS()
        batchCount = size(y,4)
        h = KnetArray(zeros(Float32,28,28,512,1))
        c = KnetArray(zeros(Float32,28,28,512,1))
        for t=1:batchCount
            printUpdate(@sprintf("Time: %5d/%-5d",t,batchCount))
            #Accuracy
            ypred = Array(DAVIS_LSTM(w,h,c,KnetArray(x[:,:,:,t:t])))
            totalloss += pixelwiseSoftloss(ypred,y[:,:,:,t:t])
            testAcc = pixelwiseAccuracy(ypred,y[:,:,:,t:t])
            totalcorrect += testAcc[1]
            totalcount += testAcc[2]
            #Image conversion
            if saveImages
                savePrediction(ypred[:,:,:,1],colors,outputPath(@sprintf("DAVIS/LSTM/%05d.png",imcounter)))
                imcounter += 1
            end
        end
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss/batchCount,totalcorrect/totalcount*100))
    end
end

function run_DAVIS_Conv_test()
    test_DAVIS_Conv("DAVIS/Conv.jld"; batchsize=32,saveImages=true)
end

function run_DAVIS_LSTM_test()
    test_DAVIS_LSTM("DAVIS/LSTM.jld"; saveImages=true)
end

export run_DAVIS_Conv_test,run_DAVIS_LSTM_test