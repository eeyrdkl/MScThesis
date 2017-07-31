function test_Robot_Conv(model,predict,w; batchsize=32,saveImages=true)
    @time begin
        printlnFlush("Loading weights from $(w)")
        w = loadNetwork(w)
        colormap = map((x)->convert(RGB,x),linspace(Colors.HSV{Float32}(0.0,1.0,1.0),Colors.HSV{Float32}(240.0,1.0,1.0),5))
        colors = Dict()
        for i=1:length(colormap)
            colors[i] = Float32[colormap[i].r, colormap[i].g, colormap[i].b]
        end
        totalloss = totalcorrect = totalcount = imcounter = 0
        x,y = loadData_Robot(model,"test")
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
            ypred = Array(predict(w,batchx))
            totalloss += pixelwiseSoftloss(ypred,batchy)
            testAcc = pixelwiseAccuracy(ypred,batchy)
            totalcorrect += testAcc[1]
            totalcount += testAcc[2]
            #Image conversion
            if saveImages
                for im=1:size(ypred,4)
                    savePrediction(ypred[:,:,:,im],colors,outputPath(@sprintf("Robot/Conv/%s/%05d.png",model,imcounter)))
                    imcounter += 1
                end
            end
        end
        totalloss /= batchCount
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss,totalcorrect/totalcount*100))
    end
end

function test_Robot_LSTM(model,predict,w; saveImages=true)
    @time begin
        printlnFlush("Loading weights from $(w)")
        @time w = loadNetwork(w)
        colormap = map((x)->convert(RGB,x),linspace(Colors.HSV{Float32}(0.0,1.0,1.0),Colors.HSV{Float32}(240.0,1.0,1.0),5))
        colors = Dict()
        for i=1:length(colormap)
            colors[i] = Float32[colormap[i].r, colormap[i].g, colormap[i].b]
        end
        totalloss = totalcorrect = totalcount = imcounter = 0
        x,y = loadData_Robot(model,"test")
        batchCount = size(y,4)
        h = KnetArray(zeros(Float32,28,28,512,1))
        c = KnetArray(zeros(Float32,28,28,512,1))
        for t=1:batchCount
            printUpdate(@sprintf("Time: %5d/%-5d",t,batchCount))
            #Accuracy
            ypred = Array(predict(w,h,c,KnetArray(x[:,:,:,t:t])))
            totalloss += pixelwiseSoftloss(ypred,y[:,:,:,t:t])
            testAcc = pixelwiseAccuracy(ypred,y[:,:,:,t:t])
            totalcorrect += testAcc[1]
            totalcount += testAcc[2]
            #Image conversion
            if saveImages
                savePrediction(ypred[:,:,:,1],colors,outputPath(@sprintf("Robot/LSTM/%s/%05d.png",model,imcounter)))
                imcounter += 1
            end
        end
        totalloss /= batchCount
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss,totalcorrect/totalcount*100))
    end
end

function run_Robot_Conv_RGBD_test()
    test_Robot_Conv("RGBD",Robot_Conv_RGBD,"Robot/Conv_RGBD.jld"; batchsize=32,saveImages=true)
end

function run_Robot_LSTM_RGBD_test()
    test_Robot_LSTM("RGBD",Robot_LSTM_RGBD,"Robot/LSTM_RGBD.jld"; saveImages=true)
end

export run_Robot_Conv_RGBD_test,run_Robot_LSTM_RGBD_test