function test_vKITTI_Conv(model,predict,w; batchsize=32,saveImages=true,
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testMode="even",
    experiment="First_Second"
    )
    @time begin
        printlnFlush("Loading weights from $(w)")
        w = loadNetwork(w)
        totalloss = totalcorrect = totalcount = 0
        for world in worlds
            for variation in testVariations
                x,y,colors = loadData_vKITTI([world],[variation],model,testMode)
                batchCount = Int64(ceil(size(y,4)/batchsize))
                varloss = imcounter = varcorrect = varcount = 0
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
                    varloss += pixelwiseSoftloss(ypred,batchy)
                    testAcc = pixelwiseAccuracy(ypred,batchy)
                    varcorrect += testAcc[1]
                    varcount += testAcc[2]
                    #Image conversion
                    if saveImages
                        for im=1:size(ypred,4)
                            savePrediction(ypred[:,:,:,im],colors,outputPath(@sprintf("vKITTI/%s/Conv/%s/%s/%s/%05d.png",experiment,model,world,variation,imcounter)))
                            imcounter += 1
                        end
                    end
                end
                printlnFlush(@sprintf("World: %s\tVariation: %s\tLoss: %0.10f\tAccuracy: %0.4f%%",world,variation,varloss/batchCount,varcorrect/varcount*100))
                totalloss += varloss/batchCount
                totalcorrect += varcorrect
                totalcount += varcount
            end
        end
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss,totalcorrect/totalcount*100))
    end
end

function test_vKITTI_RNN(model,predict,w; saveImages=true,
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testMode="even",
    experiment="First_Second"
    )
    @time begin
        printlnFlush("Loading weights from $(w)")
        @time w = loadNetwork(w)
        totalloss = totalcorrect = totalcount = 0
        for world in worlds
            for variation in testVariations
                varloss = imcounter = varcorrect = varcount = 0
                x,y,colors = loadData_vKITTI([world],[variation],model,testMode)
                batchCount = size(y,4)
                h = KnetArray(zeros(Float32,28,28,512,1))
                for t=1:batchCount
                    printUpdate(@sprintf("Time: %5d/%-5d",t,batchCount))
                    #Accuracy
                    ypred = Array(predict(w,h,KnetArray(x[:,:,:,t:t])))
                    varloss += pixelwiseSoftloss(ypred,y[:,:,:,t:t])
                    testAcc = pixelwiseAccuracy(ypred,y[:,:,:,t:t])
                    varcorrect += testAcc[1]
                    varcount += testAcc[2]
                    #Image conversion
                    if saveImages
                        savePrediction(ypred[:,:,:,1],colors,outputPath(@sprintf("vKITTI/%s/RNN/%s/%s/%s/%05d.png",experiment,model,world,variation,imcounter)))
                        imcounter += 1
                    end
                end
                printlnFlush(@sprintf("World: %s\tVariation: %s\tLoss: %0.10f\tAccuracy: %0.4f%%",world,variation,varloss/batchCount,varcorrect/varcount*100))
                totalloss += varloss/batchCount
                totalcorrect += varcorrect
                totalcount += varcount
            end
        end
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss,totalcorrect/totalcount*100))
    end
end

function test_vKITTI_LSTM(model,predict,w; saveImages=true,
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testMode="even",
    experiment="First_Second"
    )
    @time begin
        printlnFlush("Loading weights from $(w)")
        @time w = loadNetwork(w)
        totalloss = totalcorrect = totalcount = 0
        for world in worlds
            for variation in testVariations
                varloss = imcounter = varcorrect = varcount = 0
                x,y,colors = loadData_vKITTI([world],[variation],model,testMode)
                batchCount = size(y,4)
                h = KnetArray(zeros(Float32,28,28,512,1))
                c = KnetArray(zeros(Float32,28,28,512,1))
                for t=1:batchCount
                    printUpdate(@sprintf("Time: %5d/%-5d",t,batchCount))
                    #Accuracy
                    ypred = Array(predict(w,h,c,KnetArray(x[:,:,:,t:t])))
                    varloss += pixelwiseSoftloss(ypred,y[:,:,:,t:t])
                    testAcc = pixelwiseAccuracy(ypred,y[:,:,:,t:t])
                    varcorrect += testAcc[1]
                    varcount += testAcc[2]
                    #Image conversion
                    if saveImages
                        savePrediction(ypred[:,:,:,1],colors,outputPath(@sprintf("vKITTI/%s/LSTM/%s/%s/%s/%05d.png",experiment,model,world,variation,imcounter)))
                        imcounter += 1
                    end
                end
                printlnFlush(@sprintf("World: %s\tVariation: %s\tLoss: %0.10f\tAccuracy: %0.4f%%",world,variation,varloss/batchCount,varcorrect/varcount*100))
                totalloss += varloss/batchCount
                totalcorrect += varcorrect
                totalcount += varcount
            end
        end
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss,totalcorrect/totalcount*100))
    end
end

function test_vKITTI_GRU(model,predict,w; saveImages=true,
    testVariations=["30-deg-left","30-deg-right","clone","fog","rain"],
    testMode="even",
    experiment="First_Second"
    )
    @time begin
        printlnFlush("Loading weights from $(w)")
        @time w = loadNetwork(w)
        totalloss = totalcorrect = totalcount = 0
        for world in worlds
            for variation in testVariations
                varloss = imcounter = varcorrect = varcount = 0
                x,y,colors = loadData_vKITTI([world],[variation],model,testMode)
                batchCount = size(y,4)
                h = KnetArray(zeros(Float32,28,28,512,1))
                for t=1:batchCount
                    printUpdate(@sprintf("Time: %5d/%-5d",t,batchCount))
                    #Accuracy
                    ypred = Array(predict(w,h,KnetArray(x[:,:,:,t:t])))
                    varloss += pixelwiseSoftloss(ypred,y[:,:,:,t:t])
                    testAcc = pixelwiseAccuracy(ypred,y[:,:,:,t:t])
                    varcorrect += testAcc[1]
                    varcount += testAcc[2]
                    #Image conversion
                    if saveImages
                        savePrediction(ypred[:,:,:,1],colors,outputPath(@sprintf("vKITTI/%s/GRU/%s/%s/%s/%05d.png",experiment,model,world,variation,imcounter)))
                        imcounter += 1
                    end
                end
                printlnFlush(@sprintf("World: %s\tVariation: %s\tLoss: %0.10f\tAccuracy: %0.4f%%",world,variation,varloss/batchCount,varcorrect/varcount*100))
                totalloss += varloss/batchCount
                totalcorrect += varcorrect
                totalcount += varcount
            end
        end
        printlnFlush(@sprintf("Total Loss: %0.10f\tTotal Accuracy: %0.4f%%",totalloss,totalcorrect/totalcount*100))
    end
end

#E1 : Degrees
#E2 : Half Train/Test

#   Conv
function run_vKITTI_Conv_RGBD_E2_test()
    test_vKITTI_Conv("RGBD_VGG",vKITTI_Conv_RGBD,"vKITTI/First_Second/Conv_RGBD.jld"; batchsize=32,saveImages=true,
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testMode="secondHalf",
        experiment="First_Second")
end

#   LSTM
function run_vKITTI_LSTM_RGBD_E2_test()
    test_vKITTI_LSTM("RGBD_VGG",vKITTI_LSTM_RGBD,"vKITTI/First_Second/LSTM_RGBD.jld"; saveImages=true,
        testVariations=["30-deg-left","30-deg-right","15-deg-left","15-deg-right","clone","fog","morning","overcast","rain","sunset"],
        testMode="secondHalf",
        experiment="First_Second")
end

export run_vKITTI_Conv_RGBD_E2_test
export run_vKITTI_LSTM_RGBD_E2_test
