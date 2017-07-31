function networkTiming(; iters = 100)
    for model in [vKITTI_Conv_D,vKITTI_Conv_RGB]
        x = KnetArray(zeros(Float32,224,224,3,1));
        w = Array[
            #Conv64
            xavier(Float32,3,3,3,64),        zeros(Float32,1,1,64,1),
            xavier(Float32,3,3,64,64),       zeros(Float32,1,1,64,1),
            #Conv128
            xavier(Float32,3,3,64,128),      zeros(Float32,1,1,128,1),
            xavier(Float32,3,3,128,128),     zeros(Float32,1,1,128,1),
            #Conv256
            xavier(Float32,3,3,128,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            #Conv512
            xavier(Float32,3,3,256,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            #Scores
            xavier(Float32,1,1,512,13),      zeros(Float32,1,1,13,1),
            #Deconv
            bilinear(Float32,32,32,13,13),   zeros(Float32,1,1,13,1),
        ]; w = map(KnetArray,w);

        println(model)
        times = []
        for i=1:iters
            push!(times,(@timed y = model(w,x))[2])
        end
        println("Minimum: $(minimum(times))s ($(1/minimum(times)) FPS) Maximum: $(maximum(times))s ($(1/maximum(times)) FPS) Average: $(mean(times))s ($(1/mean(times)) FPS)")
    end
    for model in [vKITTI_Conv_RGBD,]
        x = KnetArray(zeros(Float32,224,224,6,1));
        w = Array[
            #RGB
            #Conv64
            xavier(Float32,3,3,3,64),        zeros(Float32,1,1,64,1),
            xavier(Float32,3,3,64,64),       zeros(Float32,1,1,64,1),
            #Conv128
            xavier(Float32,3,3,64,128),      zeros(Float32,1,1,128,1),
            xavier(Float32,3,3,128,128),     zeros(Float32,1,1,128,1),
            #Conv256
            xavier(Float32,3,3,128,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            #Conv512
            xavier(Float32,3,3,256,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            #Colorized Depth
            #Conv64
            xavier(Float32,3,3,3,64),        zeros(Float32,1,1,64,1),
            xavier(Float32,3,3,64,64),       zeros(Float32,1,1,64,1),
            #Conv128
            xavier(Float32,3,3,64,128),      zeros(Float32,1,1,128,1),
            xavier(Float32,3,3,128,128),     zeros(Float32,1,1,128,1),
            #Conv256
            xavier(Float32,3,3,128,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            #Conv512
            xavier(Float32,3,3,256,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            #Scores
            xavier(Float32,1,1,512,13),      zeros(Float32,1,1,13,1),
            #Deconv
            bilinear(Float32,32,32,13,13),   zeros(Float32,1,1,13,1),
        ]; w = map(KnetArray,w);

        println(model)
        times = []
        for i=1:iters
            push!(times,(@timed y = model(w,x))[2])
        end
        println("Minimum: $(minimum(times))s ($(1/minimum(times)) FPS) Maximum: $(maximum(times))s ($(1/maximum(times)) FPS) Average: $(mean(times))s ($(1/mean(times)) FPS)")
    end
    println("***********************")
    h = KnetArray(zeros(Float32,28,28,512,1))
    c = KnetArray(zeros(Float32,28,28,512,1))
    for model in [vKITTI_LSTM_D,vKITTI_LSTM_RGB]
        x = KnetArray(zeros(Float32,224,224,3,1));
        w = Array[
            #Conv64
            xavier(Float32,3,3,3,64),        zeros(Float32,1,1,64,1),
            xavier(Float32,3,3,64,64),       zeros(Float32,1,1,64,1),
            #Conv128
            xavier(Float32,3,3,64,128),      zeros(Float32,1,1,128,1),
            xavier(Float32,3,3,128,128),     zeros(Float32,1,1,128,1),
            #Conv256
            xavier(Float32,3,3,128,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            #Conv512
            xavier(Float32,3,3,256,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            #LSTM
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            #Scores
            xavier(Float32,1,1,512,13),      zeros(Float32,1,1,13,1),
            #Deconv
            bilinear(Float32,32,32,13,13),   zeros(Float32,1,1,13,1),
        ]; w = map(KnetArray,w);

        println(model)
        times = []
        for i=1:iters
            push!(times,(@timed y = model(w,h,c,x))[2])
        end
        println("Minimum: $(minimum(times))s ($(1/minimum(times)) FPS) Maximum: $(maximum(times))s ($(1/maximum(times)) FPS) Average: $(mean(times))s ($(1/mean(times)) FPS)")
    end
    for model in [vKITTI_LSTM_RGBD,]
        x = KnetArray(zeros(Float32,224,224,6,1));
        w = Array[
            #RGB
            #Conv64
            xavier(Float32,3,3,3,64),        zeros(Float32,1,1,64,1),
            xavier(Float32,3,3,64,64),       zeros(Float32,1,1,64,1),
            #Conv128
            xavier(Float32,3,3,64,128),      zeros(Float32,1,1,128,1),
            xavier(Float32,3,3,128,128),     zeros(Float32,1,1,128,1),
            #Conv256
            xavier(Float32,3,3,128,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            #Conv512
            xavier(Float32,3,3,256,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            #Colorized Depth
            #Conv64
            xavier(Float32,3,3,3,64),        zeros(Float32,1,1,64,1),
            xavier(Float32,3,3,64,64),       zeros(Float32,1,1,64,1),
            #Conv128
            xavier(Float32,3,3,64,128),      zeros(Float32,1,1,128,1),
            xavier(Float32,3,3,128,128),     zeros(Float32,1,1,128,1),
            #Conv256
            xavier(Float32,3,3,128,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            xavier(Float32,3,3,256,256),     zeros(Float32,1,1,256,1),
            #Conv512
            xavier(Float32,3,3,256,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     zeros(Float32,1,1,512,1),
            #LSTM
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
            #Scores
            xavier(Float32,1,1,512,13),      zeros(Float32,1,1,13,1),
            #Deconv
            bilinear(Float32,32,32,13,13),   zeros(Float32,1,1,13,1),
        ]; w = map(KnetArray,w);

        println(model)
        times = []
        for i=1:iters
            push!(times,(@timed y = model(w,h,c,x))[2])
        end
        println("Minimum: $(minimum(times))s ($(1/minimum(times)) FPS) Maximum: $(maximum(times))s ($(1/maximum(times)) FPS) Average: $(mean(times))s ($(1/mean(times)) FPS)")
    end
end