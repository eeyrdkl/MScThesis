function vKITTI_GRU_RGB(w,h,x)
    x = relu(conv4(w[1],x;padding=1).+w[2])
    x = relu(conv4(w[3],x;padding=1).+w[4])
    x = pool(x)
    x = relu(conv4(w[5],x;padding=1).+w[6])
    x = relu(conv4(w[7],x;padding=1).+w[8])
    x = pool(x)
    x = relu(conv4(w[9],x;padding=1).+w[10])
    x = relu(conv4(w[11],x;padding=1).+w[12])
    x = relu(conv4(w[13],x;padding=1).+w[14])
    x = relu(conv4(w[15],x;padding=1).+w[16])
    x = pool(x)
    x = relu(conv4(w[17],x;padding=1).+w[18])
    x = relu(conv4(w[19],x;padding=1).+w[20])
    x = relu(conv4(w[21],x;padding=1).+w[22])
    x = relu(conv4(w[23],x;padding=1).+w[24])
    #28,28,512,1

    r = sigm(conv4(w[25],x;padding=1) .+ conv4(w[26],h;padding=1) .+ w[27])
    z = sigm(conv4(w[28],x;padding=1) .+ conv4(w[29],h;padding=1) .+ w[30])
    h_h = tanh(conv4(w[31],x;padding=1) .+ conv4(w[32],r.*h;padding=1) .+ w[33])
    h = z.*h .+ (1.-z).*h_h
    
    x = relu(conv4(w[34],h).+w[35])
    x = deconv4(w[36],x;padding=1,stride=6).+w[37]
end

function vKITTI_GRU_RGB_w()
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
        #GRU
        xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
        xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
        xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
        #Scores
        xavier(Float32,1,1,512,13),      zeros(Float32,1,1,13,1),
        #Deconv
        bilinear(Float32,32,32,13,13),   zeros(Float32,1,1,13,1),
    ]
    w[1:24] = load(pretrainedPath("vKITTI_Conv_RGB.jld"),"w")[1:24]
    return w
end

function vKITTI_GRU_RGB_loss(w,h,x,y,t,D)
    total = 0.0
    for i in max(t-D,1):t
        ypred = vKITTI_GRU_RGB(w,h,KnetArray(x[:,:,:,i:i]))
        total += pixelwiseSoftloss(Array(ypred),y[:,:,:,i:i])
    end
    return total
end