function vKITTI_GRU_RGBD(w,h,x)
    #RGB
    x1 = KnetArray(Array(x)[:,:,1:3,:])
    #Colorized Depth
    x2 = KnetArray(Array(x)[:,:,4:6,:])
    #RGB
    x1 = relu(conv4(w[1],x1;padding=1).+w[2])
    x1 = relu(conv4(w[3],x1;padding=1).+w[4])
    x1 = pool(x1)
    x1 = relu(conv4(w[5],x1;padding=1).+w[6])
    x1 = relu(conv4(w[7],x1;padding=1).+w[8])
    x1 = pool(x1)
    x1 = relu(conv4(w[9],x1;padding=1).+w[10])
    x1 = relu(conv4(w[11],x1;padding=1).+w[12])
    x1 = relu(conv4(w[13],x1;padding=1).+w[14])
    x1 = relu(conv4(w[15],x1;padding=1).+w[16])
    x1 = pool(x1)
    x1 = relu(conv4(w[17],x1;padding=1).+w[18])
    x1 = relu(conv4(w[19],x1;padding=1).+w[20])
    x1 = relu(conv4(w[21],x1;padding=1).+w[22])
    x1 = relu(conv4(w[23],x1;padding=1).+w[24])
    #Colorized Depth
    x2 = relu(conv4(w[25],x2;padding=1).+w[26])
    x2 = relu(conv4(w[27],x2;padding=1).+w[28])
    x2 = pool(x2)
    x2 = relu(conv4(w[29],x2;padding=1).+w[30])
    x2 = relu(conv4(w[31],x2;padding=1).+w[32])
    x2 = pool(x2)
    x2 = relu(conv4(w[33],x2;padding=1).+w[34])
    x2 = relu(conv4(w[35],x2;padding=1).+w[36])
    x2 = relu(conv4(w[37],x2;padding=1).+w[38])
    x2 = relu(conv4(w[39],x2;padding=1).+w[40])
    x2 = pool(x2)
    x2 = relu(conv4(w[41],x2;padding=1).+w[42])
    x2 = relu(conv4(w[43],x2;padding=1).+w[44])
    x2 = relu(conv4(w[45],x2;padding=1).+w[46])
    x2 = relu(conv4(w[47],x2;padding=1).+w[48])
    x3 = x1.+x2
    #28,28,512,1

    r = sigm(conv4(w[49],x3;padding=1) .+ conv4(w[50],h;padding=1) .+ w[51])
    z = sigm(conv4(w[52],x3;padding=1) .+ conv4(w[53],h;padding=1) .+ w[54])
    h_h = tanh(conv4(w[55],x3;padding=1) .+ conv4(w[56],r.*h;padding=1) .+ w[57])
    h = z.*h .+ (1.-z).*h_h
    
    x3 = relu(conv4(w[58],h).+w[59])
    x3 = deconv4(w[60],x3;padding=1,stride=6).+w[61]
end

function vKITTI_GRU_RGBD_w()
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
        #GRU
        xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
        xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
        xavier(Float32,3,3,512,512),     xavier(Float32,3,3,512,512),   zeros(Float32,1,1,512,1),
        #Scores
        xavier(Float32,1,1,512,13),      zeros(Float32,1,1,13,1),
        #Deconv
        bilinear(Float32,32,32,13,13),   zeros(Float32,1,1,13,1),
    ]
    rgb_w = load(pretrainedPath("vKITTI_GRU_RGB.jld"),"w")
    d_w = load(pretrainedPath("vKITTI_GRU_D.jld"),"w")
    w[1:24] = rgb_w[1:24]
    w[25:48] = d_w[1:24]
    return w
end

function vKITTI_GRU_RGBD_loss(w,h,x,y,t,D)
    total = 0.0
    for i in max(t-D,1):t
        ypred = vKITTI_GRU_RGBD(w,h,KnetArray(x[:,:,:,i:i]))
        total += pixelwiseSoftloss(Array(ypred),y[:,:,:,i:i])
    end
    return total
end