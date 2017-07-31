function Robot_Conv_RGBD(w,x)
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
    #Scores
    x3 = x1.+x2
    x3 = relu(conv4(w[49],x3).+w[50])
    x3 = deconv4(w[51],x3;padding=1,stride=6).+w[52]
end

function Robot_Conv_RGBD_w()
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
        xavier(Float32,1,1,512,5),      zeros(Float32,1,1,5,1),
        #Deconv
        bilinear(Float32,32,32,5,5),   zeros(Float32,1,1,5,1),
    ]
    rgb_w = load(pretrainedPath("Robot_Conv_RGB.jld"),"w")
    d_w = load(pretrainedPath("Robot_Conv_D.jld"),"w")
    w[1:24] = rgb_w[1:24]
    w[25:48] = d_w[1:24]
    return w
end

function Robot_Conv_RGBD_loss(w,x,y)
    pixelwiseSoftloss(Array(Robot_Conv_RGBD(w,x)),y)
end