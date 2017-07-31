function DAVIS_Conv(w,x)
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

    x = relu(conv4(w[25],x).+w[26])
    x = deconv4(w[27],x;padding=1,stride=6).+w[28]
end

function DAVIS_Conv_w()
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
        xavier(Float32,1,1,512,2),       zeros(Float32,1,1,2,1),
        #Deconv
        bilinear(Float32,32,32,2,2),     zeros(Float32,1,1,2,1),
    ]
    w[1:24] = load(pretrainedPath("VGG19.jld"),"w")[1:24]
    return w
end

function DAVIS_Conv_loss(w,x,y)
    pixelwiseSoftloss(Array(DAVIS_Conv(w,x)),y)
end