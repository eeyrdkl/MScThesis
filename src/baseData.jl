function loadData_vKITTI(worlds,variations,model,mode)
    colors = 0; x = 0;
    y = zeros(Float32,224,224,13,1)
    if model in ("RGB_VGG",)
        #RGB
        x = zeros(Float32,224,224,3,1)
        for world in worlds
            for variation in variations
                worldfile = load(joinpath(dataDIR,"vKITTI","world_$(world)_$(variation).jld"))
                if mode == "odd"
                    x = cat(4,x,worldfile["RGB"][:,:,:,1:2:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,1:2:end])
                elseif mode == "even"
                    x = cat(4,x,worldfile["RGB"][:,:,:,2:2:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,2:2:end])
                elseif mode == "firstHalf"
                    x = cat(4,x,worldfile["RGB"][:,:,:,1:div(end,2)])
                    y = cat(4,y,worldfile["GT"][:,:,:,1:div(end,2)])
                elseif mode == "secondHalf"
                    x = cat(4,x,worldfile["RGB"][:,:,:,div(end,2)+1:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,div(end,2)+1:end])
                elseif mode == "full"
                    x = cat(4,x,worldfile["RGB"][:,:,:,:])
                    y = cat(4,y,worldfile["GT"][:,:,:,:])
                end
                colors = worldfile["colors"]
            end
        end
        x = x[:,:,:,2:end]
        y = y[:,:,:,2:end]
    elseif model in ("RGBD_VGG",)
        #RGB + Colorized depth
        x = zeros(Float32,224,224,3,1)
        d = zeros(Float32,224,224,3,1)
        for world in worlds
            for variation in variations
                worldfile = load(joinpath(dataDIR,"vKITTI","world_$(world)_$(variation).jld"))
                if mode == "odd"
                    x = cat(4,x,worldfile["RGB"][:,:,:,1:2:end])
                    d = cat(4,d,worldfile["DepthRGB"][:,:,:,1:2:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,1:2:end])
                elseif mode == "even"
                    x = cat(4,x,worldfile["RGB"][:,:,:,2:2:end])
                    d = cat(4,d,worldfile["DepthRGB"][:,:,:,2:2:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,2:2:end])
                elseif mode == "firstHalf"
                    x = cat(4,x,worldfile["RGB"][:,:,:,1:div(end,2)])
                    d = cat(4,d,worldfile["DepthRGB"][:,:,:,1:div(end,2)])
                    y = cat(4,y,worldfile["GT"][:,:,:,1:div(end,2)])
                elseif mode == "secondHalf"
                    x = cat(4,x,worldfile["RGB"][:,:,:,div(end,2)+1:end])
                    d = cat(4,d,worldfile["DepthRGB"][:,:,:,div(end,2)+1:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,div(end,2)+1:end])
                elseif mode == "full"
                    x = cat(4,x,worldfile["RGB"][:,:,:,:])
                    d = cat(4,d,worldfile["DepthRGB"][:,:,:,:])
                    y = cat(4,y,worldfile["GT"][:,:,:,:])
                end
                colors = worldfile["colors"]
            end
        end
        x = x[:,:,:,2:end]
        d = d[:,:,:,2:end]
        y = y[:,:,:,2:end]
        x = cat(3,x,d)
    elseif model in ("D_VGG",)
        #Colorized depth
        x = zeros(Float32,224,224,3,1)
        for world in worlds
            for variation in variations
                worldfile = load(joinpath(dataDIR,"vKITTI","world_$(world)_$(variation).jld"))
                if mode == "odd"
                    x = cat(4,x,worldfile["DepthRGB"][:,:,:,1:2:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,1:2:end])
                elseif mode == "even"
                    x = cat(4,x,worldfile["DepthRGB"][:,:,:,2:2:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,2:2:end])
                elseif mode == "firstHalf"
                    x = cat(4,x,worldfile["DepthRGB"][:,:,:,1:div(end,2)])
                    y = cat(4,y,worldfile["GT"][:,:,:,1:div(end,2)])
                elseif mode == "secondHalf"
                    x = cat(4,x,worldfile["DepthRGB"][:,:,:,div(end,2)+1:end])
                    y = cat(4,y,worldfile["GT"][:,:,:,div(end,2)+1:end])
                elseif mode == "full"
                    x = cat(4,x,worldfile["DepthRGB"][:,:,:,:])
                    y = cat(4,y,worldfile["GT"][:,:,:,:])
                end
                colors = worldfile["colors"]
            end
        end
        x = x[:,:,:,2:end]
        y = y[:,:,:,2:end]
    end
    return x,y,colors
end

function loadData_DAVIS()
    videoDIR = joinpath(dataDIR,"DAVIS")
    #Train
    xtrain = zeros(Float32,224,224,3,1)
    ytrain = zeros(Float32,224,224,2,1)
    #Test
    xtest = zeros(Float32,224,224,3,1)
    ytest = zeros(Float32,224,224,2,1)
    for video in readdir(videoDIR)
        datafile = load(joinpath(videoDIR,video))

        xtrain = cat(4,xtrain,datafile["x"][:,:,:,1:div(end,2)])
        ytrain = cat(4,ytrain,datafile["y"][:,:,:,1:div(end,2)])

        xtest = cat(4,xtest,datafile["x"][:,:,:,div(end,2)+1:end])
        ytest = cat(4,ytest,datafile["y"][:,:,:,div(end,2)+1:end])
    end
    xtrain = xtrain[:,:,:,2:end]
    ytrain = ytrain[:,:,:,2:end]

    xtest = xtest[:,:,:,2:end]
    ytest = ytest[:,:,:,2:end]

    return xtrain,ytrain,xtest,ytest
end

function loadData_Robot(model,mode)
    ddir = joinpath(dataDIR,"Robot")
    rooms = filter!(r".jld", readdir(ddir))

    x = 0;
    y = zeros(Float32,224,224,5,1)
    if model in ("RGB",)
        #RGB
        x = zeros(Float32,224,224,3,1)
        for room in rooms
            roomfile = load(joinpath(ddir,room))
            seqparts = []
            seqn = size(roomfile["GT"],4)
            if seqn % 4 == 0
                sz = Int(floor(seqn/4))
                for i=1:4
                    push!(seqparts,(i-1)*sz+1:i*sz)
                end
            else
                sz = Int(floor(seqn/4))
                for i=1:4
                    if i!=4
                        push!(seqparts,(i-1)*sz+1:i*sz)
                    else
                        push!(seqparts,(i-1)*sz+1:seqn)
                    end
                end
            end
            if mode == "train"
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[1]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[1]])
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[2]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[2]])
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[4]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[4]])
            elseif mode == "test"
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[3]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[3]])
            end
        end
        x = x[:,:,:,2:end]
        y = y[:,:,:,2:end]
    elseif model in ("RGBD",)
        #RGB + Colorized depth
        x = zeros(Float32,224,224,3,1)
        d = zeros(Float32,224,224,3,1)
        for room in rooms
            roomfile = load(joinpath(ddir,room))
            seqparts = []
            seqn = size(roomfile["GT"],4)
            if seqn % 4 == 0
                sz = Int(floor(seqn/4))
                for i=1:4
                    push!(seqparts,(i-1)*sz+1:i*sz)
                end
            else
                sz = Int(floor(seqn/4))
                for i=1:4
                    if i!=4
                        push!(seqparts,(i-1)*sz+1:i*sz)
                    else
                        push!(seqparts,(i-1)*sz+1:seqn)
                    end
                end
            end
            if mode == "train"
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[1]])
                d = cat(4,d,roomfile["DepthRGB"][:,:,:,seqparts[1]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[1]])
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[2]])
                d = cat(4,d,roomfile["DepthRGB"][:,:,:,seqparts[2]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[2]])
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[4]])
                d = cat(4,d,roomfile["DepthRGB"][:,:,:,seqparts[4]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[4]])
            elseif mode == "test"
                x = cat(4,x,roomfile["RGB"][:,:,:,seqparts[3]])
                d = cat(4,d,roomfile["DepthRGB"][:,:,:,seqparts[3]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[3]])
            end
        end
        x = x[:,:,:,2:end]
        d = d[:,:,:,2:end]
        y = y[:,:,:,2:end]
        x = cat(3,x,d)
    elseif model in ("D",)
        #Colorized depth
        x = zeros(Float32,224,224,3,1)
        for room in rooms
            roomfile = load(joinpath(ddir,room))
            seqparts = []
            seqn = size(roomfile["GT"],4)
            if seqn % 4 == 0
                sz = Int(floor(seqn/4))
                for i=1:4
                    push!(seqparts,(i-1)*sz+1:i*sz)
                end
            else
                sz = Int(floor(seqn/4))
                for i=1:4
                    if i!=4
                        push!(seqparts,(i-1)*sz+1:i*sz)
                    else
                        push!(seqparts,(i-1)*sz+1:seqn)
                    end
                end
            end
            if mode == "train"
                x = cat(4,x,roomfile["DepthRGB"][:,:,:,seqparts[1]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[1]])
                x = cat(4,x,roomfile["DepthRGB"][:,:,:,seqparts[2]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[2]])
                x = cat(4,x,roomfile["DepthRGB"][:,:,:,seqparts[4]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[4]])
            elseif mode == "test"
                x = cat(4,x,roomfile["DepthRGB"][:,:,:,seqparts[3]])
                y = cat(4,y,roomfile["GT"][:,:,:,seqparts[3]])
            end
        end
        x = x[:,:,:,2:end]
        y = y[:,:,:,2:end]
    end
    return x,y
end

function savePrediction(ypred,colors,filename)
    im = zeros(Float32,size(ypred,1),size(ypred,2),3)
    for wi=1:size(ypred,1)
        for hi=1:size(ypred,2)
            im[wi,hi,:] = colors[indmax(ypred[wi,hi,:])]
        end
    end
    im = convert(Image{RGB},im)
    im["spatialorder"] = ["x","y"]

    t = splitdir(filename)
    if !isdir(t[1])
        mkpath(t[1])
    end

    Images.save(filename,im)
end