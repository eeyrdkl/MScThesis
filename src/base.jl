function printUpdate(o...)
    if isinteractive()
        print("\u1b[1G")
        print(o...)
        print("\u1b[K")
    end
end

function printlnFlush(o...)
    println(o...)
    if !isinteractive()
        flush(STDOUT)
    end
end

function saveNetwork(weights,filename)
    w = map(Array,weights)
	save(pretrainedPath("$(filename).jld"),"w",w,compress=true)
end

function loadNetwork(filename)
    w = load(pretrainedPath(filename),"w")
    w = map(KnetArray,w)
end

function pixelwiseSoftloss(ypred,y)
    -sum(y.*logp(ypred,3)) / (size(y,1)*size(y,2)*size(y,4))
end

function pixelwiseAccuracy(ypred,y)
    ncorrect = sum(y.*(ypred.==maximum(ypred,3)))
    ncount = size(ypred,1)*size(ypred,2)*size(ypred,4)
    return ncorrect,ncount
end