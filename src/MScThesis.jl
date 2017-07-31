__precompile__()
module MScThesis
    info("MSc Thesis Implementation by Ekrem Emre Yurdakul")

    using Knet, AutoGrad, JLD, Images, Colors

    dataDIR = joinpath(dirname(dirname(@__FILE__)),"data")
    pretrainedPath(name) = joinpath(dataDIR,"pretrained",name)
    outputPath(name) = joinpath(dataDIR,"output",name)
    worlds=["0001","0002","0006","0018","0020"]

    include("base.jl");
    include("baseData.jl");

    include("DAVIS/Conv.jl");
    include("DAVIS/LSTM.jl");
    include("DAVIS/Test.jl");
    include("DAVIS/Train.jl");

    include("Robot/Conv_D.jl");
    include("Robot/Conv_RGB.jl");
    include("Robot/Conv_RGBD.jl");
    include("Robot/LSTM_D.jl");
    include("Robot/LSTM_RGB.jl");
    include("Robot/LSTM_RGBD.jl");
    include("Robot/Test.jl");
    include("Robot/Train.jl");

    include("vKITTI/Conv_D.jl");
    include("vKITTI/Conv_RGB.jl");
    include("vKITTI/Conv_RGBD.jl");
    include("vKITTI/GRU_D.jl");
    include("vKITTI/GRU_RGB.jl");
    include("vKITTI/GRU_RGBD.jl");
    include("vKITTI/LSTM_D.jl");
    include("vKITTI/LSTM_RGB.jl");
    include("vKITTI/LSTM_RGBD.jl");
    include("vKITTI/RNN_D.jl");
    include("vKITTI/RNN_RGB.jl");
    include("vKITTI/RNN_RGBD.jl");
    include("vKITTI/Test.jl");
    include("vKITTI/Train.jl");

    include("timing.jl");
end