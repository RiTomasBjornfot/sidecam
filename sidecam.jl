using Pipe, Images, Statistics, LinearAlgebra, DelimitedFiles, Plots
include("crop.jl")
import JLD
"Sets all values between 0 and 1"
norm(x) = @pipe x .- minimum(x) |> _ ./ maximum(_)

"loads an image from file"
loadim(path) = @pipe load(path) |> Gray.(_) |> Float64.(_) 

"import an image"
imtovec(dir, fname) = @pipe (dir*fname 
  |> loadim
  |> Crop.crop(_, (0, 0.9), 10) 
  |> imresize(_, (60, 60))
  |> _[1:25, :]
  |> imfilter(_, Kernel.gaussian(1))
  |> vec
  |> norm
)

# MAIN
# ==========================================
indata = JLD.load("targets.jld")
target = indata["targets"] 
fname = indata["fname"]

# Loads the images as columns in z
xdir = "data/bc/"
z = @pipe fname |> map(x -> imtovec(xdir, x), _) |> hcat(_...)

# Creating a training data set: z and the feature set: x and the eigenvectors: v
v = @pipe cov(z) |> eigen(_).vectors |> _[:, 1:end-1] 
x = cov(z)*v

result, δ = [], []
for ti∈1:size(z, 2)
  println("Test index: ", ti)
  
  # Creating the test feature set: y
  z_test = z[:, ti]
  y = @pipe z_test |> cov(_, z) |> _*v

  # evaulating the best match
  d = @pipe (x .- y).^2 |> sum(_, dims=2) |> vec
  j = sortperm(d)[1:6]
  push!(result, target[j] .- target[j[1]] .== 0)  
  push!(δ ,d[j])
  
  # plotting
  p = @pipe (xdir .* fname[j] 
    |> load.(_) 
    |> plot.(_, xticks=false, xaxis=false, yticks=false, yaxis=false) 
    |> plot(_... , layout=(3, 2))
    |> savefig(_, "result/"*string(ti)*".png")
  )
end

result = @pipe hcat(result...) |> _[2:end, :] |> _'
δ = @pipe hcat(δ...) |> _[2:end, :] |> 1e5*_ |> _' |> round.(_)
writedlm("result/result.csv", [fname result δ])
