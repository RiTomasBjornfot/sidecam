using Pipe, Images, Statistics, LinearAlgebra, DelimitedFiles, Plots
import JLD
"Sets all values between 0 and 1"
norm(x) = @pipe x .- minimum(x) |> _ ./ maximum(_)

"Finds the boundaries in the image"
iml(z, λ, dim) = @pipe (z 
  |> norm
  |> (_ .< λ) 
  |> sum(_, dims=dim) 
  |> vec 
  |> findall(x -> x > 0, _) 
  |> extrema
)

"cuts an image over a dimension"
imc(z, dim, idx) = (dim == 1) ? z[:, idx[1]:idx[2]] : z[idx[1]:idx[2], :]

"Cuts an image"
function imcut(z, λ)
  z2 = @pipe iml(z, λ, 1) |> imc(z, 1, _)
  @pipe iml(z2, λ, 2) |> imc(z2, 2, _)
end

"import an image"
imtovec(dir, fname) = @pipe (dir*fname 
  |> load(_) 
  |> Gray.(_)
  |> Float64.(_) 
  |> imcut(_, 0.98) 
  |> imresize(_, (60, 60))
  |> _[1:25, :]
  #|> imfilter(_, Kernel.gaussian(2))
  |> vec
  #|> _ / std(_)
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
