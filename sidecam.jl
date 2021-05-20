using Pipe, Images, PyPlot, Statistics, LinearAlgebra


norm(x) = @pipe x .- minimum(x) |> _ ./ maximum(_)

"Finds the boundaries"
iml(z, λ, dim) = @pipe (z 
  |> (_ .< λ) 
  |> sum(_, dims=dim) 
  |> vec 
  |> norm
  |> findall(x -> x > 0, _) 
  |> extrema
)

"cuts an image over a dimension"
imc(z, dim, idx) = (dim == 1) ? z[:, idx[1]:idx[2]] : z[idx[1]:idx[2], :]

"Cuts an image "
function imcut(z, λ)
  z2 = @pipe iml(z, λ, 1) |> imc(z, 1, _)
  z3 = @pipe iml(z2, λ, 2) |> imc(z2, 2, _)
  z3
end

"import an image"
imtovec(dir, fname) = @pipe (dir*fname 
  |> load(_) 
  |> Float64.(_) 
  |> imcut(_, 0.5) 
  |> imresize(_, (40, 60))
  |> _[1:20, :]
  |> vec
  |> _ / std(_)
)

# MAIN
# ==========================================
no_img, no_edges = 4, 4
xdir, ydir = "train2/", "test2/"

# Creating a training data set: z and the feature set: x and the eigenvectors: v
fns = readdir(xdir)
z = @pipe readdir(xdir) |> map(x -> imtovec(xdir, x), _) |> hcat(_...)
v = @pipe cov(z) |> eigen(_).vectors[:, 1:end-1] 
x = cov(z)*v

# Creating the test feature set: y
fn = "4_03.png"
y = @pipe imtovec(ydir, fn) |> cov(_, z) |> _*v

# evaulating the best match
δ = @pipe (x .- y).^2 |> sum(_, dims=2) |> vec
println("δ: ", δ)
println("Best hit at image: ", fns[argmin(δ)])

# plotting
#=
figure("train")
for i∈1:size(z, 2)
  subplot(no_img/2, 2, i)
  imshow(reshape(z[:, i], :, 60), cmap="gray")
end
figure("train cx*v")
for i∈1:size(x, 1)
  subplot(no_img/2, 2, i)
  plot(x[i, 1:end], "-o", label=string(i))
  grid(true)
end
legend()

figure("test")
imshow(reshape(zt, : , 60), cmap="gray")

figure("test ct*v")
plot(y[1:end], "-o", color="C1")
grid(true)
=#

#=
figure()
z2 = z*v
for i∈1:size(cz, 1)
  subplot(5, 2, i)
  imshow(reshape(z2[:, i], :, 60), cmap="gray")
end
=#

#=
figure()
x = cz*v
for i∈1:size(x, 1)
  subplot(5, 2, i)
  plot(x[i, :], "-o", label=string(i))
  grid(true)
end
legend()
show()
=#

#zt = @pipe load("test/smallup.png") |> Gray.(_) |> save("test/smallup_gray.png")
