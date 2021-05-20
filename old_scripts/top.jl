using Pipe, FileIO, PyPlot, Images
using Statistics, CircleFit

"Normalizes a matrix"
norm(x) = @pipe x .- minimum(x) |> _./maximum(_)

"Calculates circle coordinates" 
circXY(x0, y0, r0) = @pipe (0:0.05:2*π 
  |> (r0*cos.(_) .+ x0, r0*sin.(_) .+ y0)
 ) 

"Converts an image to a binary image"
toBim(z, th) = @pipe (z
  |> imedge 
  |> norm(_[3])
  |> (_ .> th)

)

"Takes a sub matrix from a matrix z of size 2*sz+1"
function submat(z, i, sz) 
  mat = z[i[1]-sz:i[1]+sz, i[2]-sz:i[2]+sz]
  mat[sz+1, sz+1] = 0
  mat
end

"Adds a frame to an matrix"
function frame(z, sz)
  x = zeros(size(z) .+ 2*sz)
  x[sz+1:end-sz, sz+1:end-sz] = z
  x
end

"Finds a connected region of ones in image z, starting at index i"
function region(z, i, sz)
  ii, δ = [i], CartesianIndex{}(sz+1, sz+1)
  z[i] = 0
  for n∈1:Int(1e9)
    n > length(ii) && break
    mat = submat(z, ii[n], sz)
    for m in findall(x -> x == 1, mat)
      j = ii[n] + m - δ
      z[j] = 0
      push!(ii, j)
    end
  end
  unique(ii)
end

"Finds all connected regions in an image."
function regions(z, sz)
  idx = []
  for n∈1:100
    si = findfirst(x -> x == 1, z)
    si == nothing && break 
    push!(idx, region(z, si, sz)) 
  end
  idx
end

"Converts an array of tuples tuples to array of array x and y."
function toXY(z)
  x, y = [], []
  for i∈z
    append!(x, i[2])   
    append!(y, i[1])   
  end
  x, y
end

"Calculate the mean error between data and a circle"
function circleError(data, c)
  x, y = data[1], data[2]
  x0, y0, r0 = c[1], c[2], c[3]
  @pipe ((x .- x0).^2 .+ (y .- y0).^2 
    |> sqrt.(_) |> _ .- r0 
    |> abs.(_) 
    |> mean 
    |> _/r0 
  )
end
"Calculates the best circle fit"
function  bestfit(idx)
  Ε, x = [], []
  for pts∈toXY.(idx)
    c = circfit(pts...)
    ϵ = circleError(pts, c)
    length(pts[1]) < 200 && continue # too few points
    isnan(c[1]) && continue # is not a number
    ϵ > 0.01 && continue # accuracy over ...
    c[3] < 200 && continue # min radius
    #push!(x, circXY(c...))
    push!(x, c)
    push!(Ε, ϵ)
  end
  x[argmax(Ε)]
end

""
function polarImage(x, c)
  x0, y0, r0 = c[1], c[2], c[3]
  rows = Int(ceil(y0-r0)):Int(floor(y0+r0))
  cols = Int(ceil(x0-r0)):Int(floor(x0+r0))

  x, r, θ, v = x[rows, cols], [], [], []
  for row=1:size(x, 1), col=1:size(x, 2)
    append!(r, sqrt((col-r0)^2 + (row-r0)^2) |> round |> Int)
    append!(θ, angle((col-r0) + (row-r0)*im) |> rad2deg |> round |> Int) 
    append!(v, x[row, col])
  end

  X = zeros(361, maximum(r) + 1)
  for i∈1:length(r)
    X[θ[i]+181, r[i]+1] = v[i]
  end
  X[:, 1:(r0+10 |> round |> Int)]
end

# MAIN
# ===================================
fname = "image.png"
img = @pipe load(fname) |> imresize(_, ratio=8/10)
bim = @pipe toBim(img, 0.5) |> frame(_, 3)

idx = regions(copy(bim), 3)
c = bestfit(idx)
x, y = circXY(c...)
pim = polarImage(Float64.(img), c)

# plotting
figure("img")
imshow(Float64.(img), cmap="gray")
figure("bim")
imshow(bim, cmap="gray")
plot(x .- 1, y .- 1)
figure("Polar")
imshow(pim, cmap="gray")
show()

