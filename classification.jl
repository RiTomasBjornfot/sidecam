using Pipe, Images, Plots, JLD
dir = "data/bc/"
fname = readdir(dir)
p = @pipe [load.(dir*f) for f∈fname] |> plot.(_);
x = 0
t = [
  1 1 0 0 1
  0 1 0 1 1
  0 1 1 1 1
  0 1 0 0 1
  1 1 1 1 1
  0 1 0 0 1
  0 1 0 1 1
  0 1 1 1 1
  1 0 0 1 1
  0 0 1 0 0
  1 1 0 1 1
  1 1 1 1 1
  1 1 0 1 0
  1 0 1 0 1
  0 1 1 0 1
  1 1 1 0 0
]
tt = @pipe t |> _' |> vec

JLD.save("targets.jld", "fname", fname, "targets", tt)  
