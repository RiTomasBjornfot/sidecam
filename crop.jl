module Crop

  using Pipe, Images, PyPlot, Statistics

  "Loads an image from the file system."
  loadim(fname) = @pipe load(fname) |> Gray.(_) |> Float64.(_)

  "Makes a binary image from threshold λ"
  bim(x, λ) = (λ[1] .< abs.(x) .< λ[2])
  
  "
  i = boxes(z, λ)
  Finds boxes around areas where pixel values > 0.
    z : Float64 matrix of an image.
    λ : The minimum allowed box width or height.
    i : Two arrays; row and column indecies of regions with pixels > 0.
  "
  function boxes(z, λ) 
    f(z, d, λ) = @pipe (z 
      |> sum(_, dims=d) 
      |> vec 
      |> (_ .≠ 0) 
      |> diff
      |> findall(x -> x == λ, _)
    )
    g(x) = x[2] - x[1]
    # finds all boxes
    x_mn, x_mx = f(z, 1, 1), f(z, 1, -1) 
    y_mn, y_mx = f(z, 2, 1), f(z, 2, -1) 
    # reordering
    x = [(x_mn[i], x_mx[i]) for i∈1:length(x_mn)]
    y = [(y_mn[i], y_mx[i]) for i∈1:length(y_mn)]
    # removing small regions
    filter!(x -> g(x) > λ, x); filter!(x -> g(x) > λ, y)
    x, y
  end

  "Cuts an image according to i"
  cutim(x, i) = x[i[2][1][1]:i[2][1][2], i[1][1][1]:i[1][1][2]]

  "test function"
  function test()
    br = loadim("test/background1.png")
    sp1 = loadim("test/speaker1.png")
    sp2 = loadim("test/speaker2.png")
    sp1c = crop(sp1, br, (0.4, 1.0), 100)
    sp2c = crop(sp2, (0, 0.9), 100)
    figure("test speaker1")
    subplot(211); imshow(sp1, cmap="gray")
    subplot(212); imshow(sp1c, cmap="gray")
    figure("test speaker")
    subplot(121); imshow(sp2, cmap="gray")
    subplot(122); imshow(sp2c, cmap="gray")
  end

  # Use these functions
  # ========================
  "
  Crops an image according to λ and δ.

  crop(z, bg, λ, δ) || crop(z, λ, δ)
    z : The image as a Float64
    bg : The background image (optional)
    λ : The binary thresholds as (min, max) tuple
    δ : The maximum allowed with and height
  "
  crop(z, λ, δ) = @pipe bim(z, λ) |> boxes(_, δ) |> cutim(z, _)
  crop(z, bg, λ, δ) = @pipe bim(z .- bg, λ) |> boxes(_, δ) |> cutim(z, _)
end
