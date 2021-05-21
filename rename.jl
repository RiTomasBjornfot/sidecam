for fn∈readdir()
  fn[end-4:end] != ".jpeg" && continue
  println(fn)
  mv(fn, fn[1:end-4]*".png")
end
