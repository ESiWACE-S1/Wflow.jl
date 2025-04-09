using Metal
using KernelAbstractions
using Parameters
using Adapt
using BenchmarkTools

N = 1000000
a_cpu = rand(Float32, N)
a = MtlArray(a_cpu)
b_cpu = rand(Float32, N)
b = MtlArray(b_cpu)
c_cpu = zeros(Float32, N)
c = MtlArray(c_cpu)

@kernel function vadd!(c, @Const(a), @Const(b))
  i = @index(Global)
  @inbounds c[i] = a[i] + b[i]
end

cpu_backend = KernelAbstractions.get_backend(a_cpu)
gpu_backend = KernelAbstractions.get_backend(a)

# Call with vadd!(backend, 64)(c, a, b)

struct ShallowWaterRiver{T}
  q :: T
  h :: T
  zb :: T
end

function Adapt.adapt_structure(to, from::ShallowWaterRiver)
  q = adapt(to, from.q)
  h = adapt(to, from.h)
  zb = adapt(to, from.zb)
  ShallowWaterRiver(q, h, zb)
end

@inline function inner_func(dt, q, h)
  return dt * sin(q) / (exp(h)^2 + 0.01f0)
end

@inline function inner_func_struct(i, sw, dt)
  maxof = max(sw.q[i], sw.h[i])
  return dt * maxof * sin(sw.q[i]) / (exp(sw.h[i])^2 + 0.01f0)
end

@kernel function test_update(sw, dt)
  i = @index(Global)
  #@inbounds sw.zb[i] = inner_func(dt, sw.q[i], sw.h[i])
  @inbounds sw.zb[i] = inner_func_struct(i, sw, dt)
end

function update(backend)
  if backend == cpu_backend
    sw = ShallowWaterRiver(a_cpu, b_cpu, c_cpu)
  else
    sw = ShallowWaterRiver(a, b, c)
  end
  dt = Float32(24.0*3600)
  dt_in = Float32(3600.0)
  t = 0 
  while t < dt
    test_update(backend, 64)(sw, dt, ndrange = size(sw.q))
    t += dt_in
  end
  KernelAbstractions.synchronize(backend)
end
