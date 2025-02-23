using Distributions
using QuadGK
using Printf

"""

    function pCond(M::Float64, N::Int64, t::Float64, a::Float64, h::Float64)

"""
function pCond(M::Float64, N::Int64, t::Float64, a::Float64, h::Float64)
                               
  q = 1.0 - exp(-h * t)

  d = Normal(0, 1)
  x = quantile(d, q)
  q_m = cdf(d, (x - a * M)/sqrt(1 - a * a))
  
  K = N + 1
  p = zeros(K, K)
  p_cond = zeros(K)
  
  p[1,1] = 1.0
  for k in 2:K
    p[k,1] = p[k-1,1] * (1.0 - q_m)
    for l in 2:K
      p[k,l] = p[k-1,l]*(1.0 - q_m) + p[k-1,l-1] * q_m
    end 
  end
  p[K-1,K-1] = p[K-2,K-2] * q_m

  p_cond = p[K-1,:];
  
  return p_cond  
end



"""
    function f(x::Float64, N::Int64, t::Float64, a::Float64, h::FLoat64,
            l::Int64)
"""
function f(x::Float64, N::Int64, t::Float64, a::Float64, h::Float64,
        l::Int64)

    p_cond = pCond(x, N, t, a, h)
    y = p_cond[l] * 1.0/√(2.0 * π) * exp(-0.5*x^2)

    return y 
end


"""
    function probDist(N::Int64, t::Float64, a::Float64, h::Float64)

"""
function probDist(N::Int64, t::Float64, a::Float64, h::Float64)

    p_uncond = zeros(N+1)
    for l in 1:N+1
        p_uncond[l], _ = quadgk(x -> f(x, N, t, a, h, l), -Inf, Inf)
    end
    
    return p_uncond
end


N = 100
A = 1.0e+07
R = 0.40
T = 5
num_periods = T * 4

a = sqrt(0.75)
single_name_spread = 98/10000
h = single_name_spread/(1-R)
dt = (3/12)
r_f = 0.0425

attach = 0.000
detach = 0.025
H = detach * (A * N)
L = attach * (A * N)

d = exp.(-r_f * dt .* (1:num_periods))

el = zeros(num_periods)
ufee = zeros(num_periods)

p_dist = zeros(N+1, num_periods)
for i in 1:num_periods
    tper = i * dt
    p_dist[:, i] = probDist(N, tper, a, h)
    loss = 0.0
    for l in 1:N+1
        loss += p_dist[l, i] * max(min((l-1) * A * (1-R), H) - L, 0)
    end
    el[i] = loss
    ufee[i] = (d[i] * dt * ((H - L) - el[i]))

    println("i: $i, tper: $tper, el: $(el[i]), ufee: $(ufee[i])")
end

contingent = zeros(num_periods)
contingent[1] = d[1] * el[1] 
for i in 2:num_periods 
    contingent[i] = d[i] * (el[i] - el[i-1])
end
total_contingent = sum(contingent)
spar = total_contingent / sum(ufee)
println("Par Spread on $(100*attach)% to $(100*detach)% tranche: $(10_000*spar) bps")

s = 315/10000
fee = s * sum(ufee)
mtm = fee - total_contingent
