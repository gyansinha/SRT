using Distributions
using QuadGK
using Dates
using Distributed
using BenchmarkTools


"""

    function pCond(M::Float64, N::Int64, t::Float64, a::Float64, h::Float64)

"""
function pCond!(p::Matrix{Float64}, M::Float64, N::Int64, t::Float64, a::Float64, h::Float64)
                               
  q = 1.0 - exp(-h * t)

  d = Normal()
  x = quantile(d, q)
  q_m = cdf(d, (x - a * M)/sqrt(1 - a * a))
  
  K::Int64 = N + 1
  
  p[1,1] = 1.0
  for k in 2:K
    p[k,1] = p[k-1,1] * (1.0 - q_m)
    for l in 2:K
      p[k,l] = p[k-1,l]*(1.0 - q_m) + p[k-1,l-1] * q_m
    end 
  end;

  p[K-1,K-1] = p[K-2,K-2] * q_m
  return p
end


"""
    function f(x::Float64, N::Int64, t::Float64, a::Float64, h::FLoat64, l::Int64, p::Matrix{Float64})
"""
function f(x::Float64, N::Int64, t::Float64, a::Float64, h::Float64, l::Int64, p::Matrix{Float64})

    pCond!(p, x, N, t, a, h)
    y = p[N, l] * 1.0/√(2.0 * π) * exp(-0.5*x^2)

    return y 
end


"""
    function vrad_integration(N::Int64, t::Float64, a::Float64, h::Float64, l::Int64, buf)
"""
function vrad_integration(N::Int64, t::Float64, a::Float64, h::Float64, l::Int64, p::Matrix{Float64}, 
    buf)

    val = quadgk(x -> f(x, N, t, a, h, l, p), -Inf, Inf; segbuf=buf)[1]
    return val
end


"""
    function probDist(N::Int64, t::Float64, a::Float64, h::Float64)

"""
function probDist!(N::Int64, t::Float64, a::Float64, h::Float64, p::Matrix{Float64},
    p_uncond::Vector{Float64}, buf)

    for l in 1:N+1
        p_uncond[l] = vrad_integration(N, t, a, h, l, p, buf)
    end
    
    return p_uncond
end


"""
    function create_schedule(A::Float64, T::Int64, cpr::Float64)
"""
function create_schedule(A::Float64, T::Int64, cpr::Float64)
    smm = 1 - (1-cpr/100)^(1/12)
    B = zeros(T)
    B[1] = A
    for t in 2:T
        B[t] = B[t-1] * (1 - smm)
    end
    return B 
end


N = 100
A = 1.0e+07
R = 0.40

first_pay_date = Date("2025-04-25")
last_pay_date = Date("2032-02-25")
pmt_dates = collect(first_pay_date:Month(1):last_pay_date)
num_periods = length(pmt_dates)

ρ = 0.90
a = sqrt(ρ)
single_name_spread = 98/10_000
h = single_name_spread/(1-R)
dt = (3/12)
r_f = 0.0425

attach = 0.0475
detach = 0.1250
H = detach * (A * N)
L = attach * (A * N)

d = exp.(-r_f * dt .* (1:num_periods))

el = zeros(num_periods)
ufee = zeros(num_periods)

buf = alloc_segbuf()
p = zeros(Float64, N+1, N+1)
p_uncond = zeros(Float64, N+1)

p_dist = zeros(N+1, num_periods)

B = create_schedule(A, num_periods, 25.0)

for i in 1:num_periods
    tper = i * dt
    p_dist[:, i] = probDist!(N, tper, a, h, p, p_uncond, buf)
    per_loan_loss = B[i] * (1 - R)
    loss = 0.0
    for l in 1:N+1
        loss += p_dist[l, i] * max(min((l-1) * per_loan_loss, H) - L, 0)
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
