using Distributions
using QuadGK
using Dates
using OrderedCollections


"""

    function pCond!(p::Matrix{Float64}, M::Float64, N::Int64, a::Float64, q::Float64)

"""
function pCond!(p::Matrix{Float64}, M::Float64, N::Int64, a::Float64, q::Float64)
                               
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
function f(x::Float64, N::Int64, a::Float64, q::Float64, l::Int64, p::Matrix{Float64})

    pCond!(p, x, N, a, q)
    y = p[N, l] * 1.0/√(2.0 * π) * exp(-0.5*x^2)

    return y 
end


"""
    function vrad_integration(N::Int64, a::Float64, q::Float64, l::Int64, 
        buf)
"""
function vrad_integration(N::Int64, a::Float64, q::Float64, l::Int64, p::Matrix{Float64}, buf)

    val = quadgk(x -> f(x, N, a, q, l, p), -Inf, Inf; segbuf=buf)[1]
    return val
end


"""
    function probDist!(N::Int64, a::Float64, q::Float64, p::Matrix{Float64},
        p_uncond::Vector{Float64}, buf)

"""
function probDist!(N::Int64, a::Float64, q::Float64, p::Matrix{Float64},
    p_uncond::Vector{Float64}, buf)

    for l in 1:N+1
        p_uncond[l] = vrad_integration(N, a, q, l, p, buf)
    end
    
    return p_uncond
end


"""
    function create_schedule(A::Float64, T::Int64, cpr::Float64)
"""
function create_schedule(A::Float64, T::Int64, cpr::Float64; period::Int64=12)
    smm = 1 - (1-cpr/100)^(1/period)
    B = zeros(T)
    B[1] = A
    for t in 2:T
        B[t] = B[t-1] * (1 - smm)
    end
    return B 
end


"""
    function tranche_valuation(attach::Float64, detach::Float64, ρ::Float64, 
        single_name_spread::Float64, tranche_spread::Float64, first_pay_date::Date, last_pay_date::Date; 
        N::Int64=100, A::Float64=10.0e+06, R::Float64=0.40,  periods::Int64=12, r_f::Float64=0.0425)

"""
function tranche_valuation(attach::Float64, detach::Float64, ρ::Float64, 
    single_name_spread::Float64, tranche_spread::Float64, first_pay_date::Date, last_pay_date::Date; 
    N::Int64=100, A::Float64=10.0e+06, R::Float64=0.40,  periods::Int64=12, r_f::Float64=0.0425)

    pmt_dates = collect(first_pay_date:Month(1):last_pay_date)
    T = length(pmt_dates)

    a = sqrt(ρ)
    h = single_name_spread/(1-R)
    dt = 1/periods

    H = detach * (A * N)
    L = attach * (A * N)

    d = exp.(-r_f * dt .* (1:T))

    el = zeros(T)
    ufee = zeros(T)

    buf = alloc_segbuf()
    p = zeros(Float64, N+1, N+1)
    p_uncond = zeros(Float64, N+1)

    p_dist = zeros(N+1, T)

    B = create_schedule(A, T, 25.0)

    for i in 1:T
        t = i * dt
        q = 1.0 - exp(-h * t)
        p_dist[:, i] = probDist!(N, a, q, p, p_uncond, buf)
        per_loan_loss = B[i] * (1 - R)
        loss = 0.0
        for l in 1:N+1
            loss += p_dist[l, i] * max(min((l-1) * per_loan_loss, H) - L, 0)
        end
        el[i] = loss
        ufee[i] = (d[i] * dt * ((H - L) - el[i]))
    
        # println("i: $i, t: $t, el: $(el[i]), ufee: $(ufee[i])")
    end
    
    contingent = zeros(T)
    contingent[1] = d[1] * el[1] 
    for i in 2:T 
        contingent[i] = d[i] * (el[i] - el[i-1])
    end
    total_contingent = sum(contingent)
    spar = total_contingent / sum(ufee)
    println("Par Spread on $(100*attach)% to $(100*detach)% tranche: $(10_000*spar) bps")
    
    fee = tranche_spread * sum(ufee)
    mtm = fee - total_contingent

    result = OrderedDict("Par Spread" => spar * 10_000, "MtM" => mtm)
    return result
end






