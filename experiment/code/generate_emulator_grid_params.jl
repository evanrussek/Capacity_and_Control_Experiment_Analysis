#!/usr/bin/env julia
# LHS generator for emulator training over (m ∈ {2,3,4}) × (T_pre ∈ {0.30,0.70,1.10}), T_total=1.35
# - ζ→κ per timing so "retargeting strength" is comparable across post-cue durations
# - Meant to be sharded across many SLURM array jobs (no outer threading to avoid nested @threads in expectedRt)

using Random, Statistics, LinearAlgebra
using DataFrames, CSV
using Distributions

# Bring expectedRt into scope
include("model_simulator.jl")

# -----------------------
# Config (env overridable)
# -----------------------
const OUT_DIR        = get(ENV, "OUT_DIR", "./grid_out_m_a_lhs_base")
const REPS           = parse(Int, get(ENV, "REPS", "800"))  # reps per (η,ζ,ν,m,a) point
const SEED_BASE      = parse(Int, get(ENV, "SEED_BASE", "424242"))
const dt             = parse(Float64, get(ENV, "DT", "0.001"))

# Loads and timing
const M_LIST    = [2, 3, 4]
const T_TOTAL   = 1.35
const TPRE_LIST = [0.30, 0.70, 1.10]
const A_LIST    = TPRE_LIST ./ T_TOTAL

# q values you’ll actually use
const Q_SET     = parse.(Float64, split(get(ENV, "Q_SET", "0.70,0.92"), ","))

# Parameter ranges (sample base in (eta, zeta, nu), convert ζ→κ per T_post)
const ETA_LO = parse(Float64, get(ENV, "ETA_LO", "0.01"))
const ETA_HI = parse(Float64, get(ENV, "ETA_HI", "20.0"))
const ZETA_LO = parse(Float64, get(ENV, "ZETA_LO", "0.0001"))
const ZETA_HI = parse(Float64, get(ENV, "ZETA_HI", "0.9999"))
const NU_LO  = parse(Float64, get(ENV, "NU_LO", "0.25"))
const NU_HI  = parse(Float64, get(ENV, "NU_HI", "12.0"))
const ZETA_MIX_FRAC = parse(Float64, get(ENV, "ZETA_MIX_FRAC", "0.3"))  # frac from Beta(0.5,0.5)

# LHS configuration
const N_LHS_POINTS = parse(Int, get(ENV, "N_LHS_POINTS", "30000"))
const N_POINTS     = parse(Int, get(ENV, "N_POINTS", string(N_LHS_POINTS)))
const ADD_CORNERS  = lowercase(get(ENV, "ADD_CORNERS", "true")) == "true"

# Optional: purposefully extreme anchors to guarantee envelope coverage
const ADD_ANCHORS  = lowercase(get(ENV, "ADD_ANCHORS", "true")) == "true"
const ANCHOR_ETA   = parse.(Float64, split(get(ENV, "ANCHOR_ETA", "0.02,0.05,0.10,6.0,10.0,15.0"), ","))
const ANCHOR_ZETA  = parse.(Float64, split(get(ENV, "ANCHOR_ZETA","0.01,0.2,0.8,0.9999"), ","))
const ANCHOR_NU    = parse.(Float64, split(get(ENV, "ANCHOR_NU",  "0.25,0.5,1.0,4.0,8.0,12.0"), ","))

# -----------------------
# Helpers
# -----------------------
@inline loglerp(u, lo, hi) = exp(log(lo) + u * (log(hi) - log(lo)))
@inline kappa_from_zeta(ζ::Float64, Tpost::Float64) = -log(1 - ζ) / Tpost

function lhs(n::Int, d::Int; rng=Random.default_rng())
    U = zeros(n, d)
    for j in 1:d
        perm = randperm(rng, n)
        for i in 1:n
            U[i, j] = (perm[i] - rand(rng)) / n
        end
    end
    U
end

function corner_points_3d()
    pts = Float64[]
    push_triplet(u1,u2,u3) = (push!(pts, u1); push!(pts, u2); push!(pts, u3))
    for u1 in (0.0, 1.0), u2 in (0.0, 1.0), u3 in (0.0, 1.0); push_triplet(u1,u2,u3); end
    for (u1,u2,u3) in ((0.5,0.5,0.0), (0.5,0.5,1.0), (0.5,0.0,0.5),
                       (0.5,1.0,0.5), (0.0,0.5,0.5), (1.0,0.5,0.5)); push_triplet(u1,u2,u3); end
    push_triplet(0.5,0.5,0.5)
    reshape(pts, 3, :)'
end

@inline function u_to_base(u1,u2,u3; rng=Random.default_rng(),
                           eta_lo::Float64=ETA_LO, eta_hi::Float64=ETA_HI,
                           zeta_lo::Float64=ZETA_LO, zeta_hi::Float64=ZETA_HI,
                           nu_lo::Float64=NU_LO,   nu_hi::Float64=NU_HI)
    η = loglerp(u1, eta_lo, eta_hi)
    ν = loglerp(u2, nu_lo,  nu_hi)
    # Mixture for ζ: 70% uniform + 30% Beta(0.5,0.5) within [zeta_lo, zeta_hi]
    ζ = if rand(rng) < ZETA_MIX_FRAC
        zeta_lo + quantile(Beta(0.5, 0.5), rand(rng)) * (zeta_hi - zeta_lo)
    else
        zeta_lo + u3 * (zeta_hi - zeta_lo)
    end
    ζ = clamp(ζ, zeta_lo, zeta_hi)
    (η, ζ, ν)
end

function build_base_points()
    pts = Vector{NTuple{3,Float64}}()
    rng = MersenneTwister(20250909)

    # Base (full range) via LHS
    U = lhs(N_POINTS, 3; rng=rng)
    if ADD_CORNERS
        U = vcat(U, corner_points_3d())
    end
    sizehint!(pts, size(U,1))
    for i in 1:size(U,1)
        push!(pts, u_to_base(U[i,1], U[i,2], U[i,3]; rng=rng,
                             eta_lo=ETA_LO, eta_hi=ETA_HI,
                             zeta_lo=ZETA_LO, zeta_hi=ZETA_HI,
                             nu_lo=NU_LO,   nu_hi=NU_HI))
    end

    if ADD_ANCHORS
        for η in ANCHOR_ETA, ζ in ANCHOR_ZETA, ν in ANCHOR_NU
            push!(pts, (η, ζ, ν))
        end
    end
    pts
end

# -----------------------
# SLURM chunking
# -----------------------
mkpath(OUT_DIR)
base_pts = build_base_points()
N_base   = length(base_pts)

task_id   = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
task_cnt  = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "1"))
chunk_sz  = cld(N_base, task_cnt)
i_start   = (task_id - 1) * chunk_sz + 1
i_end     = min(N_base, task_id * chunk_sz)

println("Total base points: $N_base | Tasks: $task_cnt | This task: $task_id handles [$i_start, $i_end]")
println("Timing levels (T_total=$(T_TOTAL)) : T_pre = $(join(TPRE_LIST, ", ")), a = $(join(string.(round.(A_LIST,digits=3)), ", "))")
println("q set: $(join(Q_SET, ", "))  | REPS per call: $(REPS)  | dt = $(dt)")

# -----------------------
# Run sims for this chunk (serial outer loops—your expectedRt is already threaded)
# -----------------------
rows = DataFrame(eta=Float64[], kappa=Float64[], nu=Float64[],
                 q=Float64[], m=Int[], a=Float64[],
                 p_valid_mean=Float64[], p_invalid_mean=Float64[],
                 T_total=Float64[], T_pre=Float64[], T_post=Float64[], reps=Int[])

flush_every = 10_000   # periodic flush to reduce RAM / help with long runs
buf = Vector{NamedTuple{(:eta,:kappa,:nu,:q,:m,:a,:p_valid_mean,:p_invalid_mean,:T_total,:T_pre,:T_post,:reps),
                        Tuple{Float64,Float64,Float64,Float64,Int,Float64,Float64,Float64,Float64,Float64,Float64,Int}}}()

for i in i_start:i_end
    η, ζ, ν = base_pts[i]
    # Reproducibility: one seed per base index (thread-safe with nested threads in expectedRt)
    Random.seed!(SEED_BASE + i)

    for m in M_LIST
        for Tpre in TPRE_LIST
            a = Tpre / T_TOTAL
            Tpost = T_TOTAL - Tpre
            κ = kappa_from_zeta(ζ, Tpost)

            pv, pi = expectedRt(η, κ, m, Q_SET, T_TOTAL, a, dt, REPS; nu=ν)

            @inbounds for j in eachindex(Q_SET)
                push!(buf, (eta=η, kappa=κ, nu=ν, q=Q_SET[j], m=m, a=a,
                            p_valid_mean=pv[j], p_invalid_mean=pi[j],
                            T_total=T_TOTAL, T_pre=Tpre, T_post=Tpost, reps=REPS))
            end
        end
    end

    if length(buf) >= flush_every
        tmpdf = DataFrame(buf)
        append!(rows, tmpdf)
        empty!(buf)
        @info "Progress" i i_end nrows=size(rows,1)
    end
end

if !isempty(buf)
    append!(rows, DataFrame(buf))
end

outpath = joinpath(OUT_DIR, "grid_chunk_$(task_id).csv")
CSV.write(outpath, rows)
println("Wrote $(nrow(rows)) rows to $outpath")


