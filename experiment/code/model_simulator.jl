using Random
using LinearAlgebra
using Base.Threads
using Distributions


function expectedRt(eta::Float64, kappa::Float64, m::Int, q_values::Vector{Float64},
                    T::Float64, a::Float64, dt::Float64, reps::Int; nu::Float64=1.0)

    # -- Time range --
    tspan     = 0.0:dt:T
    len_tspan = length(tspan)
    
    # -- Resource targets pre‐ and post‐cue --
    x0_hat = fill(eta / m, m)  # baseline
    
    # -- OU‐like parameters --
    delta = 1 - exp(-kappa * dt)
    sigma = sqrt(nu * (1 - exp(-2*kappa * dt)) / (2*kappa))  # nu controls noise scaling

    # -- When cue arrives (index in tspan) --
    tq = Int(floor(a * T / dt)) + 1

    # --------------------
    #  1) Simulate X(t)
    # --------------------
    function Xtrad_sim(X0::Vector{Float64}, epsilon::Matrix{Float64}, q::Float64)
        # Calculate post-cue targets for this specific q value
        x1_hat = clamp.(
            x0_hat .+ log((m - 1)*q / (1 - q)) / m .* vcat(m - 1, fill(-1, m-1)),
            0, eta
        )
        
        Xt = copy(X0)
        Xtrad = zeros(len_tspan, m)
        Xtrad[1, :] = Xt

        # Before cue arrives:
        for t in 2:tq-1
            theta_Xt     = Xt .> 0.0
            sum_theta_Xt = count(theta_Xt)
            raw_deltaX   = delta .* (x0_hat .- Xt) .+ sigma .* epsilon[t, :]
            deltaX       = theta_Xt .* raw_deltaX .- (dot(theta_Xt, raw_deltaX)/sum_theta_Xt) .* theta_Xt
            Xt          .= Xt .+ deltaX
            
            # Safeguard: clamp & renormalize after each update
            Xt .= max.(Xt, 0.0)
            s = sum(Xt)
            if s > 0
                Xt .*= (eta / s)   # keep total resource conserved
            else
                Xt .= eta / m      # degenerate rescue
            end
            
            Xtrad[t, :] = Xt
        end

        # After cue arrives:
        for t in tq:len_tspan
            theta_Xt     = Xt .> 0.0
            sum_theta_Xt = count(theta_Xt)
            raw_deltaX   = delta .* (x1_hat .- Xt) .+ sigma .* epsilon[t, :]
            deltaX       = theta_Xt .* raw_deltaX .- (dot(theta_Xt, raw_deltaX)/sum_theta_Xt) .* theta_Xt
            Xt          .= Xt .+ deltaX
            
            # Safeguard: clamp & renormalize after each update
            Xt .= max.(Xt, 0.0)
            s = sum(Xt)
            if s > 0
                Xt .*= (eta / s)   # keep total resource conserved
            else
                Xt .= eta / m      # degenerate rescue
            end
            
            Xtrad[t, :] = Xt
        end
        return Xtrad
    end

    # ---------------------------
    #  2) Recall probability
    # ---------------------------

    # ---- "internal" recall probabilities (ignoring q) ----
    @inline function R_internal_cued(x::AbstractVector{<:Real})
        return 1 - exp(-x[1])
    end

    @inline function R_internal_non_cued(x::AbstractVector{<:Real})
        exp_neg_x = exp.(-x)
        return (1/(m-1)) * sum(1 .- exp_neg_x[2:end])
    end

    # ----------------------------------------
    #  3) Run many simulations & average
    # ----------------------------------------
    
    # Initialize arrays to store results for each q value
    Rmean_intCue_all  = zeros(length(q_values))
    Rmean_intNcue_all = zeros(length(q_values))
    
    # Loop through each q value
    for (q_idx, q) in enumerate(q_values)
        Rsum_intCue  = zeros(nthreads())
        Rsum_intNcue = zeros(nthreads())

        @threads for i in 1:reps
            
            X0 = eta * rand(Dirichlet(ones(m)))   # random initial distribution
            epsilon  = randn(len_tspan, m)
            Xtrad_out = Xtrad_sim(X0, epsilon, q)

            Rsum_intCue[threadid()] += R_internal_cued(Xtrad_out[end, :])
            Rsum_intNcue[threadid()] += R_internal_non_cued(Xtrad_out[end, :])  
        end

        # Average over reps for this q value
        Rmean_intCue_all[q_idx]  = sum(Rsum_intCue) ./ reps
        Rmean_intNcue_all[q_idx] = sum(Rsum_intNcue) ./ reps
    end

    return (Rmean_intCue_all, Rmean_intNcue_all)

end

# Example usage
# eta    = 3.00 # capacity -- between > .00001 and 5
# kappa    = .5 # control  --  greater than 1.  # between 0 - 1
# m    = 2 # load
# q_values = [0.6, 0.7, 0.8, 0.9, 0.92] # cue reliability values to test
# T    = 2.45 # total time
# a    = 0.73 # proportion of time cue arrives
# dt   = 0.001 # time per step
# reps = 10^2 # number steps
# nu    = 1.0  # noise scaling parameter (default 1.0)

# Get recall probabilities for all q values
# Rmean_intCue_all, Rmean_intNcue_all = expectedRt(eta, kappa, m, q_values, T, a, dt, reps, nu=1.0)

# Print results
# println("Cue reliability values: ", q_values)
# println("Cued recall probabilities: ", Rmean_intCue_all)
# println("Non-cued recall probabilities: ", Rmean_intNcue_all)
