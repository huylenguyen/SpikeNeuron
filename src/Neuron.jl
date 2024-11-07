module Neuron

# package imports
using Random
using Distributions
using DataStructures
using Roots

using ..SpikeNeuron: InputPattern, getSpikeQueue

mutable struct NeuronModel
    #=
    An integrate-and-fire spiking neuron model with double-exponential
    temporal integration kernel and exponential temporal reset kernel. 

    Variables
    ---------
    τₘ::Float64 
        Membrane time constant [ms]
    τₛ::Float64 
        synaptic time constant [ms]
    ϑ::Float64 
        spiking threshold [mV]
    Vrest::Float64 
        reset voltage [mV]
    η::Float64 
        τₘ/τₛ ratio
    Vnorm::Float64 
        Normalising constant of temporal integration kernel
    A::Float64 
        (τₘ*τₛ)/(τₘ-τₛ). Convenience variable for computing spike time
    logη::Float64 
        Log of η. Convenience variable for computing spike time
    Eₘ::Float64
        State variable containing current state of integration
        kernel first exponential
    Eₛ::Float64
        State variable containing current state of integration
        kernel second exponential
    Eₑ::Float64
        State variable containing current state of reset kernel
    τₑ::Float64
        temporal scaling parameter for preventing numerical overflow
    Nϵ::Float64
        Convenience variable for computing membrane potential.
        Prevents numerical overflow when simulation time is large.
    ΔTϵ::Float64
        Convenience variable for computing membrane potential.
        Represents time scaled by Nϵ and τϵ
    τϵ::Float64
        Convenience variable for computing membrane potential.
        Scales the temporal dimension by a constant factor.
    N::Int64
        Number of presynaptic connections
    w::Vector{Float64}
        Presynaptic weights
    K::function
        input kernel K(t-tᵢ)
    K̇::function
        time derivative of input kernel K̇(t)
    S::function
        output kernel S(t-tₛ)
    =#

    # Constants
    τₘ::Float64
    τₛ::Float64
    ϑ::Float64
    Vrest::Float64
    η::Float64 # τₘ/τₛ ratio
    Vnorm::Float64
    logη::Float64
    A::Float64

    # Membrane potential state variables
    Eₘ::Float64
    Eₛ::Float64
    Eₑ::Float64
    τₑ::Float64
    Nϵ::Float64
    ΔTϵ::Float64
    τϵ::Float64

    # Weights
    N::Int64
    w::Vector{Float64}

    # functions
    K
    K̇
    S
end#NeuronModel

function NeuronModel(
    N::Integer; 
    τₘ::Float64=20.0, 
    τₛ::Float64=τₘ/4,
    ϑ::Float64=1.0, 
    Vrest::Float64=0.0,
    w₀::Union{Vector{Float64}, Nothing}=nothing,
    Φ::Distribution=Normal(6/N, 5/N))::NeuronModel
    τₑ::Real=700
    #=
    Initialises a Neuron object. 

    Parameters
    ----------
    N::Int64
        Number of presynaptic connections
    τₘ::Float64, Optional
        Membrane time constant [ms]
    τₛ::Float64, Optional
        synaptic time constant [ms]
    ϑ::Float64, Optional
        spiking threshold [mV]
    Vrest::Float64, Optional
        reset voltage [mV]
    w₀::Vector{Float64}, Optional
        initial weights
    Φ::Distribution, Optional
        distribution to generate weights from if initial weights are not provided
    τₑ::Float64
        temporal scaling parameter for preventing numerical overflow
    =#
    
    # Validate inputs
    @assert N > 1 "There must be at least one input connection"
    @assert τₘ ≥ 0 "Membrane time constant must be positive"
    @assert τₛ ≥ 0 "Synaptic time constant must be positive"
    @assert Vrest ≤ ϑ "Threshold must be above rest potential"
    if !isnothing(w₀) 
        @assert length(w₀) == N "Initial weights does not contain N values"
    end#if

    # Calculate constants
    η = τₘ/τₛ
    logη = log(η)
    Vnorm = η^(η/(η-1))/(η-1)
    A = (τₘ*τₛ)/(τₘ-τₛ)

    # Input kernel K. Takes Real time to enable discrete inputs
    K(t::Real)::Float64 = t < 0 ? 0 : (exp(-(t)/τₘ) - exp(-(t)/τₛ)) * Vnorm
    K̇(t::Real)::Float64 = t < 0 ? 0 : (-exp(-(t)/τₘ)/τₘ - exp(-(t)/τₛ)/τₛ) * Vnorm # TODO check this derivative in relation to my version of Vnorm

    # Output kernel S
    S(t::Real)::Float64 = t < 0 ? 0 : exp(-(t)/τₘ)

    # membrane potential state variables
    Eₘ, Eₛ, Eₑ  = 0., 0., 0.
    Nϵ, ΔTϵ, τϵ = 0., 0., τₑ*τₛ

    # Initialise weights
    if isnothing(w₀) 
        w = rand(Φ, N) 
    else 
        w = w₀ 
    end#if

    # debug
    @debug "w", w

    return NeuronModel(τₘ,τₛ,ϑ,Vrest,η,Vnorm,logη,A,Eₘ,Eₛ,Eₑ,τₑ,Nϵ,ΔTϵ,τϵ,N,w,K,K̇,S)
end#NeuronModel()


function resetNeuron!(
        neuron::NeuronModel,
        Eₘ::Float64=0.,
        Eₛ::Float64=0.,
        Eₑ::Float64=0.
    )::Nothing
    #=
    Reset a neuron to its resting state, or optionally to a
    specific membrane potential state based on the provided
    variables.

    Parameters
    ----------
    neuron::NeuronModel
        The neuron to reset
    Eₘ::Float64
        State variable containing current state of integration
        kernel first exponential
    Eₛ::Float64
        State variable containing current state of integration
        kernel second exponential
    Eₑ::Float64
        State variable containing current state of reset kernel

    Return
    ------
    Nothing
    =#

    # set membrane potential state variables
    neuron.Eₘ = Eₘ
    neuron.Eₛ = Eₛ
    neuron.Eₑ = Eₑ
    neuron.Nϵ = 0.
    neuron.ΔTϵ = 0.

    return nothing
end#resetNeuron!()


function simulateEventBased!(neuron::NeuronModel, pattern::InputPattern)
    #=
    Simulate a neuron using a spatiotemporal input pattern in an event-based 
    fashion.

    Parameters
    ----------
    neuron::NeuronModel
        The neuron to simulate
    pattern::InputPattern
        The input pattern

    Return
    ------
    Vector{Float64}
        The output spike times in ms
    =#

    # get event queue
    eq = getSpikeQueue(pattern)

    # output spike times
    spikes = Float64[]
    # Normalise layer weights
    W = neuron.w * neuron.Vnorm

    # Process each spike event
    while length(eq) > 0

        # get current spike
        currentSpike = dequeue!(eq)
        nᵢ, tᵢ = currentSpike.channel, currentSpike.time

        # get next spike
        if length(eq) > 1
            tⱼ = peek(eq)[2]
            tⱼ = tᵢ+3neuron.τₘ
        else
            tⱼ = tᵢ+3neuron.τₘ
        end#if
        
        # Protect against numerical overflow on exponential computations
        N_ϵ = Int(floor(tᵢ / neuron.τϵ))
        if N_ϵ > neuron.Nϵ
            ΔT_ϵ = (N_ϵ - neuron.Nϵ) * neuron.τϵ
            neuron.Eₘ *= exp(-ΔT_ϵ / neuron.τₘ)
            neuron.Eₛ  *= exp(-ΔT_ϵ / neuron.τₛ)
            neuron.Eₑ *= exp(-ΔT_ϵ / neuron.τₘ)
            neuron.Nϵ  = N_ϵ
            neuron.ΔTϵ = neuron.Nϵ * neuron.τϵ
        end

        # Update membrane potential state
        neuron.Eₘ += W[nᵢ] * exp((tᵢ-neuron.ΔTϵ) / neuron.τₘ)
        neuron.Eₛ += W[nᵢ] * exp((tᵢ-neuron.ΔTϵ) / neuron.τₛ)

        # calculate membrane potential maximum
        tmax = getTMax(tᵢ, tⱼ, neuron)
        vmax = Vt(tmax, neuron)

        # @debug vmax ≥ neuron.ϑ, neuron.Eₘ, neuron.Eₛ, neuron.Eₑ
        
        # Solve for precise spike time
        tₛ = tᵢ
        while vmax ≥ neuron.ϑ

            # compute spike
            tₒ = findNextSpike!(neuron, tₛ, tⱼ, tmax, vmax)

            # save spike event
            push!(spikes, tₒ)

            # update reset
            neuron.Eₑ += exp((tₒ-neuron.ΔTϵ) / neuron.τₘ)

            # continue searching for next spike
            tₛ    = tₒ + eps(Float64)
            tmax = getTMax(tᵢ, tⱼ, neuron)
            vmax = Vt(tmax, neuron)
            
        end#while vmax≥ϑ
    end#while spike
    
    return spikes
end#simulateEventBased!()


function simulateClockBased!(
    neuron::NeuronModel, 
    pattern::InputPattern, 
    tStart::Real, 
    tEnd::Real, 
    dt::Real
)::Tuple{Vector{Float64}, Vector{Float64}}
    #=
    Simulate a neuron using a spatiotemporal input pattern in a clock-based 
    fashion.

    Parameters
    ----------
    neuron::NeuronModel
        The neuron to simulate
    pattern::InputPattern
        The input pattern
    tStart::Real
        Simulation start time
    tEnd::Real
        Simulation end time
    dt::Real
        The time increment value

    Return
    ------
    Tuple{Vector{Float64}, Vector{Float64}}
        The output spike times in ms, and the membrane potential trace
    =#

    # get event queue
    eq = getSpikeQueue(pattern)

    # validate inputs
    @assert tStart ≤ peek(eq)[2] "Start time is later than first input spike"

    # output spike times
    spikes = Float64[]
    # membrane potential trace
    Vs     = Float64[]
    # Normalise layer weights
    W = neuron.w * neuron.Vnorm

    # Process each spike event
    t = tStart
    while length(eq) > 0

        # get next spike
        nextSpike = dequeue!(eq)
        nᵢ, tᵢ = nextSpike.channel, nextSpike.time

        # simulate neuron until input spike
        while t < tᵢ
            # calculate membrane potential
            V = Vt(t, neuron)
            append!(Vs, V)

            # check for output spike
            if V > neuron.ϑ
                # update reset
                neuron.Eₑ += exp((t-neuron.ΔTϵ) / neuron.τₘ)

                append!(spikes, t)
            end#if 

            t += dt
        end#for
        
        # Protect against numerical overflow on exponential computations
        N_ϵ = Int(floor(tᵢ / neuron.τϵ))
        if N_ϵ > neuron.Nϵ
            ΔT_ϵ = (N_ϵ - neuron.Nϵ) * neuron.τϵ
            neuron.Eₘ *= exp(-ΔT_ϵ / neuron.τₘ)
            neuron.Eₛ  *= exp(-ΔT_ϵ / neuron.τₛ)
            neuron.Eₑ *= exp(-ΔT_ϵ / neuron.τₘ)
            neuron.Nϵ  = N_ϵ
            neuron.ΔTϵ = neuron.Nϵ * neuron.τϵ
        end

        # Update membrane potential state with input spike
        neuron.Eₘ += W[nᵢ] * exp((tᵢ-neuron.ΔTϵ) / neuron.τₘ)
        neuron.Eₛ += W[nᵢ] * exp((tᵢ-neuron.ΔTϵ) / neuron.τₛ)

    end#while

    # simulate until tEnd
    for i = t:dt:tEnd
        V = Vt(i, neuron)
        append!(Vs, V)

        # check for output spike
        if V > neuron.ϑ
            # update reset
            neuron.Eₑ += exp((i-neuron.ΔTϵ) / neuron.τₘ)
            
            append!(spikes, i)
        end#if 
    end#for

    return spikes, Vs
end#simulateClockBased!()


function getTMax(tᵢ::Real, tⱼ::Real, neuron::NeuronModel)::Float64
    #= 
    Search for the next time of local membrane potential maxima.

    Parameters
    ----------
    tᵢ::Real
        Search start time
    tⱼ::Real
        Search end time
    neuron::NeuronModel
        The neuron model

    Return
    ------
    Tuple{Float64, Bool}
        A tuple denoting the maximum time, and whether a local maximum was found
    =#

    # get next local extrema
    rem = (neuron.Eₘ - neuron.ϑ*neuron.Eₑ) / neuron.Eₛ

    localmax = true

    if rem ≤ 0
        localmax = false
    else
        # Analytical expression for tmax, obtained by evaluating dV/dt = 0
        tmax = neuron.ΔTϵ + neuron.A * (neuron.logη - log(rem))
    end#if

    # Clamp the local maximum to the search interval
    localmax = localmax && (tᵢ < tmax < tⱼ)
    if !(localmax && (tᵢ < tmax < tⱼ)) 
        tmax = tⱼ 
    end#if

    return tmax
end#getTMax()


function Vt(t::Real, neuron::NeuronModel)::Float64
    #=
    Calculate membrane potential at time t

    Parameters
    ----------
    t::Real
        Time to query voltage
    neuron::NeuronModel
        The neuron model

    Return
    ------
    Float64
        membrane voltage in mV
    =#

    # calculate time difference to current state
    Δt = t - neuron.ΔTϵ

    # compute membrane potential
    emt = exp(-Δt/neuron.τₘ)
    est = exp(-Δt/neuron.τₛ)

    return emt*neuron.Eₘ - est*neuron.Eₛ - neuron.ϑ*emt*neuron.Eₑ

end#Vt()


function findNextSpike!(
    neuron::NeuronModel,
    tᵢ::Real, 
    tⱼ::Real, 
    tmax::Union{Real, Nothing}=nothing,
    vmax::Union{Real, Nothing}=nothing
)::Union{Float64, Nothing}
    #=
    Compute the next spike time based on the current state of the
    neuron. 

    Parameters
    ----------
    neuron::NeuronModel
        The neuron model
    tᵢ::Real
        Search start time
    tⱼ::Real
        Search end time
    tmax::Union{Real, Nothing}
        If provided, skips searching for max spike time
    vmax::Union{Real, Nothing}
        If provided, skips searching for max membrane potential

    Return
    ------
    Union{Float64, Nothing}
        A spike time if one was found within the search range
    =#

    # calculate membrane potential maximum
    if isnothing(tmax) && isnothing(vmax)
        tmax = getTMax(tᵢ, tⱼ, neuron)
        vmax = Vt(tmax, neuron)
    end#if

    if vmax == neuron.ϑ tₒ = tmax
    elseif vmax > neuron.ϑ
        try
            tₒ = Roots.find_zero(t::Float64 -> Vt(t, neuron)-neuron.ϑ, (tᵢ,tmax), Roots.Brent())
        catch err
            @debug "Error: ", err
            tₒ = Roots.find_zero(t::Float64 -> Vt(t, neuron)-neuron.ϑ, tᵢ)
        end#try
    else
        tₒ = nothing
    end#if

    # @debug tmax, Vt(tᵢ, neuron), vmax, tₒ

    return tₒ
end#findNextSpike!()

end#Neuron