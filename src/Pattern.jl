module Pattern

using Random
using Distributions
using DataStructures

struct Spike
    #=
    A single spike event.

    Variables
    ---------
    channel::Int64
        Index of presynaptic neuron
    time::Real
        Spike time (ms)
    =#
    channel::Int64
    time::Real
end#Spike

mutable struct InputPattern
    #=
    A temporal input spike pattern.

    Variables
    ---------
    pattern::Vector{Vector{Float64}}
        Two-dimensional spike events
    =#

    pattern::Vector{Vector{Float64}}
end#InputPattern

function InputPattern(N::Int64, tStart::Real, tEnd::Real, p::Float64)::InputPattern
    #=
    Generates a vector containing random discrete binary patterns
    as a homogeneous Poisson point process. The pattern can be empty
    if T*p is too small. 

    Parameters
    ----------
    N::Int64
        The number of patterns to generate
    T::Float64
        The Lebesgue measure, or pattern duration (ms)
    p::Float64
        The Poisson rate (spike/ms)

    Return
    ------
    Vector{Vector{Float64}}
    =#

    # generate pattern
    pattern = [sort(rand(Uniform(tStart, tEnd), rand(Poisson(p * (tEnd-tStart))))) for x in 1:N]

    return InputPattern(pattern)

end#InputPattern()


function getSpikeQueue(input::InputPattern)::PriorityQueue{Spike, Real}
    #=
    Convert an input pattern into an Event queue
    =#
    pattern = input.pattern

    eq = PriorityQueue{Spike, Real}()
    for n in 1:length(pattern)
        for spike in pattern[n]
            enqueue!(eq, Spike(n, spike), spike)
        end#for
    end#for

    return eq
end


end#module
