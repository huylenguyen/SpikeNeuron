module SpikeNeuron

include("./src/Pattern.jl")
using .Pattern: InputPattern, Spike, getSpikeQueue
export InputPattern, Spike, getSpikeQueue

include("./src/Neuron.jl")
using .Neuron: NeuronModel, resetNeuron!, findNextSpike!, Vt, simulateEventBased!, simulateClockBased!
export NeuronModel, resetNeuron!, findNextSpike!, Vt, simulateEventBased!, simulateClockBased!

end