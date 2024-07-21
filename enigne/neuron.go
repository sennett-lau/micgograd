package enigne

import (
	"math/rand"
)

type Neuron struct {
	Value *Value
	Weights []*Value
	Bias    *Value
	Inputs  []*Value
}

func NewNeuron(inputs []*Value) *Neuron {
	act := NewValue(0)

	weights := make([]*Value, len(inputs))
	for i := range weights {
		w := NewValue(rand.Float64()*2 - 1)
		weights[i] = w
		act = act.Add(w.Mul(inputs[i]))
	}

	bias := NewValue(rand.Float64()*2 - 1)
	act = act.Add(bias)

	output := act.Tanh()

	return &Neuron{Value: output, Weights: weights, Bias: bias, Inputs: inputs}
}

func (n *Neuron) GetParams() []*Value {
	return append(n.Weights, n.Bias)
}