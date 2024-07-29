package enigne

import (
	"math/rand"
)

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(numOfInputs int) *Neuron {
	weights := make([]*Value, numOfInputs)
	for i := range weights {
		w := NewValue(rand.Float64()*2 - 1)
		weights[i] = w
	}

	bias := NewValue(rand.Float64()*2 - 1)

	return &Neuron{Weights: weights, Bias: bias}
}

func (n *Neuron) Forward(inputs []*Value) *Value {
	act := NewValue(0)
	for i, input := range inputs {
		act = act.Add(n.Weights[i].Mul(input))
	}
	act = act.Add(n.Bias)
	return act.Tanh()
}

func (n *Neuron) GetParams() []*Value {
	return append(n.Weights, n.Bias)
}

func (n *Neuron) ZeroGrad() {
	params := n.GetParams()
	for _, p := range params {
		p.Grad = 0
	}
}
