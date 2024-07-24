package enigne

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(numOfInputs int, numOfNeurons int) *Layer {
	neurons := make([]*Neuron, numOfNeurons)
	for i := range neurons {
		neurons[i] = NewNeuron(numOfInputs)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Forward(inputs []*Value) []*Value {
	outputs := make([]*Value, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Forward(inputs)
	}
	return outputs
}

func (l *Layer) GetParams() []*Value {
	params := []*Value{}
	for _, n := range l.Neurons {
		params = append(params, n.GetParams()...)
	}
	return params
}
