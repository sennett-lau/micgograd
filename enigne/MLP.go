package enigne

type MLP struct {
	Layers []*Layer
}

func NewMLP(inputSize int, outputSizes []int) *MLP {
	layers := make([]*Layer, len(outputSizes) + 1)

	for i, outputSize := range append([]int{inputSize}, outputSizes...) {
		layers[i] = NewLayer(inputSize, outputSize)
		inputSize = outputSize
	}

	return &MLP{Layers: layers}
}

func (m *MLP) Forward(inputs []*Value) []*Value {
	for _, layer := range m.Layers {
		inputs = layer.Forward(inputs)
	}
	return inputs
}

func (m *MLP) GetParams() []*Value {
	params := []*Value{}
	for _, layer := range m.Layers {
		params = append(params, layer.GetParams()...)
	}
	return params
}

func (m *MLP) ZeroGrad() {
	for _, layer := range m.Layers {
		layer.ZeroGrad()
	}
}

func (m *MLP) GradientDescent(lr float64) {
	params := m.GetParams()
	for _, p := range params {
		p.Data -= p.Grad * lr
	}
}
