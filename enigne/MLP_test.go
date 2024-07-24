package enigne

import (
	"math"
	"testing"
)

func TestMLP(t *testing.T) {
	a := NewValue(2)
	b := NewValue(3)
	c := NewValue(-1)

	inputs := []*Value{a, b, c}
	outputSizes := []int{4, 4, 1}

	mlp := NewMLP(3, outputSizes)

	outs := mlp.Forward(inputs)

	currInputs := inputs

	expectedOutOfMLP := 0.0
	for _, layer := range mlp.Layers {
		expectedOutOfLayer := 0.0
		newInputs := []*Value{}
		for _, neuron := range layer.Neurons {
			expectedOutOfNeuron := neuron.Bias.Data
			for k, w := range neuron.Weights {
				expectedOutOfNeuron += w.Data * currInputs[k].Data
			}
			expectedOutOfNeuron = math.Tanh(expectedOutOfNeuron)
			newInputs = append(newInputs, NewValue(expectedOutOfNeuron))
			expectedOutOfLayer += expectedOutOfNeuron
		}
		expectedOutOfMLP = expectedOutOfLayer
		currInputs = newInputs
	}

	if outs[0].Data > expectedOutOfMLP + 0.000001 || outs[0].Data < expectedOutOfMLP - 0.000001 {
		t.Errorf("Expected %f, got %f", expectedOutOfMLP, outs[0].Data)
	}
}