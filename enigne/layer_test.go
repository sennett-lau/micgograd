package enigne

import (
	"math"
	"testing"
)

func TestLayer(t *testing.T) {
	a := NewValue(1)
	b := NewValue(2)

	inputs := []*Value{a, b}
	
	L := NewLayer(2, 3)

	outs := L.Forward(inputs)

	for i, n := range L.Neurons {
		expect := n.Bias.Data

		for ii, w := range n.Weights {
			expect += w.Data * inputs[ii].Data
		}
		expect = math.Tanh(expect)
		if outs[i].Data != expect {
			t.Errorf("Expected %f, got %f", expect, outs[i].Data)
		}
	}
}
