package enigne

import (
	"math"
	"testing"
)

func TestNeuronData(t *testing.T) {
	a := NewValue(3)
	b := NewValue(2)

	c := NewNeuron(2)
	
	out := c.Forward([]*Value{a, b})

	params := c.GetParams()
	
	w1 := params[0]
	w2 := params[1]
	bias := params[2]

	act := w1.Data * a.Data + w2.Data * b.Data + bias.Data

	if (out.Data != math.Tanh(act)) {
		t.Errorf("Neuron value is not correct")
	}
}

func TestNeuronGrad(t *testing.T) {
	a := NewValue(3)
	b := NewValue(2)
	c := NewNeuron(2)
	
	params := c.GetParams()
	out := c.Forward([]*Value{a, b})
	
	w1 := params[0]
	w2 := params[1]
	bias := params[2]

	act := w1.Data * a.Data + w2.Data * b.Data + bias.Data


	out.Backward()
	

	if (out.Grad != 1) {
		t.Errorf("Output gradient is not correct")
	}

	tanhGrad := 1 - math.Pow(math.Tanh(act), 2)

	if (bias.Grad != tanhGrad) {
		t.Errorf("Bias gradient is not correct")
	}

	if (w1.Grad != a.Data * tanhGrad) {
		t.Errorf("Weight 1 gradient is not correct")
	}

	if (w2.Grad != b.Data * tanhGrad) {
		t.Errorf("Weight 2 gradient is not correct")
	}
}
