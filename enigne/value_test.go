package enigne

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	a := NewValue(2)
	b := NewValue(3)
	c := a.Add(b)

	if c.Data != 5 {
		t.Errorf("Expected c.Data to be 5, but got %v", c.Data)
	}
}

func TestMul(t *testing.T) {
	a := NewValue(2)
	b := NewValue(3)
	c := a.Mul(b)

	if c.Data != 6 {
		t.Errorf("Expected c.Data to be 6, but got %v", c.Data)
	}
}

func TestPow(t *testing.T) {
	a := NewValue(2)
	b := NewValue(3)
	c := a.Pow(b)

	if c.Data != 8 {
		t.Errorf("Expected c.Data to be 8, but got %v", c.Data)
	}
}

func TestSub(t *testing.T) {
	a := NewValue(5)
	b := NewValue(3)
	c := a.Sub(b)

	if c.Data != 2 {
		t.Errorf("Expected c.Data to be 2, but got %v", c.Data)
	}
}

func TestDiv(t *testing.T) {
	a := NewValue(6)
	b := NewValue(3)
	c := a.Div(b)

	if c.Data != 2 {
		t.Errorf("Expected c.Data to be 2, but got %v", c.Data)
	}
}

func TestBackward(t *testing.T) {
	a := NewValue(2)
	b := NewValue(3)
	
	// Test Add
	c := a.Add(b)
	c.Backward()
	if a.Grad != 1 || b.Grad != 1 {
		t.Errorf("Add: Expected a.Grad and b.Grad to be 1, but got a.Grad=%v, b.Grad=%v", a.Grad, b.Grad)
	}

	// Reset gradients
	a.Grad, b.Grad = 0, 0

	// Test Sub
	c = a.Sub(b)
	c.Backward()
	if a.Grad != 1 || b.Grad != -1 {
		t.Errorf("Sub: Expected a.Grad to be 1 and b.Grad to be -1, but got a.Grad=%v, b.Grad=%v", a.Grad, b.Grad)
	}

	// Reset gradients
	a.Grad, b.Grad = 0, 0

	// Test Mul
	c = a.Mul(b)
	c.Backward()
	if a.Grad != b.Data || b.Grad != a.Data {
		t.Errorf("Mul: Expected a.Grad to be %v and b.Grad to be %v, but got a.Grad=%v, b.Grad=%v", b.Data, a.Data, a.Grad, b.Grad)
	}

	// Reset gradients
	a.Grad, b.Grad = 0, 0

	// Test Div
	c = a.Div(b)
	c.Backward()
	if a.Grad != 1/b.Data || b.Grad != -a.Data/(b.Data*b.Data) {
		t.Errorf("Div: Expected a.Grad to be %v and b.Grad to be %v, but got a.Grad=%v, b.Grad=%v", 1/b.Data, -a.Data/(b.Data*b.Data), a.Grad, b.Grad)
	}

	// Reset gradients
	a.Grad, b.Grad = 0, 0

	// Test Pow
	c = a.Pow(b)
	c.Backward()
	expectedAGrad := b.Data * math.Pow(a.Data, b.Data-1)
	if a.Grad != expectedAGrad {
		t.Errorf("Pow: Expected a.Grad to be %v, but got a.Grad=%v", expectedAGrad, a.Grad)
	}
}