package enigne

import "math"

type Value struct {
	Data     float64
	Grad     float64
	prev     []Value
	_backward func()
}

func NewValue(data float64) *Value {
	return &Value{
		Data: data,
		Grad: 0,
		prev: []Value{},
		_backward: func() {},
	}
}

func (v *Value) Add(other *Value) *Value {
	n := NewValue(v.Data + other.Data)
	n.prev = []Value{*v, *other}
	n._backward = func() {
		v.Grad += n.Grad
		other.Grad += n.Grad
	}

	return n
}

func (v *Value) Mul(other *Value) *Value {
	n := NewValue(v.Data * other.Data)
	n.prev = []Value{*v, *other}
	n._backward = func() {
		v.Grad += n.Grad * other.Data
		other.Grad += n.Grad * v.Data
	}

	return n
}

func (v *Value) Pow(other *Value) *Value {
	n := NewValue(math.Pow(v.Data, other.Data))
	n.prev = []Value{*v, *other}
	n._backward = func() {
		v.Grad += other.Data * math.Pow(v.Data, other.Data - 1)  * n.Grad
	}

	return n
}

func (v *Value) Tanh() *Value {
	n := NewValue(math.Tanh(v.Data))
	n.prev = []Value{*v}
	n._backward = func() {
		v.Grad += (1 - math.Pow(n.Data, 2)) * n.Grad
	}

	return n
}

func (v *Value) Neg() *Value {
	return v.Mul(NewValue(-1))
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(NewValue(-1)))
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := make(map[*Value]bool)

	var buildTopo func(*Value)

	buildTopo = func(Value *Value) {
		if !visited[Value] {
			visited[Value] = true
			for _, child := range Value.prev {
				buildTopo(&child)
			}
			topo = append(topo, Value)
		}
	}

	buildTopo(v)
	
	v.Grad = 1
	
	for i := len(topo) - 1; i >= 0; i-- {
		Value := topo[i]
		Value._backward()
	}
}
