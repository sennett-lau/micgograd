package enigne

func MSE(ys, yps []*Value) *Value {
	loss := NewValue(0)
	for i, y := range ys {
		loss = loss.Add(yps[i].Sub(y).Pow(NewValue(2)))
	}

	return loss
}