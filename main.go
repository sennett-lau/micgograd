package main

import (
	"fmt"

	"github.com/sennett-lau/micgograd/enigne"
)

func main() {
	xs := [][]*enigne.Value{
		{enigne.NewValue(2), enigne.NewValue(3), enigne.NewValue(-1)},
		{enigne.NewValue(3), enigne.NewValue(-1), enigne.NewValue(0.5)},
		{enigne.NewValue(0.5), enigne.NewValue(1), enigne.NewValue(1)},
		{enigne.NewValue(1), enigne.NewValue(1), enigne.NewValue(-1)},
	}


	ys_values := []*enigne.Value{
		enigne.NewValue(1),
		enigne.NewValue(-1),
		enigne.NewValue(-1),
		enigne.NewValue(1),
	}

	mlp := enigne.NewMLP(3, []int{4, 4, 1})

	preds := make([]*enigne.Value, len(xs))

	loss := enigne.NewValue(0)

	for epoch := 0; epoch < 50000; epoch++ {
		for i, x := range xs {
			preds[i] = mlp.Forward(x)[0]
		}

		loss = enigne.MSE(ys_values, preds)

		if epoch%100 == 0 {
			fmt.Print("Epoch\t", epoch, "\tLoss\t", loss.Data, "\tPredictions:\t")
			for _, pred := range preds {
				if pred.Data > 0 {
					fmt.Printf("+%.8f\t", pred.Data)
				} else {
					fmt.Printf("%.8f\t", pred.Data)
				}
			}
			fmt.Println()
		}

		loss.Backward()

		lr := 0.00001

		mlp.GradientDescent(lr)
		mlp.ZeroGrad()
	}

	fmt.Print("Epoch\tFinal\tLoss\t", loss.Data, "\tPredictions:\t")
	for _, pred := range preds {
		if pred.Data > 0 {
			fmt.Printf("+%.8f\t", pred.Data)
		} else {
			fmt.Printf("%.8f\t", pred.Data)
		}
	}

}
