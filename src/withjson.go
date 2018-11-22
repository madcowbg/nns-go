package main

import (
	"./neuralnet"
	"encoding/json"
	"io"
	"os"
)

func arrayOfSize(nRepeat int, value float64) []float64 {
	result := make([]float64, nRepeat)
	for i := range result {
		result[i] = value
	}
	return result
}

type Request struct {
	ShouldFit bool
	Order     neuralnet.NNOrder
	Wts       neuralnet.WeightVector
	X         neuralnet.XSample
	T         neuralnet.YSample
}

type Result struct {
	Wts       neuralnet.WeightVector
	Predicted neuralnet.YSample
	ErfValue  float64
	Gradient  neuralnet.WeightVector
	Hidden    [][]float64
}

func main() {
	dec := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(os.Stdout)
	for {
		request := Request{}

		if err := dec.Decode(&request); err != nil {
			if err != io.EOF {
				panic(err)
			}
			return
		}

		w0 := request.Wts
		if w0 == nil {
			os.Stderr.WriteString("Wts not given, defaulting to 1s...\n")
			w0 = make([]float64, request.Order.ExpectedPackedWeightsCount())
			for i := range w0 {
				w0[i] = 1
			}
		}

		x := request.X
		if x == nil {
			os.Stderr.WriteString("X not given, defaulting to 1s...\n")
			x = make(neuralnet.XSample, 1)
			x[0] = arrayOfSize(request.Order.D, 0.1)
		}

		t := request.T
		if t == nil {
			os.Stderr.WriteString("T not given, defaulting to 1s...\n")
			t = make(neuralnet.YSample, 1)
			t[0] = arrayOfSize(request.Order.K, 1.0)
		}

		var nn *neuralnet.SingleLayerNN
		var wts []float64
		if request.ShouldFit {
			nn, wts = request.Order.FitByCG(x, t, w0)
		} else {
			nn = request.Order.ForWeights(w0)
			wts = w0
		}

		result := Result{
			wts,
			nn.PredictSample(x),
			nn.ErfSampleValue(x, t),
			nn.GradientSample(x, t),
			nn.HiddenSample(x),
		}

		if err := enc.Encode(result); err != nil {
			panic(err)
		}
	}
}
