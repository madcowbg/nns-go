package main

import (
	"./neuralnet"
	"encoding/json"
	"io"
	"os"
)

type Request struct {
	ShouldFit bool
	NetworkRT neuralnet.NetworkResponseType
	Order     neuralnet.NNOrder
	Wts       neuralnet.WeightVector
	X         neuralnet.XSample
	T         neuralnet.YSample
	Verbose   bool
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
			os.Stderr.WriteString("X not given, defaulting to 0.1s...\n")
			x = make(neuralnet.XSample, 1)
			x[0] = neuralnet.ArrayOfSize(request.Order.D, 0.1)
		}

		t := request.T
		if t == nil {
			os.Stderr.WriteString("T not given, defaulting to 1s...\n")
			t = make(neuralnet.YSample, 1)
			t[0] = neuralnet.ArrayOfSize(request.Order.K, 1.0)
		}

		responseType := request.NetworkRT
		if responseType == "" {
			responseType = neuralnet.Regression
		}

		var nn neuralnet.NeuralNetwork
		if request.ShouldFit {
			nn = neuralnet.FitByCG(request.Order.OfResponseType(responseType).ForWeights, x, t, w0, request.Verbose, 1e-12, 10000)
		} else {
			nn = request.Order.OfResponseType(responseType).ForWeights(w0)
		}
		wts := nn.PackedWts()

		result := Result{
			wts,
			neuralnet.PredictSample(nn, x),
			neuralnet.ErfSampleValue(nn, x, t),
			neuralnet.GradientSample(nn, x, t),
			neuralnet.HiddenSample(nn, x),
		}

		if err := enc.Encode(result); err != nil {
			panic(err)
		}
	}
}
