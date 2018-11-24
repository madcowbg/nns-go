package neuralnet

import (
	"fmt"
	"math"
	"os"
)

type WeightVector []float64
type XVector []float64
type YVector []float64
type XSample []XVector
type YSample []YVector

type NNOrder struct {
	D int
	M []int
	K int
}

type NNStructure struct {
	NNOrder
	H             func(float64) float64
	H_prim        func(float64) float64
	Sigma         func(float64) float64
	ErrorFunction func(YVector, YVector) float64
}

type NetworkResponseType string

const (
	Regression       NetworkResponseType = "regression"
	BinaryClassifier NetworkResponseType = "binary"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func tanhDerivative(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}

func sigmoidDerivative(x float64) float64 {
	sx := sigmoid(x)
	return sx * (1 - sx)
}

func (order NNOrder) OfResponseType(responseType NetworkResponseType) *NNStructure {
	switch responseType {
	case Regression:
		return &NNStructure{order, math.Tanh, tanhDerivative, func(x float64) float64 { return x }, func(y YVector, t YVector) float64 { return ssqdiff(y, t) / 2 }}
	case BinaryClassifier:
		return &NNStructure{order, math.Tanh, tanhDerivative, sigmoid, crossentropy}
	default:
		panic(fmt.Sprintf("unknown response type %s", responseType))
	}
}

func (order *NNOrder) ExpectedPackedWeightsCount() int {
	if len(order.M) == 0 {
		panic("no hidden layers given - M = 0")
	}
	hiddenLayerWeights := order.M[0] * order.D
	for i := 1; i < len(order.M); i++ {
		hiddenLayerWeights += order.M[i-1] * order.M[i]
	}
	return hiddenLayerWeights + order.M[len(order.M)-1]*order.K
}

func (structure *NNStructure) ForWeights(wts WeightVector) NeuralNetwork {
	if len(wts) != structure.ExpectedPackedWeightsCount() {
		panic(fmt.Sprintf("invalid length of weights %d != %d", len(wts), structure.ExpectedPackedWeightsCount()))
	}
	return &MultiLayerNN{structure, wts, networkLayers(structure)}
}

func (structure *NNStructure) SNForWeights(wts WeightVector) NeuralNetwork {
	if len(wts) != structure.ExpectedPackedWeightsCount() {
		panic(fmt.Sprintf("invalid length of weights %d != %d", len(wts), structure.ExpectedPackedWeightsCount()))
	}
	if len(structure.M) != 1 {
		panic(fmt.Sprintf("can't creata e single hidden layer when there are more requested: %v", structure.M))
	}
	return &SingleLayerNN{structure, wts}
}

func networkLayers(structure *NNStructure) []int {
	L := make([]int, 2+len(structure.M))
	L[0] = structure.D
	copy(L[1:len(L)-1], structure.M)
	L[len(L)-1] = structure.K
	return L
}

func FitByCG(networkFor func(w0 WeightVector) NeuralNetwork, sampleX XSample, sampleT YSample, w0 WeightVector, verbose bool) NeuralNetwork {
	eta := 1.0
	MIN_ERF_IMPROVEMENT := 1e-12
	MAX_TRIES := 10000

	for tries := 0; tries < MAX_TRIES; tries++ {
		n0 := networkFor(w0)

		gradient := GradientSample(n0, sampleX, sampleT)

		ErfValueW0 := ErfSampleValue(n0, sampleX, sampleT)
		for et := 0; et <= 15; et++ {
			if ErfSampleValue(networkFor(perturbed(w0, gradient, -eta)), sampleX, sampleT) < ErfValueW0 {
				break
			}
			eta /= 2
		}

		if verbose {
			os.Stderr.WriteString(fmt.Sprintf("decreasing eta to %f...\n", eta))
		}

		for et := 0; et <= 15; et++ {
			if ErfSampleValue(networkFor(perturbed(w0, gradient, -2*eta)), sampleX, sampleT) >= ErfValueW0 {
				break
			}
			eta *= 2
		}

		if verbose {
			os.Stderr.WriteString(fmt.Sprintf("increasing eta to %f...\n", eta))
		}

		w1 := perturbed(w0, gradient, -eta)
		E_new := ErfSampleValue(networkFor(w1), sampleX, sampleT)

		if E_new-ErfValueW0 > -MIN_ERF_IMPROVEMENT || eta < 1e-15 {
			os.Stderr.WriteString(fmt.Sprintf("found the best error funciton... %f\n", ErfValueW0))
			return networkFor(w0)
		}

		if verbose {
			os.Stderr.WriteString(fmt.Sprintf("%f -> %f\n", ErfValueW0, E_new))
		}
		w0 = w1
	}

	best_nn := networkFor(w0)
	os.Stderr.WriteString(fmt.Sprintf("could not optimize error function beyond beyond %f...\n", ErfSampleValue(best_nn, sampleX, sampleT)))
	return best_nn
}
