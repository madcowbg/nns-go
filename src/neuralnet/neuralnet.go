package neuralnet

import "gonum.org/v1/gonum/floats"

type NeuralNetwork interface {
	PackedWts() []float64

	Predict(x XVector) YVector

	ErfValue(x XVector, t YVector) float64

	Gradient(x XVector, t YVector) WeightVector

	Hidden(x XVector) []float64
}

func ErfSampleValue(nn NeuralNetwork, x XSample, t YSample) float64 {
	value := 0.0
	for i := range x {
		value += nn.ErfValue(x[i], t[i])
	}
	return value
}

func PredictSample(nn NeuralNetwork, sample_x XSample) YSample {
	result := make(YSample, len(sample_x))
	for i, xv := range sample_x {
		result[i] = nn.Predict(xv)
	}
	return result
}

func GradientSample(nn NeuralNetwork, sample_x XSample, sample_t YSample) []float64 {
	gradient := make([]float64, len(nn.PackedWts()))
	for n := range sample_x {
		floats.Add(gradient, nn.Gradient(sample_x[n], sample_t[n]))
	}
	return gradient
}

func HiddenSample(nn NeuralNetwork, sample_x XSample) [][]float64 {
	result := make([][]float64, len(sample_x))
	for i, xv := range sample_x {
		result[i] = nn.Hidden(xv)
	}
	return result
}
