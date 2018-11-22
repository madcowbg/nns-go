package neuralnet

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"math"
	"os"
)

type WeightVector []float64
type XVector []float64
type YVector []float64
type XSample []XVector
type YSample []YVector

type NNOrder struct {
	D, M, K int
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
	return (order.M*order.D + order.K*order.M)
}

func (order *NNStructure) ForWeights(wts WeightVector) *SingleLayerNN {
	if len(wts) != order.ExpectedPackedWeightsCount() {
		panic(fmt.Sprintf("invalid length of weights %d != %d", len(wts), order.ExpectedPackedWeightsCount()))
	}
	return &SingleLayerNN{order, wts}
}

type SingleLayerNN struct {
	structure *NNStructure
	wts       []float64
}

func (nn *SingleLayerNN) dm(d int, m int) int {
	if !(d < nn.structure.D && m < nn.structure.M) {
		panic(fmt.Sprintf("invalid indexes %d %d", d, m))
	}
	return m + d*nn.structure.M
}

func (nn *SingleLayerNN) mk(m int, k int) int {
	if !(m < nn.structure.M && k < nn.structure.K) {
		panic(fmt.Sprintf("invalid indexes %d %d", m, k))
	}
	return nn.structure.M*nn.structure.D + m + k*nn.structure.M
}

func (nn *SingleLayerNN) a_j(x XVector) []float64 {
	if len(x) != nn.structure.D {
		panic(fmt.Sprintf("invalid length of x: %d != %d", len(x), nn.structure.D))
	}
	a_j := make([]float64, nn.structure.M)
	for m := range a_j {
		for d, xv := range x {
			a_j[m] += nn.wts[nn.dm(d, m)] * xv
		}
	}
	return a_j
}

func mapOverVector(vs []float64, f func(float64) float64) []float64 {
	vsm := make([]float64, len(vs))
	for i, v := range vs {
		vsm[i] = f(v)
	}
	return vsm
}

func (nn *SingleLayerNN) z_j(a_j []float64) []float64 {
	return mapOverVector(a_j, nn.structure.H)
}

func (nn *SingleLayerNN) a_k(z_j []float64) []float64 {
	a_k := make([]float64, nn.structure.K)
	for k := range a_k {
		for m := range z_j {
			a_k[k] += nn.wts[nn.mk(m, k)] * (z_j)[m]
		}
	}
	return a_k
}

func (nn *SingleLayerNN) z_k(a_k []float64) []float64 {
	return mapOverVector(a_k, nn.structure.Sigma)
}

func (nn *SingleLayerNN) Hidden(x XVector) []float64 {
	return nn.z_j(nn.a_j(x))
}

func (nn *SingleLayerNN) Predict(x XVector) YVector {
	a_j := nn.a_j(x)
	z_j := nn.z_j(a_j)
	a_k := nn.a_k(z_j)
	y_k := nn.z_k(a_k)
	return y_k
}

func (nn *SingleLayerNN) ErfSampleValue(x XSample, t YSample) float64 {
	value := 0.0
	for i := range x {
		value += nn.ErfValue(x[i], t[i])
	}
	return value
}

func (nn *SingleLayerNN) ErfValue(x XVector, t YVector) float64 {
	if len(t) != nn.structure.K {
		panic(fmt.Sprintf("invalid length of t: %d != %d", len(t), nn.structure.K))
	}

	return nn.structure.ErrorFunction(nn.Predict(x), t)
}

func (nn *SingleLayerNN) PredictSample(sample_x XSample) YSample {
	result := make(YSample, len(sample_x))
	for i, xv := range sample_x {
		result[i] = nn.Predict(xv)
	}
	return result
}

func (nn *SingleLayerNN) HiddenSample(sample_x XSample) [][]float64 {
	result := make([][]float64, len(sample_x))
	for i, xv := range sample_x {
		result[i] = nn.Hidden(xv)
	}
	return result
}

func (nn *SingleLayerNN) GradientSample(sample_x XSample, sample_t YSample) []float64 {
	gradient := make([]float64, nn.structure.ExpectedPackedWeightsCount())
	for n := range sample_x {
		floats.Add(gradient, nn.Gradient(sample_x[n], sample_t[n]))
	}
	return gradient
}

func (nn *SingleLayerNN) Gradient(x XVector, t YVector) WeightVector {
	gradient := make([]float64, nn.structure.ExpectedPackedWeightsCount())

	// forward...
	a_j := nn.a_j(x)
	z_j := nn.z_j(a_j)
	a_k := nn.a_k(z_j)
	y := nn.z_k(a_k)

	delta_k := make([]float64, nn.structure.K)
	for k := range delta_k {
		delta_k[k] = y[k] - t[k] // Assuming canonical link function is used...
	}

	delta_j := make([]float64, nn.structure.M)
	for j := range delta_j {
		for k := range delta_k {
			delta_j[j] += nn.wts[nn.mk(j, k)] * delta_k[k]
		}
		delta_j[j] *= nn.structure.H_prim(a_j[j])
	}

	for j, dj := range delta_j {
		for i, xi := range x {
			gradient[nn.dm(i, j)] = dj * xi
		}
	}

	for k, dk := range delta_k {
		for j, zj := range z_j {
			gradient[nn.mk(j, k)] = dk * zj
		}
	}

	return gradient
}

func perturbed(w []float64, p []float64, eta float64) []float64 {
	result := make([]float64, len(w))
	for i := range result {
		result[i] = w[i] + p[i]*eta
	}
	return result
}

func (order NNStructure) FitByCG(sampleX XSample, sampleT YSample, w0 WeightVector, verbose bool) (*SingleLayerNN, []float64) {
	eta := 1.0
	MIN_ERF_IMPROVEMENT := 1e-12

	for tries := 0; tries < 10000; tries++ {
		n0 := order.ForWeights(w0)

		gradient := n0.GradientSample(sampleX, sampleT)

		ErfValueW0 := n0.ErfSampleValue(sampleX, sampleT)
		for et := 0; et <= 15; et++ {
			if order.ForWeights(perturbed(w0, gradient, -eta)).ErfSampleValue(sampleX, sampleT) < ErfValueW0 {
				break
			}
			eta /= 2
		}

		if verbose {
			os.Stderr.WriteString(fmt.Sprintf("decreasing eta to %f...\n", eta))
		}

		for et := 0; et <= 15; et++ {
			if order.ForWeights(perturbed(w0, gradient, -2*eta)).ErfSampleValue(sampleX, sampleT) >= ErfValueW0 {
				break
			}
			eta *= 2
		}

		if verbose {
			os.Stderr.WriteString(fmt.Sprintf("increasing eta to %f...\n", eta))
		}

		w1 := perturbed(w0, gradient, -eta)
		E_new := order.ForWeights(w1).ErfSampleValue(sampleX, sampleT)

		if E_new-ErfValueW0 > -MIN_ERF_IMPROVEMENT || eta < 1e-15 {
			os.Stderr.WriteString(fmt.Sprintf("found the best error funciton... %f\n", ErfValueW0))
			break
		}

		if verbose {
			os.Stderr.WriteString(fmt.Sprintf("%f -> %f\n", ErfValueW0, E_new))
		}
		w0 = w1
	}
	best_nn := order.ForWeights(w0)

	os.Stderr.WriteString(fmt.Sprintf("could not optimize error function beyond beyond %f...\n", best_nn.ErfSampleValue(sampleX, sampleT)))

	return best_nn, best_nn.wts
}

func ssqdiff(a YVector, b YVector) float64 {
	ssq := 0.0
	for i := range a {
		ssq += math.Pow(a[i]-b[i], 2)
	}
	return ssq
}

func crossentropy(y YVector, t YVector) float64 {
	result := 0.0
	for i := range t {
		result += -(t[i]*math.Log(y[i]) + (1-t[i])*math.Log(1-y[i]))
	}
	return result
}
