package neuralnet

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"math"
	"os"
)

// FIXME replace with proper types!!!
type WeightVector []float64
type Vector []float64
type Sample []Vector

type NNOrder struct {
	D, M, K int
}

type NNProcessing struct {
	H     func(float64) float64
	Sigma func(float64) float64
}

func (order *NNOrder) ExpectedPackedWeightsCount() int {
	return (order.M*order.D + order.K*order.M)
}

func (order *NNOrder) ForWeights(wts []float64) *SingleLayerNN {
	if len(wts) != order.ExpectedPackedWeightsCount() {
		panic(fmt.Sprintf("invalid length of weights %d != %d", len(wts), order.ExpectedPackedWeightsCount()))
	}
	return &SingleLayerNN{order, &NNProcessing{math.Tanh, func(x float64) float64 { return x }}, wts}
}

type SingleLayerNN struct {
	order *NNOrder
	proc  *NNProcessing
	wts   []float64
}

func (nn *SingleLayerNN) dm(d int, m int) int {
	if !(d < nn.order.D && m < nn.order.M) {
		panic(fmt.Sprintf("invalid indexes %d %d", d, m))
	}
	return m + d*nn.order.M
}

func (nn *SingleLayerNN) mk(m int, k int) int {
	if !(m < nn.order.M && k < nn.order.K) {
		panic(fmt.Sprintf("invalid indexes %d %d", m, k))
	}
	return nn.order.M*nn.order.D + m + k*nn.order.M
}

func (nn *SingleLayerNN) a_j(x []float64) []float64 {
	if len(x) != nn.order.D {
		panic(fmt.Sprintf("invalid length of x: %d != %d", len(x), nn.order.D))
	}
	a_j := make([]float64, nn.order.M)
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
	return mapOverVector(a_j, nn.proc.H)
}

func (nn *SingleLayerNN) a_k(z_j []float64) []float64 {
	a_k := make([]float64, nn.order.K)
	for k := range a_k {
		for m := range z_j {
			a_k[k] += nn.wts[nn.mk(m, k)] * (z_j)[m]
		}
	}
	return a_k
}

func (nn *SingleLayerNN) z_k(a_k []float64) []float64 {
	return mapOverVector(a_k, nn.proc.Sigma)
}

func (nn *SingleLayerNN) Hidden(x []float64) []float64 {
	return nn.z_j(nn.a_j(x))
}

func (nn *SingleLayerNN) Predict(x []float64) []float64 {
	a_j := nn.a_j(x)
	z_j := nn.z_j(a_j)
	a_k := nn.a_k(z_j)
	y_k := nn.z_k(a_k)
	return y_k
}

func (nn *SingleLayerNN) ErfSampleValue(x [][]float64, t [][]float64) float64 {
	value := 0.0
	for i := range x {
		value += nn.ErfValue(x[i], t[i])
	}
	return value
}

func (nn *SingleLayerNN) ErfValue(x []float64, t []float64) float64 {
	if len(t) != nn.order.K {
		panic(fmt.Sprintf("invalid length of t: %d != %d", len(t), nn.order.K))
	}

	// TODO implement other error functions
	y := nn.Predict(x)
	return ssqdiff(y, t) / 2
}

func (nn *SingleLayerNN) PredictSample(sample_x [][]float64) [][]float64 {
	return mapOverSample(nn.Predict, sample_x)
}

func (nn *SingleLayerNN) HiddenSample(sample_x [][]float64) [][]float64 {
	return mapOverSample(nn.Hidden, sample_x)
}

func (nn *SingleLayerNN) GradientSample(sample_x [][]float64, sample_t [][]float64) []float64 {
	gradient := make([]float64, nn.order.ExpectedPackedWeightsCount())
	for n := range sample_x {
		floats.Add(gradient, nn.Gradient(sample_x[n], sample_t[n]))
	}
	return gradient
}

func h_prim(x float64) float64 { //TODO implement other derivatives...
	return 1 - math.Pow(math.Tanh(x), 2)
}

func (nn *SingleLayerNN) Gradient(x []float64, t []float64) []float64 {
	gradient := make([]float64, nn.order.ExpectedPackedWeightsCount())

	// forward...
	a_j := nn.a_j(x)
	z_j := nn.z_j(a_j)
	a_k := nn.a_k(z_j)
	y := nn.z_k(a_k)

	delta_k := make([]float64, nn.order.K)
	for k := range delta_k {
		delta_k[k] = y[k] - t[k]
	}

	delta_j := make([]float64, nn.order.M)
	for j := range delta_j {
		for k := range delta_k {
			delta_j[j] += nn.wts[nn.mk(j, k)] * delta_k[k]
		}
		delta_j[j] *= h_prim(a_j[j])
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

func (order NNOrder) FitByCG(sample_x [][]float64, sample_t [][]float64, w0 []float64) (*SingleLayerNN, []float64) {
	eta := 1.0

	for tries := 0; tries < 1000; tries++ {
		n0 := order.ForWeights(w0)

		gradient := n0.GradientSample(sample_x, sample_t)

		ErfValueW0 := n0.ErfSampleValue(sample_x, sample_t)
		for et := 0; et <= 15; et++ {
			if order.ForWeights(perturbed(w0, gradient, -eta)).ErfSampleValue(sample_x, sample_t) < ErfValueW0 {
				break
			}
			eta /= 2
		}

		os.Stderr.WriteString(fmt.Sprintf("decreasing eta to %f...\n", eta))

		for et := 0; et <= 15; et++ {
			if order.ForWeights(perturbed(w0, gradient, -2*eta)).ErfSampleValue(sample_x, sample_t) >= ErfValueW0 {
				break
			}
			eta *= 2
		}

		os.Stderr.WriteString(fmt.Sprintf("increasing eta to %f...\n", eta))

		w1 := perturbed(w0, gradient, -eta)
		E_new := order.ForWeights(w1).ErfSampleValue(sample_x, sample_t)

		if E_new > ErfValueW0 || eta < 1e-15 {
			os.Stderr.WriteString(fmt.Sprintf("found the best error funciton... %f\n", ErfValueW0))
			break
		}

		os.Stderr.WriteString(fmt.Sprintf("%f -> %f\n", ErfValueW0, E_new))
		w0 = w1
	}
	best_nn := order.ForWeights(w0)
	return best_nn, best_nn.wts
}

func mapOverSample(f func([]float64) []float64, sample_x [][]float64) [][]float64 {
	result := make([][]float64, len(sample_x))
	for i, xv := range sample_x {
		result[i] = f(xv)
	}
	return result
}

func ssqdiff(a []float64, b []float64) float64 {
	ssq := 0.0
	for i := range a {
		ssq += math.Pow(a[i]-b[i], 2)
	}
	return ssq
}
