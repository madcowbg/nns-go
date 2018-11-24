package neuralnet

import (
	"fmt"
	"math"
)

type SingleLayerNN struct {
	structure *NNStructure
	wts       WeightVector
}

func (nn *SingleLayerNN) PackedWts() []float64 {
	return nn.wts
}

func (nn *SingleLayerNN) ExpectedPackedWeightsCount() int {
	return nn.structure.ExpectedPackedWeightsCount()
}

func (nn *SingleLayerNN) dm(d int, m int) int {
	if !(d < nn.structure.D && m < nn.structure.M[0]) {
		panic(fmt.Sprintf("invalid indexes %d %d", d, m))
	}
	return m + d*nn.structure.M[0]
}

func (nn *SingleLayerNN) mk(m int, k int) int {
	if !(m < nn.structure.M[0] && k < nn.structure.K) {
		panic(fmt.Sprintf("invalid indexes %d %d", m, k))
	}
	return nn.structure.M[0]*nn.structure.D + m + k*nn.structure.M[0]
}

func (nn *SingleLayerNN) a_j(x XVector) []float64 {
	if len(x) != nn.structure.D {
		panic(fmt.Sprintf("invalid length of x: %d != %d", len(x), nn.structure.D))
	}
	a_j := make([]float64, nn.structure.M[0])
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

func (nn *SingleLayerNN) ErfValue(x XVector, t YVector) float64 {
	if len(t) != nn.structure.K {
		panic(fmt.Sprintf("invalid length of t: %d != %d", len(t), nn.structure.K))
	}

	return nn.structure.ErrorFunction(nn.Predict(x), t)
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

	delta_j := make([]float64, nn.structure.M[0])
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
