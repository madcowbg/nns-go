package neuralnet

import "fmt"

type MultiLayerNN struct {
	structure *NNStructure
	wts       WeightVector
	L         []int
}

func (nn *MultiLayerNN) PackedWts() []float64 {
	return nn.wts
}

func (nn *MultiLayerNN) wt_idx(layer int, j int, i int) int {
	if layer >= len(nn.L)-1 {
		panic(fmt.Sprintf("layer index too high: %d >= %d", layer, len(nn.L)-1))
	}
	if i >= nn.L[layer] {
		panic(fmt.Sprintf("index out of bounds for layer %d: %d >= %d", layer, i, nn.L[layer]))
	}
	if j >= nn.L[layer+1] {
		panic(fmt.Sprintf("index out of bounds for layer %d: %d >= %d", layer+1, j, nn.L[layer+1]))
	}

	offset := 0
	for l := 0; l < layer; l++ {
		offset += nn.L[l] * nn.L[l+1]
	}
	result := offset + i + j*nn.L[layer]
	if result >= nn.structure.ExpectedPackedWeightsCount() {
		panic(fmt.Sprintf("can't access index %d >= %d", result, nn.structure.ExpectedPackedWeightsCount()))
	}
	return result
}

func (nn *MultiLayerNN) Predict(x XVector) YVector {
	_, z := nn.fwdPropHidden(x)

	a_k := nn.a_j(len(nn.L)-2, z[len(nn.L)-2])
	y_k := mapOverVector(a_k, nn.structure.Sigma)

	return y_k
}

func (nn *MultiLayerNN) z_j(l int, layer_next_a []float64) XVector {
	layer_z := make([]float64, nn.L[l+1])
	for j := range layer_next_a {
		layer_z[j] = nn.structure.H(layer_next_a[j])
	}
	return layer_z
}

func (nn *MultiLayerNN) a_j(l int, layer_z XVector) []float64 {
	layer_next_a := make([]float64, nn.L[l+1])
	for j := range layer_next_a {
		for i := range layer_z {
			layer_next_a[j] += nn.wts[nn.wt_idx(l, j, i)] * layer_z[i]
		}
	}
	return layer_next_a
}

func (nn *MultiLayerNN) ErfValue(x XVector, t YVector) float64 {
	if len(t) != nn.structure.K {
		panic(fmt.Sprintf("invalid length of t: %d != %d", len(t), nn.structure.K))
	}

	return nn.structure.ErrorFunction(nn.Predict(x), t)
}

func (nn *MultiLayerNN) Gradient(x XVector, t YVector) WeightVector {
	gradient := make([]float64, nn.structure.ExpectedPackedWeightsCount())

	a, z := nn.fwdPropHidden(x)
	a_k := nn.a_j(len(nn.L)-2, z[len(nn.L)-2])
	y := mapOverVector(a_k, nn.structure.Sigma)
	delta_k := make([]float64, nn.structure.K)
	for k := range delta_k {
		delta_k[k] = y[k] - t[k] // Assuming canonical link function is used...
	}

	delta_j := make([][]float64, len(nn.L))
	delta_j[len(nn.L)-1] = delta_k
	for l := len(nn.L) - 2; l >= 1; l-- { // backprop
		delta_j[l] = make([]float64, nn.L[l])
		for j := range delta_j[l] {
			for k := range delta_j[l+1] {
				delta_j[l][j] += nn.wts[nn.wt_idx(l, k, j)] * delta_j[l+1][k]
			}
			delta_j[l][j] *= nn.structure.H_prim(a[l][j])
		}
	}

	for l := 0; l < len(nn.L)-1; l++ {
		for j, dj := range delta_j[l+1] {
			for i, zi := range z[l] {
				gradient[nn.wt_idx(l, j, i)] = dj * zi
			}
		}
	}
	return gradient
}

func (nn *MultiLayerNN) fwdPropHidden(x XVector) ([][]float64, [][]float64) {
	a := make([][]float64, len(nn.L)-1)
	z := make([][]float64, len(nn.L)-1)
	z[0] = x
	for l := 0; l < len(nn.L)-2; l++ { // hidden layers
		a[l+1] = nn.a_j(l, z[l])
		z[l+1] = nn.z_j(l, a[l+1])
	}
	return a, z
}

func (nn *MultiLayerNN) Hidden(x XVector) []float64 {
	_, z := nn.fwdPropHidden(x)
	hiddenCnt := 0
	for _, l := range nn.structure.M {
		hiddenCnt += l
	}
	z_flat := make([]float64, hiddenCnt)
	i := 0
	for _, zv := range z[1:] {
		for _, v := range zv {
			z_flat[i] = v
			i++
		}
	}
	return z_flat
}
