package neuralnet

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"math/rand"
	"testing"
)

func TestSingleLayerNetwork(t *testing.T) {
	structure := NNOrder{2, []int{4}, 3}.OfResponseType(Regression)
	w0 := ArrayOfSize(structure.ExpectedPackedWeightsCount(), 1.0)

	sample_x := XSample{{1, 1}}
	sample_t := YSample{{1, 2, 3}}

	nn := structure.SNForWeights(w0)
	best_nn := FitByCG(structure.SNForWeights, sample_x, sample_t, w0, false)

	expectTemplateNetwork(t, nn, best_nn, sample_x, sample_t)
}

func TestMLNWithSingleLayer(t *testing.T) {
	structure := NNOrder{2, []int{4}, 3}.OfResponseType(Regression)
	w0 := ArrayOfSize(structure.ExpectedPackedWeightsCount(), 1.0)

	sample_x := XSample{{1, 1}}
	sample_t := YSample{{1, 2, 3}}

	nn := structure.ForWeights(w0)
	best_nn := FitByCG(structure.ForWeights, sample_x, sample_t, w0, false)

	expectTemplateNetwork(t, nn, best_nn, sample_x, sample_t)
}

func TestMultiLayerNetwork(t *testing.T) {
	structure := NNOrder{2, []int{3, 2}, 3}.OfResponseType(Regression)
	w0 := ArrayOfSize(structure.ExpectedPackedWeightsCount(), 1.0)

	sample_x := XSample{{1, 1}, {1, 2}, {2, 1}}
	sample_t := YSample{{1, 2, 3}, {3, 2, 3}, {3, 2, 1}}

	nn := structure.ForWeights(w0)

	ExpectNN(
		t, sample_x, sample_t, nn,
		[]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		[][]float64{{1.9877342232748791, 1.9877342232748791, 1.9877342232748791}, {1.9898124033061484, 1.9898124033061484, 1.9898124033061484}, {1.9898124033061484, 1.9898124033061484, 1.9898124033061484}},
		3.020912,
		[]float64{0 - 0.000482985965138744, -0.0008840068279246431, -0.000482985965138744, -0.0008840068279246431, -0.000482985965138744, -0.0008840068279246431, -0.02127463868926193, -0.02127463868926193, -0.02127463868926193, -0.02127463868926193, -0.02127463868926193, -0.02127463868926193, -1.028407250015801, -1.028407250015801, -0.03246195834709224, -0.03246195834709224, -1.026329069984532, -1.026329069984532},
		[][]float64{{0.9640275800758169, 0.9640275800758169, 0.9640275800758169, 0.9938671116374396, 0.9938671116374396}, {0.9950547536867305, 0.9950547536867305, 0.9950547536867305, 0.9949062016530742, 0.9949062016530742}, {0.9950547536867305, 0.9950547536867305, 0.9950547536867305, 0.9949062016530742, 0.9949062016530742}})

	best_nn := FitByCG(structure.ForWeights, sample_x, sample_t, w0, false)
	ExpectNN(
		t, sample_x, sample_t, best_nn,
		[]float64{-1.3630863383661246e-05, 0.43681311195274314, -1.3630863383661246e-05, 0.43681311195274314, -1.3630863383661246e-05, 0.43681311195274314, 0.6035930751398187, 0.6035930751398187, 0.6035930751398187, 0.6035930751398187, 0.6035930751398187, 0.6035930751398187, 1.6652442959580869, 1.6652442959580869, 1.385667784689702, 1.385667784689702, 1.6652536030673724, 1.6652536030673724},
		[][]float64{{2.1038068972494903, 1.7506004673322657, 2.10381865550067}, {2.846577647639609, 2.368668040194433, 2.846593557262792}, {2.103765837947448, 1.7505663014430686, 2.103777595969146}},
		2.175248,
		[]float64{6.34592946206975e-05, 5.475564025763297e-05, 6.34592946206975e-05, 5.475564025763297e-05, 6.34592946206975e-05, 5.475564025763297e-05, 1.7232581634985614e-05, 1.7232581634985614e-05, 1.7232581634985614e-05, 1.7232581634985614e-05, 1.7232581634985614e-05, 1.7232581634985614e-05, 3.3384292219551526e-07, 3.3384292219551526e-07, 1.0964644797384349e-06, 1.0964644797384349e-06, 4.1298589130711605e-06, 4.1298589130711605e-06},
		[][]float64{{0.410988029888952, 0.410988029888952, 0.410988029888952, 0.6316811600423706, 0.6316811600423706}, {0.7032049498664709, 0.7032049498664709, 0.7032049498664709, 0.8547027167571981, 0.8547027167571981}, {0.41097670136706094, 0.41097670136706094, 0.41097670136706094, 0.6316688317305001, 0.6316688317305001}})
}

func TestGradientsInMultiLayerNetworkEqualApproximation(t *testing.T) {
	order := NNOrder{2, []int{3, 2}, 3}
	random_w0 := []float64{0.6046602879796196, 0.9405090880450124, 0.6645600532184904, 0.4377141871869802, 0.4246374970712657, 0.6868230728671094, 0.06563701921747622, 0.15651925473279124, 0.09696951891448456, 0.30091186058528707, 0.5152126285020654, 0.8136399609900968, 0.21426387258237492, 0.380657189299686, 0.31805817433032985, 0.4688898449024232, 0.28303415118044517, 0.29310185733681576}

	single_x := []float64{1, 1}
	single_t := []float64{2, 2, 2}

	RunTestForNNGradients(t, order.OfResponseType(Regression).ForWeights, random_w0, single_x, single_t)

	RunTestForNNGradients(t, order.OfResponseType(BinaryClassifier).ForWeights, random_w0, single_x, single_t)
}

func TestGradientsInSingleLayerNetworkEqualApproximation(t *testing.T) {
	order := NNOrder{2, []int{5}, 3}
	random_w0 := []float64{0.6790846759202163, 0.21855305259276428, 0.20318687664732285, 0.360871416856906, 0.5706732760710226, 0.8624914374478864, 0.29311424455385804, 0.29708256355629153, 0.7525730355516119, 0.2065826619136986, 0.865335013001561, 0.6967191657466347, 0.5238203060500009, 0.028303083325889995, 0.15832827774512764, 0.6072534395455154, 0.9752416188605784, 0.07945362337387198, 0.5948085976830626, 0.05912065131387529, 0.692024587353112, 0.30152268100656, 0.17326623818270528, 0.5410998550087353, 0.544155573000885}

	single_x := []float64{1, 1}
	single_t := []float64{2, 2, 2}

	RunTestForNNGradients(t, order.OfResponseType(Regression).SNForWeights, random_w0, single_x, single_t)

	RunTestForNNGradients(t, order.OfResponseType(BinaryClassifier).SNForWeights, random_w0, single_x, single_t)
}

func RunTestForNNGradients(t *testing.T, builderFun func(WeightVector) NeuralNetwork, w0 []float64, single_x XVector, single_t YVector) {
	nn := builderFun(w0)
	gradient := nn.Gradient(single_x, single_t)
	erf := nn.ErfValue(single_x, single_t)

	delta := 0.000001
	approximation := make([]float64, len(gradient))
	for i := range w0 {
		p0 := make([]float64, len(w0))
		p0[i] = 1

		w1 := perturbed(w0, p0, delta)
		nn1 := builderFun(w1)
		erf2 := nn1.ErfValue(single_x, single_t)

		approximation[i] = (erf2 - erf) / delta
	}

	if !floats.EqualApprox(approximation, gradient, 1e-5) {
		t.Errorf("gradient is not close to numerical approximation: %f > %f: %v != %v", floats.Distance(approximation, gradient, 1), 1e-5, gradient, approximation)
	}
}

func fillRandom(n int) []float64 {
	w0 := make([]float64, n)
	for i := range w0 {
		w0[i] = rand.Float64()
	}
	return w0
}

func expectTemplateNetwork(t *testing.T, nn NeuralNetwork, best_nn NeuralNetwork, sample_x XSample, sample_t YSample) {
	ExpectNN(
		t, sample_x, sample_t, nn,
		[]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		[][]float64{{3.8561103, 3.8561103, 3.8561103}},
		6.167718281704448,
		[]float64{0.39340717544369125, 0.39340717544369125, 0.39340717544369125, 0.39340717544369125, 0.39340717544369125, 0.39340717544369125, 0.39340717544369125, 0.39340717544369125, 2.7533691205115254, 2.7533691205115254, 2.7533691205115254, 2.7533691205115254, 1.7893415404357085, 1.7893415404357085, 1.7893415404357085, 1.7893415404357085, 0.8253139603598916, 0.8253139603598916, 0.8253139603598916, 0.8253139603598916},
		[][]float64{{0.9640275800758169, 0.9640275800758169, 0.9640275800758169, 0.9640275800758169}})

	ExpectNN(
		t, sample_x, sample_t, best_nn,
		[]float64{0.650301311766122, 0.650301311766122, 0.650301311766122, 0.650301311766122, 0.650301311766122, 0.650301311766122, 0.650301311766122, 0.650301311766122, 0.2900640306608268, 0.2900640306608268, 0.2900640306608268, 0.2900640306608268, 0.5801280613216536, 0.5801280613216536, 0.5801280613216536, 0.5801280613216536, 0.8701920919824802, 0.8701920919824802, 0.8701920919824802, 0.8701920919824802},
		[][]float64{{0.9999994748565776, 1.9999989497131552, 2.999998424569732}},
		1.9304292999575734e-12,
		[]float64{-5.484200183404358e-07, -5.484200183404358e-07, -5.484200183404358e-07, -5.484200183404358e-07, -5.484200183404358e-07, -5.484200183404358e-07, -5.484200183404358e-07, -5.484200183404358e-07, -4.5260967504489726e-07, -4.5260967504489726e-07, -4.5260967504489726e-07, -4.5260967504489726e-07, -9.052193500897945e-07, -9.052193500897945e-07, -9.052193500897945e-07, -9.052193500897945e-07, -1.3578290258045056e-06, -1.3578290258045056e-06, -1.3578290258045056e-06, -1.3578290258045056e-06},
		[][]float64{{0.8618782140777406, 0.8618782140777406, 0.8618782140777406, 0.8618782140777406}})
}

func ExpectNN(t *testing.T, sample_x XSample, sample_t YSample, nn NeuralNetwork, w0 WeightVector, predictSample [][]float64, erfSample float64, gradientOfSample []float64, hiddenSample [][]float64) {
	ExpectEqualArrays(t, nn.PackedWts(), w0, 1e-10, "nn packed weights")
	ExpectEqualSampleArrays(t, AsArray(PredictSample(nn, sample_x)), predictSample, 1e-5, "nn prediction")
	if !floats.EqualWithinRel(ErfSampleValue(nn, sample_x, sample_t), erfSample, 1e-5) {
		t.Errorf("error function is different: %f != %f", ErfSampleValue(nn, sample_x, sample_t), erfSample)
	}
	ExpectEqualArrays(t, GradientSample(nn, sample_x, sample_t), gradientOfSample, 1e-5, "nn gradient")
	ExpectEqualSampleArrays(t, HiddenSample(nn, sample_x), hiddenSample, 1e-5, "nn hidden")
}

func AsArray(sample YSample) [][]float64 {
	result := make([][]float64, len(sample))
	for i := range result {
		result[i] = sample[i]
	}
	return result
}

func ExpectEqualSampleArrays(t *testing.T, a [][]float64, b [][]float64, epsilon float64, failMsg string) {
	if len(a) != len(b) {
		t.Errorf("%s - sample arrays of different length: %d != %d :%v %v", failMsg, len(a), len(b), a, b)
		return
	}
	for i := range a {
		ExpectEqualArrays(t, a[i], b[i], epsilon, failMsg+fmt.Sprintf("[%d]", i))
	}
}

func ExpectEqualArrays(t *testing.T, a []float64, b []float64, epsilon float64, failMsg string) {
	if !floats.EqualLengths(a, b) {
		t.Errorf("%s - arrays of different length: %d != %d: %v %v", failMsg, len(a), len(b), a, b)
		return
	}
	if !floats.EqualApprox(a, b, epsilon) {
		t.Errorf("%s - arrays are different by %f > %f: %v %v", failMsg, floats.Distance(a, b, 1), epsilon, a, b)
	}
}
