package neuralnet

func ArrayOfSize(nRepeat int, value float64) []float64 {
	result := make([]float64, nRepeat)
	for i := range result {
		result[i] = value
	}
	return result
}
