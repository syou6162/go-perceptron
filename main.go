package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

var str2id = make(map[string]int)

func GetId(str string) int {
	id, ok := str2id[str]
	if ok {
		return id
	} else {
		l := len(str2id)
		str2id[str] = l
		return l
	}
}

type Instance struct {
	label    int
	features map[int]float64
}

type Model struct {
	weight    map[int]float64
	cumWeight map[int]float64
	count     int
}

func ParseLine(line string) (*Instance, error) {
	tmp := strings.Split(strings.TrimSpace(line), " ")
	if len(tmp) < 2 {
		return nil, errors.New("Invalid line")
	}

	label, err := strconv.ParseInt(tmp[0], 10, 32)
	if err != nil {
		return nil, err
	}

	features := make(map[int]float64)
	for _, v := range tmp[1:] {
		tmp := strings.Split(v, ":")
		n, err := strconv.ParseFloat(tmp[1], 64)
		if err != nil {
			return nil, err
		}
		features[GetId(tmp[0])] = n
	}
	return &Instance{int(label), features}, nil
}

func GetAccuracy(gold []int, predict []int) float64 {
	if len(gold) != len(predict) {
		return 0.0
	}
	sum := 0.0
	for i, v := range gold {
		if v == predict[i] {
			sum += 1.0
		}
	}
	return sum / float64(len(gold))
}

func (model *Model) Learn(instance Instance) {
	predict := model.predictForTraining(instance.features)
	if instance.label != predict {
		for k, v := range instance.features {
			w, _ := model.weight[k]
			cumW, _ := model.cumWeight[k]
			model.weight[k] = w + float64(instance.label)*v
			model.cumWeight[k] = cumW + float64(model.count)*float64(instance.label)*v
		}
		model.count += 1
	}
}

func (model *Model) predictForTraining(features map[int]float64) int {
	result := 0.0
	for k, v := range features {
		w, ok := model.weight[k]
		if ok {
			result = result + w*v
		}
	}
	if result > 0 {
		return 1
	}
	return -1
}

func (model Model) Predict(features map[int]float64) int {
	result := 0.0
	for k, v := range features {
		w, ok := model.weight[k]
		if ok {
			result = result + w*v
		}

		w, ok = model.cumWeight[k]
		if ok {
			result = result - w*v/float64(model.count)
		}

	}
	if result > 0 {
		return 1
	}
	return -1
}

func Readln(r *bufio.Reader) (string, error) {
	var (
		isPrefix bool  = true
		err      error = nil
		line, ln []byte
	)
	for isPrefix && err == nil {
		line, isPrefix, err = r.ReadLine()
		ln = append(ln, line...)
	}
	return string(ln), err
}

func ReadData(r *bufio.Reader) []Instance {
	result := make([]Instance, 0)
	s, e := Readln(r)
	for e == nil {
		instance, err := ParseLine(s)
		if err == nil { // skip invalid line
			result = append(result, *instance)
		}
		s, e = Readln(r)
	}
	return result
}

func ExtractGoldLabels(data []Instance) []int {
	golds := make([]int, 0, 0)
	for _, instance := range data {
		golds = append(golds, instance.label)
	}
	return golds
}

func main() {
	bio := bufio.NewReader(os.Stdin)
	data := ReadData(bio)

	n := int(float64(len(data)) * float64(0.8))
	train := data[:n]
	test := data[n+1:]
	model := Model{make(map[int]float64), make(map[int]float64), 1}

	trainGolds := ExtractGoldLabels(train)
	testGolds := ExtractGoldLabels(test)

	for iter := 0; iter < 10; iter++ {
		for _, instance := range train {
			model.Learn(instance)
		}

		trainPredicts := make([]int, 0, 0)
		for _, instance := range train {
			trainPredicts = append(trainPredicts, model.Predict(instance.features))
		}
		testPredicts := make([]int, 0, 0)
		for _, instance := range test {
			testPredicts = append(testPredicts, model.Predict(instance.features))
		}

		fmt.Printf("%d\t%0.3f\t%0.3f\n", iter, GetAccuracy(trainGolds, trainPredicts), GetAccuracy(testGolds, testPredicts))
	}
}
