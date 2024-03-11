package main

import "fmt"
import "math/rand"
import "math/bits"

type Layer struct {
	inputsize int
	outputpattern []int
	weights [][]uint64 // 1 = reinforcing, 0 = discouraging
	previous *Layer
}

type State struct {
	State []uint64
	Input []*State
	Layer *Layer
	Offset int
}

func NewLayer(inputsize int, outputpattern []int) *Layer {
	result := new(Layer)
	result.inputsize = inputsize
	result.outputpattern = outputpattern
	outputsize := 0
	for _, size := range outputpattern {
		outputsize += size
	}
	result.weights = make([][]uint64, 64 * outputsize)
	for i, _ := range result.weights {
		result.weights[i] = make([]uint64, inputsize)
	}
	result.RandomizeWeights()
	return result
}

func InputState(data []uint64) *State {
	result := new(State)
	result.State = data
	result.Input = nil
	result.Layer = nil
	return result
}

func NewState(size int) *State {
	result := new(State)
	result.State = make([]uint64, size)
	result.Input = nil
	result.Layer = nil
	return result
}

func (l *Layer) Compute(input []*State) []*State {
	result := make([]*State, len(l.outputpattern))
	inp_offset := 0
	for i, sz := range l.outputpattern {
		st := new(State)
		st.State = make([]uint64, sz)
		st.Input = input
		st.Layer = l
		st.Offset = inp_offset
		result[i] = st
		inp_offset += sz
	}
	oi := 0
	oj := 0

	m := uint64(0)
	for i, w := range l.weights {
		sum := 0
		ii := 0 // input pointer
		ij := 0
		for j := 0; j < l.inputsize; j++ {
			for ij >= len(input[ii].State) {
				// move to next input array
				ij = 0
				ii++
			}
			w := input[ii].State[ij] ^ w[j]
			ij++
			sum -= 32 - bits.OnesCount64(w)
		}
		v := uint64(int64(sum)) >> 63 // 1 or 0
		m = m | (v << (i % 64))
		if i % 64 == 63 {
			// write a uint64
			for oj >= len(result[oi].State) {
				oj = 0
				oi++
			}
			result[oi].State[oj] = m
			m = 0
			oj++
		}
	}
	return result
}

func (l *Layer) RandomizeWeights() {
	for i, w := range l.weights {
		for j, _ := range w {
			l.weights[i][j] = rand.Uint64()
		}
	}
}

func (s *State) Cleanup(maxdepth int) {
	if maxdepth <= 1 {
		// forget previous data
		s.Input = nil
		s.Layer = nil
	} else {
		for _, in := range s.Input {
			in.Cleanup(maxdepth - 1)
		}
	}
}

func (s *State) Reinforce(idx int, val uint64) {
	if s.Input == nil {
		return // cannot reinforce history that I forgot
	}
	idx += s.Offset // offset in source state
	// iterate through all 64 bits
	for i := 64 * idx; i < 64 * idx + 64; i++ {
		w := s.Layer.weights[i]
		sum := 0
		ii := 0 // input pointer
		ij := 0
		for j := 0; j < s.Layer.inputsize; j++ {
			for ij >= len(s.Input[ii].State) {
				// move to next input array
				ij = 0
				ii++
			}
			w := s.Input[ii].State[ij] ^ w[j]
			ij++
			sum -= 32 - bits.OnesCount64(w)
		}
		v := uint64(int64(sum)) >> 63 // 1 or 0
		if ((val >> (i % 64)) & 1) != v {
			//fmt.Println("need correction in bit", i)
			ii := 0 // input pointer
			ij := 0
			for j := 0; j < s.Layer.inputsize; j++ {
				for ij >= len(s.Input[ii].State) {
					// move to next input array
					ij = 0
					ii++
				}
				changemask := s.Input[ii].State[ij] ^ w[j] // changemask is a bit vector that shows a 1 wherever either weight or previous state has to change
				if v == 0 {
					// v = 0 but should be 1; it becomes one when w = 0x00
				} else {
					// v = 1 but should be 0; it becomes one when w = 0xFF
					changemask = ^changemask // negate w so we know the exact bitmask where w is not good at yet
				}
				changemask = changemask & rand.Uint64() // do not change everything in every cycle!
				if s.Input[ii].Input == nil {
					// pure weight change since it is a hard input node
					w[j] = w[j] ^ changemask // the bits that need to be flipped
				} else {
					// split the change between weight and reinforcement
					reinforcemask := rand.Uint64() // if the mask is 1, flip the weight; if it is 0, reinforce the parent's state
					w[j] = w[j] ^ (reinforcemask & changemask) // the bits that need to be flipped
					betterstate := s.Input[ii].State[ij] ^ (changemask & (^reinforcemask)) // the desired state for the previous layer
					s.Input[ii].Reinforce(ij, betterstate) // recurse over the whole network
				}
			}
		}
	}
}

func main() {
	fmt.Println("Hello World")
	networksize := 16
	layer := NewLayer(networksize, []int{networksize, 1})

	for it := 0; it < 10000; it++ {
		s := "Hello I am Alex"
		s2 := []byte{}

		// start with a new state
		state := NewState(networksize)
		for _, c := range s {
			// at first get the networks prediction
			//input := InputState([]uint64{uint64(c)})
			//o := layer.Compute([]*State{input, state})
			o := layer.Compute([]*State{state}) // compute ohne input
			state = o[0] // next state
			state.Cleanup(2)

			// read out what character the network would print
			outbyte := o[1].State[0]
			s2 = append(s2, byte(outbyte & 0xff))
			o[1].Reinforce(0, uint64(c))
		}
		fmt.Println(string(s2))
	}
}
