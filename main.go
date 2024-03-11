package main

import "fmt"
import "math/rand"
import "math/bits"

type Layer struct {
	state, laststate []uint64 // 1 = reinforced, 0 = discouraged
	weights [][]uint64 // 1 = reinforcing, 0 = discouraging
	previous *Layer
}

func New(prevlayersize, statesize int) *Layer {
	result := new(Layer)
	result.state = make([]uint64, statesize)
	result.laststate = make([]uint64, statesize)
	result.weights = make([][]uint64, 64 * statesize)
	for i, _ := range result.weights {
		result.weights[i] = make([]uint64, prevlayersize)
	}
	return result
}

func (l *Layer) SetPrevious(p *Layer) {
	l.previous = p
}

func (l *Layer) SetState(i int, v uint64) {
	l.state[i] = v
}

func (l *Layer) GetState(i int) uint64 {
	return l.state[i]
}

func (l *Layer) Reinforce(idx int, val uint64, maxdepth int) {
	l.ReinforceEx(idx, val, maxdepth, l, 0)
}
func (l *Layer) ReinforceEx(idx int, val uint64, maxdepth int, l2 *Layer, xdepth int) {
	if maxdepth < 1 {
		return
	}
	ps := l.previous.state
	if l.previous == l2 {
		xdepth++
	}
	if xdepth > 1 {
		ps = l.previous.laststate
	}
	// iterate through all 64 bits
	for i := 64 * idx; i < 64 * idx + 64; i++ {
		w := l.weights[i]
		sum := 0
		for j := 0; j < len(ps); j++ {
			w := ps[j] ^ w[j]
			sum -= 32 - bits.OnesCount64(w)
		}
		v := uint64(int64(sum)) >> 63 // 1 or 0
		if ((val >> (i % 64)) & 1) != v {
			//fmt.Println("need correction in bit", maxdepth, i)
			for j := 0; j < len(ps); j++ {
				changemask := ps[j] ^ w[j]
				if v == 0 {
					// v = 0 but should be 1; it becomes one when w = 0x00
				} else {
					// v = 1 but should be 0; it becomes one when w = 0xFF
					changemask = ^changemask // negate w so we know the exact bitmask where w is not good at yet
				}
				changemask = changemask & rand.Uint64() // do not change everything in every cycle!
				reinforcemask := rand.Uint64() // if the mask is 1, flip the weight; if it is 0, reinforce the parent's state
				w[j] = w[j] ^ (reinforcemask & changemask) // the bits that need to be flipped
				betterstate := ps[j] ^ (changemask & (^reinforcemask)) // the desired state for the previous layer
				l.previous.ReinforceEx(j, betterstate, maxdepth - 1, l2, xdepth)
			}
		}
	}
	// insert the desired state into our state so the rest of the network can learn it better
	//l.state[idx] = val
}

func (l *Layer) Proceed() {
	// double bufferec
	l.state, l.laststate = l.laststate, l.state
	m := uint64(0)
	for i, w := range l.weights {
		sum := 0
		for j := 0; j < len(l.previous.state); j++ {
			w := l.previous.state[j] ^ w[j]
			sum -= 32 - bits.OnesCount64(w)
		}
		v := uint64(int64(sum)) >> 63 // 1 or 0
		m = m | (v << (i % 64))
		if i % 64 == 63 {
			l.state[i / 64] = m
			m = 0
		}
	}
}

func (l *Layer) RandomizeWeights() {
	for i, w := range l.weights {
		for j, _ := range w {
			l.weights[i][j] = rand.Uint64()
		}
	}
}

func (l *Layer) Reset() {
	for i, _ := range l.state {
		l.state[i] = 0
		l.laststate[i] = 0
	}
}

func main() {
	fmt.Println("Hello World")
	networksize := 8
	layers := []*Layer{New(networksize, networksize), New(networksize, networksize)}
	layers[0].SetPrevious(layers[1])
	layers[1].SetPrevious(layers[0])
	layers[0].RandomizeWeights()
	layers[1].RandomizeWeights()

	for it := 0; it < 10000; it++ {
		s := "Hello I am Alex"
		s2 := []byte{}
		layers[0].Reset()
		layers[1].Reset()
		for i, c := range s {
			// at first get the networks prediction
			layers[0].Proceed()
			layers[1].Proceed()

			// read out what character the network would print
			state := layers[1].GetState(1)
			s2 = append(s2, byte(state & 0xff))

			// compare state to the result and learn
			state = (state & uint64(^uint64(0xff))) | uint64(c)
			layers[1].Reinforce(1, state, 2) // learn how it should work

			/*
			// then simulate the result
			input := (uint64(c) << 24) | (uint64(c) << 16) | (uint64(c)) // character as input
			layers[0].SetState(0, input)
			layers[1].SetState(0, input)
			*/
			if i >= it/1000 {
				break
			}
		}
		fmt.Println(string(s2))
	}
}
