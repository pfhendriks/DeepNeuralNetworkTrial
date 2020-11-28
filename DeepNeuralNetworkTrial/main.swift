//
//  main.swift
//  DeepNeuralNetworkTrial
//
//  Created by Pieter Hendriks on 28/11/2020.
//

import Foundation


let t0 = CFAbsoluteTimeGetCurrent()
/*
// Example 1 - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
var x = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.05, 0.10) )
var y = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.01, 0.99) )

let W1 = Matrix(2, 2, withArray: [Double](arrayLiteral: 0.15, 0.20, 0.25, 0.30) )
let B1 = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.35, 0.35) )
let W2 = Matrix(2, 2, withArray: [Double](arrayLiteral: 0.40, 0.45, 0.50, 0.55) )
let B2 = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.60, 0.60) )

var nn = NeuralNetwork(learningRate: 0.5)
nn.addLayer(layer: Layer.input(ID: 0, size: 2))
nn.addLayer(layer: Layer.fullyConnected(ID: 1, size: 2, activation: .sigmoid))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size: 2, activation: .sigmoid))

nn.printResults = false
let success = nn.compile()

nn.setLayerWeights(ID: 1, W: W1, B: B1)
nn.setLayerWeights(ID: 2, W: W2, B: B2)
*/

/*
// Example 2 - https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
var x = Matrix(3, 1, withArray: [Double](arrayLiteral: 1.00, 4.00, 5.00) )
var y = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.10, 0.05) )

let W1 = Matrix(2, 3, withArray: [Double](arrayLiteral: 0.1, 0.3, 0.5, 0.2, 0.4, 0.6) )
let B1 = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.5, 0.5) )
let W2 = Matrix(2, 2, withArray: [Double](arrayLiteral: 0.7, 0.9, 0.8, 0.1) )
let B2 = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.5, 0.5) )

var nn = NeuralNetwork(learningRate: 0.01)
nn.addLayer(layer: Layer.input(ID: 0, size: 3))
nn.addLayer(layer: Layer.fullyConnected(ID: 1, size: 2, activation: .sigmoid))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size: 2, activation: .sigmoid))

nn.printResults = false
let success = nn.compile()

nn.setLayerWeights(ID: 1, W: W1, B: B1)
nn.setLayerWeights(ID: 2, W: W2, B: B2)
*/

/*
// Example 3 - https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
var x = Matrix(2, 1, withArray: [Double](arrayLiteral: 1.00, 1.00) )
var y = Matrix(1, 1, withArray: [Double](arrayLiteral: 0.00) )

let W1 = Matrix(3, 2, withArray: [Double](arrayLiteral: 0.8, 0.2, 0.4, 0.9, 0.3, 0.5) )
let B1 = Matrix(3, 1, withArray: [Double](arrayLiteral: 0.0, 0.0, 0.0) )
let W2 = Matrix(1, 3, withArray: [Double](arrayLiteral: 0.3, 0.5, 0.9) )
let B2 = Matrix(1, 1, withArray: [Double](arrayLiteral: 0.0) )

var nn = NeuralNetwork(learningRate: 1.0)
nn.addLayer(layer: Layer.input(ID: 0, size: 2))
nn.addLayer(layer: Layer.fullyConnected(ID: 1, size: 2, activation: .sigmoid))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size: 1, activation: .sigmoid))

nn.printResults = false
let success = nn.compile()

nn.setLayerWeights(ID: 1, W: W1, B: B1)
nn.setLayerWeights(ID: 2, W: W2, B: B2)
*/

/*
// Example 4 - https://hmkcode.com/ai/backpropagation-step-by-step/
var x = Matrix(2, 1, withArray: [Double](arrayLiteral: 2.00, 3.00) )
var y = Matrix(1, 1, withArray: [Double](arrayLiteral: 1.00) )

let W1 = Matrix(2, 2, withArray: [Double](arrayLiteral: 0.11, 0.21, 0.12, 0.08) )
let B1 = Matrix(2, 1, withArray: [Double](arrayLiteral: 0.00, 0.00, 0.00) )
let W2 = Matrix(1, 2, withArray: [Double](arrayLiteral: 0.14, 0.15) )
let B2 = Matrix(1, 1, withArray: [Double](arrayLiteral: 0.00) )

var nn = NeuralNetwork(learningRate: 0.05)
nn.addLayer(layer: Layer.input(ID: 0, size: 2))
nn.addLayer(layer: Layer.fullyConnected(ID: 1, size: 2, activation: .identity))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size: 1, activation: .identity))

nn.printResults = false
let success = nn.compile()

nn.setLayerWeights(ID: 1, W: W1, B: B1)
nn.setLayerWeights(ID: 2, W: W2, B: B2)
*/


var trainDataSet = InputDataSet(training: true)
//var testDataSet  = InputDataSet(training: true)

trainDataSet.loadData(filename: "Source Code downloads/MNIST dataset/mnist_train_small.csv")
//testDataSet.loadData(filename:  "Source Code downloads/MNIST dataset/mnist_test_small.csv")
//trainDataSet.printSample(sample: 5)

let scale = Double( 1/255 )
//let (X,Y) = trainDataSet.getSampleFromSet(sampleInSet: 5, scale: scale)
let (XM, YM) = trainDataSet.getRandomSampleSet(numberOfSamples: 3, scale: scale)

/*
var x = Matrix(784, 1, withConstant: 0.0)
x = x.randomize()

var y = Matrix( 10, 1, withConstant: 0.0)
y = y.randomize()
*/

var nn = NeuralNetwork(learningRate: 0.05)
nn.addLayer(layer: Layer.input(ID: 0, size: 784))
nn.addLayer(layer: Layer.fullyConnected(ID: 1, size: 200, activation: .sigmoid))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size: 100, activation: .sigmoid))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size:  60, activation: .sigmoid))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size:  30, activation: .sigmoid))
nn.addLayer(layer: Layer.fullyConnected(ID: 2, size:  10, activation: .sigmoid))

nn.printResults = false
nn.errorInterval = 100
let success = nn.compile()

nn.train(iterations: 2000, X: XM[0], Y: YM[0])
//nn.trainSet(iterations: 10000, XM: XM, YM: YM)

let A = nn.feedForward(X: XM[0])
A.printout()
YM[0].printout()

print()
let t1 = CFAbsoluteTimeGetCurrent()
print(t1 - t0, "sec")
