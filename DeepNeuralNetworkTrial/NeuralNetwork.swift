//
//  NeuralNetwork.swift
//  DeepNeuralNetworkTrial
//
//  Created by Pieter Hendriks on 28/11/2020.
//

import Foundation


class NeuralNetwork {
	var learningRate: Double
	var printResults: Bool
	var printError:   Bool
	var errorInterval: Int
	private var networkCompiled: Bool

	private var layers = [Layer]()
	
	
	init(learningRate: Double) {
		self.learningRate = learningRate
		self.printResults = false
		self.printError =   true
		self.errorInterval = 1
		self.networkCompiled = false
	}
	
	func addLayer(layer: Layer) {
		layers.append(layer)
	}
	
	func setLayerWeights(ID: Int, W: Matrix, B: Matrix) -> Bool {
		if networkCompiled {
			for layer in layers {
				if layer.ID == ID {
					return layer.setWeights(W: W, B: B)
				}
			}
		} else {
			print("Error - can only set weights once ")
		}
		return false
	}

	func printout() {
		print("---------------------------------------------------------------------")
		print("Layer (type)                 Size                      Param")
		print("=====================================================================")
		for layer in layers {
			//
			print("Layer", layer.ID, "            ", layer.size)
			print("---------------------------------------------------------------------")
		}
		print("=====================================================================")
		print("Total params:")
		print("---------------------------------------------------------------------")
	}
	
	func compile() -> Bool {
		// Check if we already compiled our network
		if networkCompiled {
			print("Error - network already complied")
			return false
		}
		// initialize our layers if we have more than one layer in the network
		if layers.count < 2 {
			print("Cannot compile neural network - Not enough layers defined")
			return false
		} else {
			layers[0].initialize(next: layers[1], printResults: printResults)		// our input layer
			for i in 1..<layers.count-1 {
				layers[i].initialize(previous: layers[i-1], next: layers[i+1], printResults: printResults)
			}
			layers[layers.count-1].initialize(previous: layers[layers.count-2], printResults: printResults)
			self.networkCompiled = true
			return true
		}
	}
	
	func feedForward(X: Matrix) -> Matrix {
		return layers[layers.count-1].forward(X: X)
	}
	
	func outputError(m: Double, Y: Matrix) -> Double {
		return layers[layers.count-1].outputError(m: 1, Y: Y)
	}
	
	func backPropagate() {
		//
		var i = layers.count-2
		while i>0 {
			layers[i].backward(m: 1.0)
			i -= 1
		}
	}
	
	func updateLayers() {
		for i in 1..<layers.count {
			layers[i].update(learningRate: learningRate)
		}
	}
	
	private func costPrime(A: Matrix, Y: Matrix) -> Matrix {
		let DCDA = A - Y
		return DCDA
	}
	
	
	func train(iterations: Int, X: Matrix, Y: Matrix) {
		for i in 0..<iterations {
			let A = feedForward(X: X)
			let C = outputError(m: 1.0, Y: Y)
			if i % errorInterval == 0 {
				print(i," Cost =", C)
			}
			backPropagate()
			updateLayers()
		}
	}
	
	
	func trainSet(iterations: Int, XM: [Matrix], YM: [Matrix]) {
		let m = Double( XM.count )
		for i in 0..<iterations{
			for j in 0..<XM.count {
				//
				let A = feedForward(      X: XM[j])
				let C = outputError(m: m, Y: YM[j])
				if (i % errorInterval == 0 && j==0) {
					print(i," Cost =", C)
				}
				backPropagate()
				updateLayers()
			}
		}
	}
}
