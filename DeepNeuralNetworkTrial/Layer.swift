//
//  Layer.swift
//  DeepNeuralNetworkTrial
//
//  Created by Pieter Hendriks on 28/11/2020.
//

import Foundation

class Layer {
	var size: Int
	var activation: Activation
	var ID: Int
	
	/// weights and biases for the layer
	var W: Matrix!
	var B: Matrix!
	
	/// Matrix to hold results
	var Z: Matrix!
	var A: Matrix!
	
	var deltaX: Matrix!
	var deltaW: Matrix!
	var deltaB: Matrix!

	
	var previousLayer: Layer?
	var nextLayer:     Layer?
	
	private var weightsSet: Bool

	var printResults: Bool = false
	
	init(ID: Int, size: Int, activation: Activation) {
		self.size = size
		self.activation = activation
		
		self.ID = ID
		
		self.previousLayer = nil
		self.nextLayer     = nil

		self.weightsSet   = false
		self.printResults = false
	}
	
	func initialize(previous: Layer, next: Layer, printResults: Bool) {
		initialize(next:     next,     printResults: printResults)
		initialize(previous: previous, printResults: printResults)

	}
	
	func initialize(previous: Layer, printResults: Bool) {
		if !weightsSet {
			/// initialize our weights and biases with random values when nothing has been defined
			self.W      = Matrix(size, previous.size, randomRangeMin: -1.0, randomRangeMax: 1.0)
			self.B      = Matrix(size,             1, randomRangeMin: -1.0, randomRangeMax: 1.0)
			self.deltaW = Matrix(size, previous.size, withConstant: 0.0)
			self.deltaB = Matrix(size,             1, withConstant: 0.0)
			self.weightsSet = true
		}
		
		self.previousLayer = previous
		self.printResults  = printResults
	}
	
	func initialize(next: Layer, printResults: Bool) {
		self.nextLayer     = next
		self.printResults  = printResults
	}
	
	static func input(ID: Int, size: Int) ->Layer {
		return Layer(ID: ID, size: size, activation: .none)
	}
	
	static func fullyConnected(ID: Int, size: Int, activation: Activation) -> Layer {
		return Layer(ID: ID, size: size, activation: activation)
	}
	
	func setWeights(W: Matrix, B: Matrix) -> Bool {
		if (self.W.rows == W.rows && self.W.cols == W.cols && self.B.rows == B.rows && self.B.cols == B.cols) {
			self.W = W
			self.B = B
			self.weightsSet = true
			return true
		} else {
			print("Error in setting weight or biases matrix for layer", self.ID,". Dimensions do not match network requirements")
			return false
		}
	}

	/// Layer forward propagation
	func forward(X: Matrix) -> Matrix {
		if previousLayer != nil {
			Z = W <*> previousLayer!.forward(X: X) + B
			if printResults {
				print("Forward Pass Layer", ID)
				print("W")
				W.printout()
				print("B")
				B.printout()
				print("Z")
				Z.printout()
			}
			A = activation.forward(Z)
		} else {
			A = X
			if printResults { print("Input Layer") }
		}
		
		if printResults {
			print("A")
			A.printout()
		}
		
		return A
	}
	
	/// Output Error
	/// this will only be applied to our last layer
	func outputError(m: Double, Y: Matrix) -> Double {
		let cost = 0.5 * (Y - A).sumOfSquares()
		if printResults {
			print("Cost =", cost)
			print()
		}

		let dCdA = Y - A
		self.deltaX = dCdA * ( activation.backward(Z) )

		self.deltaW = self.deltaW + ( (1 / m) * ( deltaX <*> previousLayer!.A.T ) )
		self.deltaB = self.deltaB + ( (1 / m) * deltaX )

		if printResults {
			print("dCdA")
			dCdA.printout()
			print("Sigma Prime(Z)")
			activation.backward(Z).printout()
			print("dx")
			deltaX.printout()
			print("dW")
			deltaW.printout()
			print("dB")
			deltaB.printout()
		}

		return cost
	}
	
	func backward(m: Double) {
		if printResults { print("Back Propagate Layer", ID) }

		self.deltaX = (nextLayer!.W.T <*> nextLayer!.deltaX ) * activation.backward(Z)
		self.deltaW = self.deltaW + ( (1 / m) * ( self.deltaX <*> previousLayer!.A.T ) )
		self.deltaB = self.deltaB + ( (1 / m) * ( self.deltaX ) )

		if printResults {
			print("Sigma Prime(Z)")
			activation.backward(Z).printout()
			print("dx")
			deltaX.printout()
			print("dW")
			deltaW.printout()
			print("dB")
			deltaB.printout()
		}
	}
	
	func update(learningRate: Double) {
		if printResults {
			print("Update Layer", ID)
			print("W before")
			W.printout()
		}
		let dW =  ( -learningRate * self.deltaW )
		let dB =  ( -learningRate * self.deltaB )
		self.W = self.W + dW
		self.B = self.B + dB
		if printResults {
			print("W after")
			W.printout()
		}
		// Reset our weight and bias update matrices
		self.deltaW = dW.zero()
		self.deltaB = dB.zero()
	}
}
