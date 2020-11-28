//
//  Matrix.swift
//  DeepNeuralNetworkTrial
//
//  Created by Pieter Hendriks on 28/11/2020.
//

import Foundation
import Accelerate


struct Matrix {
	var rows: Int
	var cols: Int
	var values: [Double] = []
	

	init(_ rows: Int, _ cols: Int, withConstant: Double ) {
		self.rows = rows
		self.cols = cols
		self.values = Array<Double>(repeating: withConstant, count: rows * cols)
	}

	init(_ rows: Int, _ cols: Int, randomRangeMin: Double, randomRangeMax: Double ) {
		self.rows = rows
		self.cols = cols
		var values = [Double]()
		for _ in 0..<(rows * cols) {
			let new = Double.random(in: randomRangeMin...randomRangeMax)
			values.append(new)
		}
		self.values = values
	}

	init(_ rows: Int, _ cols: Int, withArray values: [Double] ) {
		self.rows = rows
		self.cols = cols
		self.values = values
	}
	
	init(_ rows: Int, _ cols: Int, withArray values: [Int] ) {
		self.rows = rows
		self.cols = cols
		var newValues = [Double]()
		for i in 0..<values.count {
			newValues.append( Double(values[i] ) )
		}
		self.values = newValues
	}
	
	init(_ withArray: [[Double]]) {
		self.rows = withArray.count
		self.cols = withArray[0].count
		self.values = withArray.flatMap{ $0 }
	}
	
	init(_ withMatrix: Matrix) {
		self.rows = withMatrix.rows
		self.cols = withMatrix.cols
		self.values = withMatrix.values
	}
	

	subscript(_ row: Int, _ col: Int) -> Double {
		get {
			assert(row < rows && col < cols, "Row and column index must be within bounds of matrix")
			return values[ (row * cols) + col ]
		}
		set {
			assert(row >= 0 && row < rows && col >= 0 && col < cols, "Row and column index must be within bounds of matrix")
			values[ (row * cols) + col] = newValue
		}
	}
	
	public var description: String {
	  var description = ""

	  for i in 0..<rows {
		let contents = (0..<cols).map{ String(format: "%12.9f", self[i, $0]) }.joined(separator: " ")

		switch (i, rows) {
		case (0, 1):
		  description += "( \(contents) )\n"
		case (0, _):
		  description += "⎛ \(contents) ⎞\n"
		case (rows - 1, _):
		  description += "⎝ \(contents) ⎠\n"
		default:
		  description += "⎜ \(contents) ⎥\n"
		}
	  }
	  return description
	}
	
	func printout() { print(self.description) }
}


/// Summation functions
/// sum()			sum ( A[i] )
/// sumOfMagnitudes()	sum ( |A[i]| )
/// sumOfSquares()		sum ( A[i] ^2 )
/// mean()			sum( A[i] ) / n
/// meanSquare()		sum( A[i]^2 ) / n
/// rmsq()			sqrt ( sum(  A[i]^2 ) / n )

extension Matrix {
	public func sum() -> Double {
		var sum = 0.0
		vDSP_sveD(values, 1, &sum, vDSP_Length(values.count))
		return sum
	}
	
	public func sumOfMagnitudes() -> Double {
		var sum = 0.0
		vDSP_svemgD(values, 1, &sum, vDSP_Length(values.count))
		return sum
	}
	
	public func sumOfSquares() -> Double {
		var sum = 0.0
		vDSP_svesqD(values, 1, &sum, vDSP_Length(values.count))
		return sum
	}
	
	public func mean() -> Double {
		var mean = 0.0
		vDSP_meanvD(values, 1, &mean, vDSP_Length(values.count))
		return mean
	}
	
	public func meanSquare() -> Double {
		var meanSquare = 0.0
		vDSP_measqvD(values, 1, &meanSquare, vDSP_Length(values.count))
		return meanSquare
	}
	
	public func rmsq() -> Double {
		var rmsq = 0.0
		vDSP_rmsqvD(values, 1, &rmsq, vDSP_Length(values.count))
		return rmsq
	}

}

/// Matrix modification functions
/// sqaured()		C[i,j] = A[i] ^2
/// zero()		C[i,j] = 0.0
/// ones()		C[i,j] = 1.0
/// random()		C[i,i] = random value between -1 and +1
/// identity()		C[i,j] = 1 when i=j and 0 for other cells
/// transpose()	C[i,j] = C[j,i]
/// T()			C[i,j] = C[j,i]	short notation for transpose()

extension Matrix {
	public func squared() -> Matrix {
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vsqD(values, 1, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}
	
	public func zero() -> Matrix {
		return Matrix(self.rows, self.cols, withConstant: 0.0)
	}
	
	public func ones() -> Matrix {
		return Matrix(self.rows, self.cols, withConstant: 1.0)
	}

	public func randomize() -> Matrix {
		var values = [Double]()
		for _ in 0..<(self.cols * self.rows) {
			let new = Double.random(in: -1...1)
			values.append(new)
		}
		return Matrix(self.rows, self.cols, withArray: values)
	}
	
	public func identity() -> Matrix {
		assert(rows == cols, "Number of rows and columns needs to be equal to create an identity matrix")
		var values = [Double]()
		for i in 0..<self.rows {
			for j in 0..<self.cols {
				var new = 0.0
				if i == j {
					new = 1.0
				}
				values.append(new)
			}
		}
		return Matrix(self.rows, self.cols, withArray: values)
	}

	func transpose() -> Matrix {
		var resultValues = [Double](repeating: 0.0, count: (rows * cols) )
		vDSP_mtransD(values, 1, &resultValues, 1, vDSP_Length(rows), vDSP_Length(cols) )
		return Matrix(cols, rows, withArray: resultValues)
	}
	
	public var T: Matrix {
		return self.transpose()
	}
}

infix operator <*> : MultiplicationPrecedence

extension Matrix {
	// scalar operations
	public static func +(lhs: Matrix, rhs: Double) -> Matrix { return lhs.addScalarToMatrix(scalar:  rhs) }
	public static func +(lhs: Double, rhs: Matrix) -> Matrix { return rhs.addScalarToMatrix(scalar:  lhs) }
	public static func -(lhs: Matrix, rhs: Double) -> Matrix { return lhs.addScalarToMatrix(scalar: -rhs) }
	public static func -(lhs: Double, rhs: Matrix) -> Matrix { return rhs.subMatrixFromScalar(scalar: lhs)}
	public static func *(lhs: Matrix, rhs: Double) -> Matrix { return lhs.mulMatrixWithScalar(scalar: rhs) }
	public static func *(lhs: Double, rhs: Matrix) -> Matrix { return rhs.mulMatrixWithScalar(scalar: lhs) }
	public static func /(lhs: Matrix, rhs: Double) -> Matrix { return lhs.divMatrixWithScalar(scalar: rhs) }

	// matrix operations
	public static func +(lhs: Matrix, rhs: Matrix) -> Matrix { return lhs.addMatrixToMatrix(matrix: rhs) }
	public static func -(lhs: Matrix, rhs: Matrix) -> Matrix { return lhs.subMatrixToMatrix(matrix: rhs) }
	public static func *(lhs: Matrix, rhs: Matrix) -> Matrix { return lhs.mulMatrixByMatrix(matrix: rhs) }
	public static func /(lhs: Matrix, rhs: Matrix) -> Matrix { return lhs.divMatrixByMatrix(matrix: rhs) }

	// dot product
	public static func <*>(lhs: Matrix, rhs: Matrix) -> Matrix { return lhs.dot(rhs) }
}


/// Scalar functions
/// element-wise addition		c[i] = a[i] + b
/// element-wise substraction	c[i] = b - a[i]
/// element-wise multiplication	c[i] = a[i] * b
/// element-wise division		c[i] = a[i] / b
extension Matrix {
	private func addScalarToMatrix(scalar: Double) -> Matrix {
		var sc = scalar
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vsaddD(values, 1, &sc, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}
	
	private func subMatrixFromScalar(scalar: Double) -> Matrix {
		return (self.mulMatrixWithScalar(scalar: -1.0) + scalar)
	}
	
	private func mulMatrixWithScalar(scalar: Double) -> Matrix {
		var sc = scalar
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vsmulD(values, 1, &sc, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}
	
	private func divMatrixWithScalar(scalar: Double) -> Matrix {
		assert(scalar != 0.0, "Trying to divide matrix by zero")
		var sc = scalar
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vsdivD(values, 1, &sc, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}
}


/// Matrix fuctions
/// element-wise addition		c[i] = a[i] + b[i]
/// element-wise substraction	c[i] = a[i] - b[i]
/// element-wise multiplication	c[i] = a[i] * b[i]
/// element-wise division		c[i] = a[i] / b[i]
extension Matrix {
	private func addMatrixToMatrix(matrix: Matrix) -> Matrix {
		assert(rows == matrix.rows && cols == matrix.cols, "Numner of rows and columns must be the same for matrix addition")
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vaddD(values, 1, matrix.values, 1, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}
	
	private func subMatrixToMatrix(matrix: Matrix) -> Matrix {
		assert(rows == matrix.rows && cols == matrix.cols, "Numner of rows and columns must be the same for matrix substraction")
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vsubD(values, 1, matrix.values, 1, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}
	
	private func mulMatrixByMatrix(matrix: Matrix) -> Matrix {
		assert(rows == matrix.rows && cols == matrix.cols, "Numner of rows and columns must be the same for Hadamard product")
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vmulD(values, 1, matrix.values, 1, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}

	private func divMatrixByMatrix(matrix: Matrix) -> Matrix {
		assert(rows == matrix.rows && cols == matrix.cols, "Numner of rows and columns must be the same for Hadamard division")
		var resultValues = [Double](repeating: 0.0, count: values.count )
		vDSP_vdivD(matrix.values, 1, values, 1, &resultValues, 1, vDSP_Length(values.count))
		return Matrix(rows, cols, withArray: resultValues)
	}
}


/// Matrix dot-product
extension Matrix {
	private func dot(_ rhs: Matrix) -> Matrix {
		assert(cols == rhs.rows, "Number of colums in lhs need to match rows in rhs for matrix multiplication")
		var resultValues = [Double](repeating: 0.0, count: (rows * rhs.cols) )
		vDSP_mmulD(values, 1, rhs.values, 1, &resultValues, 1, vDSP_Length(rows), vDSP_Length(rhs.cols), vDSP_Length(cols) )
		return Matrix(rows, rhs.cols, withArray: resultValues)
	}
}

