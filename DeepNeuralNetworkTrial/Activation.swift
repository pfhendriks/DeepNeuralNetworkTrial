//
//  Activation.swift
//  DeepNeuralNetworkTrial
//
//  Created by Pieter Hendriks on 28/11/2020.
//

import Foundation

enum Activation {
	case sigmoid
	case ReLU
	case identity
	case TANH
	case none

	var forward: (Matrix) -> Matrix {
		switch self {
		case .sigmoid:
			return { matrix in
				let ex = Matrix(matrix.rows, matrix.cols, withArray: matrix.values.map { exp($0) } )
				return ex / (ex + 1.0)
			}
		case  .ReLU:
			return { matrix in
				let newValues = matrix.values.map { max(0.0, $0) }
				return Matrix(matrix.rows, matrix.cols, withArray: newValues )
			}
		case  .identity:
			return { matrix in
				let newValues = matrix.values.map { max($0, $0) }
				return Matrix(matrix.rows, matrix.cols, withArray: newValues )
			}
		case .TANH:
			return { matrix in
				let ex = Matrix(matrix.rows, matrix.cols, withArray: matrix.values.map { exp(2 * $0) } )
				return ( ( 2 * ex / (ex + 1.0) ) - 1.0 )
			}
		case .none:
			return { matrix in
				return matrix
			}
		}
	}

	var backward: (Matrix) -> Matrix {
		switch self {
		case .sigmoid:
			return { matrix in
				return self.forward(matrix) * (1 - self.forward(matrix))
			}
		case .ReLU:
			return { matrix in
				let newValues = matrix.values.map { $0 > 0.0 ? 1.0 : 0.0 }
				return Matrix(matrix.rows, matrix.cols, withArray: newValues )
			}
		case .identity:
			return { matrix in
				let newValues = matrix.values.map { $0 > 0.0 ? 1.0 : 1.0 }
				return Matrix(matrix.rows, matrix.cols, withArray: newValues )
			}
		case .TANH:
			return { matrix in
				return  (1 - ( self.forward(matrix) * self.forward(matrix) ) )
			}
		case .none:
			return { matrix in
				return matrix
			}
		}
	}
}
