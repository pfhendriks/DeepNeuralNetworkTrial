//
//  InputDataSet.swift
//  DeepNeuralNetworkTrial
//
//  Created by Pieter Hendriks on 28/11/2020.
//

import Foundation


struct DataSample {
	var answer: Int
	var data : [Int] = []
	
	init() {
		answer = 0
	}
}

class InputDataSet {
	let filemgr = FileManager.default
	var docsDir:  String?
	var dataFile: String?

	var trainingSet: Bool
	var size: Int
	var dataSamples: [DataSample] = []
	
	init(training: Bool) {
		self.trainingSet = training
		self.size = 0
	}
	
	func loadData(filename: String) {
		let dirPaths = filemgr.urls(for: .desktopDirectory, in: .userDomainMask)
		dataFile = dirPaths[0].appendingPathComponent(filename).path

		if filemgr.fileExists(atPath: dataFile!) {
			let databuffer = filemgr.contents(atPath: dataFile!)
			print("Reading file", dataFile!)
			let datastring = NSString(data: databuffer!, encoding: String.Encoding.utf8.rawValue)

			let s = String(datastring!)
			let lines = s.split(separator: "\n")
			for line in lines {
				var dataInLine = [Int]()
				let columns = line.split(separator: ",", omittingEmptySubsequences: false)
				for column in columns {
					if let newItem = Int(column)
					{
						dataInLine.append(newItem)
					} else {
						print("error in data format")
					}
				}

				/// fill our dataSamples
				var dataSample = DataSample()
				if trainingSet {
					dataSample.answer = dataInLine[0]
					for i in 1..<dataInLine.count {
						dataSample.data.append(dataInLine[i])
					}
				} else {
					for i in 0..<dataInLine.count {
						dataSample.data.append(dataInLine[i])
					}
				}
				dataSamples.append(dataSample)
			}
			print("Completed reading file")
			size = dataSamples.count
			print("Number of samples in dataset is :", size)
		} else {
			print("Error reading file", dataFile!)
		}
	}
	
	/// getSampleFromSet(sampleInSet: Int, scale: Double) -> (Matrix, Matrix)
	/// returns a matrix with a single data sample from the set and a matrix with the associated answer matrix
	/// Sample matrix is scaled by scale
	/// answer matrix is size (10, 1) with all zero values,except 1.0 at the index equal to the correct answer
	func getSampleFromSet(sampleInSet: Int, scale: Double) -> (Matrix, Matrix) {
		// check is sample requested is within our range
		if (sampleInSet<0 || sampleInSet>(dataSamples.count - 1)) {
			// out of range. Return zero-matrix
			print("Error - Sample requested is out of set range")
			let x = Matrix(dataSamples[0].data.count, 1, withConstant: 0.0)
			let y = Matrix(                       10, 1, withConstant: 0.0)
			return (x, y)
		}
		
		// Define our data matrix
		var sampleArray = [Double]()
		for i in 0..<dataSamples[sampleInSet].data.count {
			let scaledData = Double( dataSamples[sampleInSet].data[i] ) * scale
			sampleArray.append(scaledData)
		}
		
		// Define our answer matrix
		var answerArray = [Double]()
		for n in 0..<10 {
			var answer = 0.0
			if dataSamples[sampleInSet].answer == n {
				answer = 1.0
			}
			answerArray.append(answer)
		}

		let x = Matrix(dataSamples[0].data.count, 1, withArray: sampleArray)
		let y = Matrix(                       10, 1, withArray: answerArray)
		return (x, y)
	}
	
	/// getSampleSet()
	///
	func getRandomSampleSet(numberOfSamples: Int, scale: Double) -> ( [Matrix], [Matrix]) {
		//
		var XM = [Matrix]()
		var YM = [Matrix]()
		
		for _ in 0..<numberOfSamples {
			let NumberOfSamplesInSet = dataSamples.count
			let sampleIndex = Int.random(in: 0..<NumberOfSamplesInSet)
			let (X, Y) = getSampleFromSet(sampleInSet: sampleIndex, scale: scale)
			XM.append(X)
			YM.append(Y)
		}
		return (XM, YM)
	}
	
	/// printSample(sample: Int)
	/// print a graphical presentation of the charater data
	func printSample(sample: Int) {
		if ( sample < dataSamples.count ) {
			if ( dataSamples[sample].data.count == 784 ) {
				for i in 0..<28 {
					var s = ""
					for j in 0..<28 {
						//
						let index = i*28 + j
						let item = dataSamples[sample].data[index]
						if item == 0 {
							s += "  "
						} else {
							if item < 128 {
								s += ".."
							} else {
								s += "**"
							}
						}
					}
					print(s)
				}
				print("Correct answer =", dataSamples[sample].answer)
			} else {
				return
			}
		} else {
			print("Print error - sample is outside of range")
		}
	}
}

