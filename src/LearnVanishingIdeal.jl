module LearnVanishingIdeal
#TODO How to make sure that if z in
import HomotopyContinuation: @var, evaluate
import LinearAlgebra: norm, rank

export approximateVanishingIdeal,
       approximateVanishingIdeal_maxDegree,
       leastSquaresListOfEquations,
       leastSquaresListOfEquations_quick,
	   affineVeronese,
	   projVeronese,
	   addNoise,
	   calculateMeanDistanceToVariety

include("auxiliaryFunctions.jl")
using .auxiliaryFunctions

"""
  @input The function approximateVanishingIdeal takes a list of points,
  a list of Degrees of the generators of the vanishing ideal and a
  boolean quick as input that indicates, which algorithm to use.

  @return a coefficient vector and the error (Sampson Distance or Mean Distance)
  of the best-approximating polynomial
"""
function approximateVanishingIdeal(points, listOfDegrees; quick=false, affine=true)
	typeof(listOfDegrees)==Vector{Int} || throw(error("listOfDegrees has the wrong type!"))
	degreeList = findEqListOfDegrees(listOfDegrees)
	return( leastSquaresListOfEquations( points, degreeList, affine; quick=quick))
end

"""
  @input a maximum occuring degree n among the generators of the vanishing ideal,
  a list of points, the amount of generators (codimension of the variety, as complete
  intersection is assumed) and a boolean quick.

  @return a coefficient vector and the error (Sampson Distance or Mean Distance)
  of the best-approximating polynomial
"""
function approximateVanishingIdeal_maxDegree(n, points, numEq, quick=false, affine=true)
	listOfDegrees = [n for i in 1:numEq]
	return(approximateVanishingIdeal(points, listOfDegrees, quick, affine))
end

"""
  This method is a more accurate variant of the method
  leastSquaresListOfEquations_quick.
  It uses the Mean Distance (HomotopyContinuation) for determining the
  best-approximating polynomial. Also, it uses a threshold of tau=2.o and
  400 gradient descent steps.
"""
function leastSquaresListOfEquations(data, listOfDegrees, affine; TOL = 1e-8, quick=false)
	time1 = time()
	if quick == true
		maxiter, epochs, threshold = 200, 2, 1.5
	else
		maxiter, epochs, threshold = 400, 3, 3
	end
	points = affine ? data : [vcat(point,[1]) for point in data]
	@var var[1:length(points[1])]
	startValuesEigen, startValuesVander, outputValues, err = [], [], [], Inf
	numEq, n = sum([entry[2] for entry in listOfDegrees]), maximum([entry[1] for entry in listOfDegrees])

	for entry in listOfDegrees
		EigenValueStart, EigenValueVander = comparisonOfMethods(entry[1], points, entry[2], threshold)
		append!(startValuesEigen, [[entry[2], EigenValueStart]])
		append!(startValuesVander, [[entry[2], EigenValueVander]])
	end

	startValueCombinations = quick ? [makeCombinations(startValuesEigen)] : [makeCombinations(startValuesEigen), makeCombinations(startValuesVander)]

	@var w[1:binomial(n+length(points[1])-1,n),1:numEq]
	veronese = affineVeronese(n,var[1:length(data[1])])
	println("The search space has dimension: ", binomial(n+length(data[1]),n))

	for combinations in startValueCombinations
		for combination in combinations
			combination, zeroEntries = fillUpWithZeros(combination, n, numEq, length(var))
			intermediateValues = [[round(co/norm(comb), digits=14) for co in comb] for comb in combination]
			cl, intermediateValues = cleanUp(intermediateValues, listOfDegrees, length(var))
			cl || continue # ideal generators are not minimal
			result = [vector'*veronese for vector in intermediateValues]
			currentError = calculateMeanDistanceToVariety(points, result, var)
			currentError!="dim>0" || continue # the ideal has a higher dimension than anticipated.
			if currentError < err
				println("Ansatz without iterations takes the cake! Error: ", currentError)
				err = currentError
				outputValues = [comb for comb in intermediateValues]
			end
			currentError>=TOL || break

			for i in 1:epochs
				saverArray = sampsonDistance(points, numEq, n, w, intermediateValues)
				intermediateValues, _ = weightedGradientDescent(points, n, w, [value for value in intermediateValues], numEq, maxiter, saverArray, zeroEntries)
				result = [vector'*veronese for vector in intermediateValues]
				currentError = calculateMeanDistanceToVariety(points, result, var)
				currentError!="dim>0" || break
				if currentError < err
					println("Ansatz with ", i,  " iterations takes the cake! Error: ",currentError)
					err = currentError
					outputValues = [vector for vector in intermediateValues]
				end
			end
		end
	end
	time2 = time()
	println("The algorithm took: ",time2-time1,"s")
	_, outputValues = cleanUp(outputValues, listOfDegrees, length(var))
	return([[round(val/norm(value), digits=10) for val in value] for value in outputValues], err)
end

#=
"""
  This method learns polynomials for a given degree of the variety and
  given codimension by iterating through all possible combinations of
  degrees that make up the degree of the variety.
"""
function approximateVanishingIdealDegreeVariety(points, degVariety::Int, numEq::Int, quick=false)
	@var var[1:length(points[1])]
	listsOfDegrees = findAllCompositions(numEq, degVariety)
	error = Inf
	finaloutput = []
	finalList = []
	for list in listsOfDegrees
		indicator = 0
		for entry in list
			if entry >= 6
				indicator = 1
			end
		end
		if indicator == 1
			continue
		end
		n = maximum(list)
		try
			output, currentError = approximateVanishingIdeal(points, list, quick)
			currentError = calculateMeanDistanceToVariety(points,[value'*affineVeronese(maximum(list),var) for value in output],var)
			if(currentError < error)
				error = currentError
				finaloutput = [vector for vector in output]
				finalList = list
			end
		catch e
			println("Fehler gefunden!")
			println(e)
		end
	end
	return(finaloutput, error,finalList)
end

"""
  This method is a statistical learning ansatz. it iterates through all
  degrees up to a given terminationDegree and calculates equations in this degree
"""
function inferringVanishingIdeal(points, terminationDegree, numEq)
	@var var[1:length(points[1])]
	error = Inf
	outputValues = []
	errorsInDegrees = []
	finalDegree = 0
	for deg in 1:terminationDegree
		values, currentError, finalList = approximateVanishingIdealDegreeVariety(points, deg, numEq, true)
		println("Error of the variety with degree ",deg," has error: ",currentError)
		if(currentError < error)
			error = currentError
			outputValues = [vector for vector in values]
			finalDegree = deg
		end
		append!(errorsInDegrees,[currentError])
		display(errorsInDegrees)
	end
	return(outputValues, error, deg)
end
=#


end
