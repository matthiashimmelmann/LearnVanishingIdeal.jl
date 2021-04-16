module auxiliaryFunctions

import LinearAlgebra: zeros, Matrix, svd, pinv, transpose, det, norm, I
import DynamicPolynomials: Polynomial, @polyvar, Term, PolyVar
import HomotopyContinuation: solve, randn, differentiate, solutions, real_solutions, System
import Combinatorics: binomial, powerset, multiexponents

export affineVeronese,
	projVeronese,
	addNoise,
	calculateMeanDistanceToVariety,
	findEqListOfDegrees,
	comparisonOfMethods,
	makeCombinations,
	fillUpWithZeros,
	sampsonDistance,
	weightedGradientDescent

function addNoise(points, variance)
	points = [[entry+randn(Float64) for entry in point] for point in points]
	return(points)
end

function jacobianProd(veronese, var)
	d = length(var)

	dMatrix = Array{Term, 2}(undef, length(veronese), d)
	dArray = [collect(differentiate(poly, var) for poly in veronese)][1]
	for i in 1:length(dArray)
		for j in 1:length(dArray[i])
			dMatrix[i,j] = dArray[i][j]
		end
	end
	tranMat=dMatrix
	Mat = collect(transpose(dMatrix))
	return(tranMat*Mat)
end

function vandermonde(n, d, array, proj=false)
	exponents = vcat(map(i -> collect(multiexponents(n,-i)), -d:0)...)
	if (proj == true)
		exponents = []
		for k in n:-1:0
			exp = collect(multiexponents(length(array[1])-1,k))
			[append!(entry,n-k) for entry in exp]
			append!(exponents,exp)
		end
	end
	output = Array{Float64,2}(undef, length(array), length(exponents))
	for i in 1:length(array)
		for j in 1:length(exponents)
			prod = 1.0
			for k in 1:length(exponents[j])
				if(exponents[j][k]!=0)
					prod = prod*((array[i][k])^(exponents[j][k]))
				end
			end
			output[i,j]=prod
		end
	end
	return(output)
end

function evaluationOfMatrix(gamma, Z, var)
	output = zeros(size(gamma)[1], size(gamma)[2])
	for iter in 1:length(Z)
		helper = Array{Float64, 2}(undef, size(gamma)[1], size(gamma)[2])
		for i in 1:size(output)[1]
			for j in 1:size(output)[2]
				helper[i,j] = (gamma[i,j])(var=>Z[iter])
			end
		end
		output = output + helper
	end
	return(output./length(Z))
end

function affineVeronese(n, var)
	exponents = vcat(map(i -> collect(multiexponents(length(var),-i)), -n:0)...)
	output = [prod(var.^exp) for exp in exponents]
	return(output)
end

function projVeronese(n,var)
	output = []
	for k in n:-1:0
		exponents = multiexponents(length(var)-1,k)
		append!(output,[prod(var[1:length(var)-1].^exp)*var[length(var)]^(n-k) for exp in exponents])
	end
	return(output)
end


function calculateMeanDistanceToVariety(points, equations, var)
	#HomotopyContinuation: https://www.juliahomotopycontinuation.org/examples/critical-points/
	@polyvar u[1:length(var)]
	if (length(var)-length(equations) == 1)
		d = [differentiate(equation, var) for equation in equations]
		matrix = Array{Polynomial, 2}(undef, length(var), length(var))
		matrix[1,:] = var-u
		for i in 1:length(d)
			matrix[i+1,:] = d[i]
		end
		systemArray = [det(matrix)]
		append!(systemArray, equations)
		F_u = System(systemArray, variables = var, parameters = u)
	elseif (length(var)-length(equations) > 1)
		d = [differentiate(equation, var) for equation in equations]
		matrix = Array{Polynomial, 2}(undef, length(equations)+1, length(var))
		matrix[1,:] = var-u
		for i in 1:length(d)
			matrix[i+1,:] = d[i]
		end
		binomialsets = filter(p->length(p)==length(equations)+1, collect(powerset(1:length(var))))
		saverMatrix=Array{Polynomial,2}(undef,length(equations)+1,length(equations)+1)
		systemArray=Array{Polynomial,1}(undef,0)
		for entry in binomialsets
			for i in 1:length(entry)
				saverMatrix[:,i] = matrix[:, entry[i]]
			end
			push!(systemArray,det(saverMatrix))
		end
		append!(systemArray, equations)
		F_u = System(systemArray, variables = var, parameters = u)
	else
		throw(error("The method is not yet supported for non-complete intersections!"))
	end
	p = randn(ComplexF64, length(points[1]))
	#@suppress begin
		result_p = solve(F_u, target_parameters = p)
		realSolutions = solve(
							F_u,
							solutions(result_p);
							start_parameters =  p,
							target_parameters = points,
							transform_result = (r,u) -> minimum([norm(rel-u) for rel in solutions(r)])
		)
		realSolutions = filter(p -> p!=Inf, realSolutions)
		return(sum(realSolutions)/length(realSolutions))
	#end
end

#=
 Output is a good estimate for the starting vector of our Least Square Iteration
 =#
function comparisonOfMethods(n,points,numEq,tau)
	timer = round(Int64, time() * 1000)
	projPoints = [vcat(point,[1]) for point in points]
	@polyvar projVar[1:length(projPoints[1])]
	@polyvar var[1:length(points[1])]

	veroneseProj = projVeronese(n,var)
	veroProdProj = sum([projVeronese(n,point)*projVeronese(n,point)' for point in points])/length(points)
	jacoProdProj = evaluationOfMatrix(jacobianProd(veroneseProj,var), points, var)
	svdSingular = svd(pinv(jacoProdProj)*veroProdProj)
	firstS = [entry / maximum(svdSingular.S) for entry in svdSingular.S]
	smallestS = firstS[length(firstS)-numEq+1]*tau
	display(filter(p-> p<=smallestS, firstS))
	numberOfSmallSingularValues = length(filter(p-> p<=smallestS, firstS))
	firstV = [svdSingular.V[:,i] for i in (length(veroneseProj)-numberOfSmallSingularValues+1):length(veroneseProj)]
	timer2 = round(Int64, time() * 1000)
	try
		Vandermonde = vandermonde(length(var),n,points,true)
		svdVander = svd(Vandermonde)
		secondS = [entry / maximum(svdVander.S) for entry in svdVander.S]
		secondV = svdVander.V[:,(length(veroneseProj)-numberOfSmallSingularValues+1):length(veroneseProj)]
		secondV = [secondV[:,i] for i in 1:size(secondV)[2]]
		return(firstV, secondV)
	catch e
		return(firstV, firstV)
		prinln("Error caught",e)
	end
end

function weightedGradientDescent(points, n, var, curw0, nEq, maxIter, saverArray, zeroEntries)
	#TODO implement batch GD
	zeroMatrix = zeros(Float64,length(curw0[1]),length(curw0))

	w0Matrix = Array{Float64,2}(undef,length(curw0[1]),length(curw0))
	for i in 1:length(curw0)
		w0Matrix[:,i] = curw0[i]
	end
	for zero in zeroEntries
		zeroMatrix[zero[2],zero[1]] = w0Matrix[zero[2],zero[1]]
	end
	curLoss = sum([(projVeronese(n,points[j])'*w0Matrix)*saverArray[j]*w0Matrix'*projVeronese(n,points[j]) for j in 1:length(points)])/length(points)+0.1*sum([entry.^2 for entry in zeroMatrix])
	i, prevLoss, lambda, prevw0Matrix = 1, curLoss+1, 0.1, w0Matrix+Matrix{Float64}(I, size(w0Matrix)[1], size(w0Matrix)[2])
	while  (i < maxIter )#&& lambda > 10^(-10) && curLoss > 10^(-16) && sqrt(sum([sum((w0Matrix[i,:]-prevw0Matrix[i,:]).^2)/size(w0Matrix)[2] for i in 1:size(w0Matrix)[1]])/size(w0Matrix)[1]) > 10^(-14))
		if ( curLoss > prevLoss )
			lambda = lambda/2
			w0Matrix, curLoss = prevw0Matrix, prevLoss
			i = i-1
		elseif (prevLoss > curLoss/0.9)
			lambda = lambda/0.9
		elseif (prevLoss > curLoss/0.95)
			lambda = lambda/0.95
		elseif (prevLoss > curLoss/0.99)
			lambda = lambda/0.99
		end

		prevLoss, prevw0Matrix = curLoss, w0Matrix
		for zero in zeroEntries
			zeroMatrix[zero[2],zero[1]] = w0Matrix[zero[2],zero[1]]
		end
		dLossHelper = 2*sum([(projVeronese(n,points[j])*projVeronese(n,points[j])')*w0Matrix*saverArray[j] for j in 1:length(points)])./length(points)+2*zeroMatrix
		w0Matrix = w0Matrix - lambda * dLossHelper
		curLoss = sum([(projVeronese(n,points[j])'*w0Matrix)*saverArray[j]*w0Matrix'*projVeronese(n,points[j]) for j in 1:length(points)])/length(points)+0.1*sum([entry.^2 for entry in zeroMatrix])
		i=i+1
	end

	return([w0Matrix[:,i]./norm(w0Matrix[:,i]) for i in 1:size(w0Matrix)[2]], curLoss)
end

function sampsonDistance(points, nEq, n, var, startValues)
	@polyvar zed[1:length(points[1])]
	veronese = projVeronese(n, zed)
	Qstart = [start'*veronese for start in startValues]
	J = [differentiate(q,zed) for q in Qstart]
	matrix = Array{Polynomial,2}(undef, nEq, length(points[1]))
	for i in 1:nEq
		matrix[i,:] = J[i]
	end

	prod = matrix*matrix'
	helper = Array{Float64,2}(undef, nEq, nEq)
	saverArray = []

	for point in points
		for i in 1:nEq
			for j in 1:nEq
				helper[i,j] = prod[i,j](zed=>point)
			end
		end
		append!(saverArray,[pinv(helper)])
	end
	return(saverArray)
end

function makeCombinations(values)
	saver = []
	for entry in values
		combis = collect(powerset(entry[2],entry[1],entry[1]))
		append!(saver, [combis])
	end
	#TODO use some sort of cartesian product instead.
	#(list1,list2,...,listn)->(list1[1]cuplist2[1]cup...cuplistn[1],...)
	return(cartesianUnion(saver))
end

function cartesianUnion(combinations)

	product = 1
	for comb in combinations
		product = product*length(comb)
	end
	output = Array{Any, 1}(undef, product)
	for i in 1:length(output)
		output[i]=[]
	end
	number = [1 for i in 1:length(combinations)]
	max = [length(comb) for comb in combinations]
	a = []
	j = 1
	while number[1]<=length(combinations[1])
		a = []
		for i in 1:length(combinations)
			append!(a,combinations[i][number[i]])
		end
		output[j] = [element for element in a]
		number[length(number)] = number[length(number)]+1
		number = checkNumber(number, max)
		j = j+1
	end
	return(output)
end

function checkNumber(number, max)
	for i in length(number):-1:2
		if(number[i]>max[i])
			number[i-1]=number[i-1]+1
			number[i]=1
		end
	end
	return(number)
end

function fillUpWithZeros(combination, n, numEq,d)
	zeroEntries = []

	for i in 1:length(combination)
		output = Array{Float64, 1}(undef, binomial(n+d-1,n))
		for j in 1:length(output)
			if(j<=length(output)-length(combination[i]))
				output[j]=0
				append!(zeroEntries,[[i,j]])
			else
				output[j] = combination[i][j-length(output)+length(combination[i])]
			end
		end
		combination[i] = [out for out in output]
	end
	return(combination, zeroEntries)
end

function findEqListOfDegrees(listOfDegrees)
	sort!(listOfDegrees)
	degreeList = []
	for x in listOfDegrees
		index = 0
		for entry in degreeList
			if entry[1]==x
				entry[2]=entry[2]+1
				index = 1
				break
			end
		end
		if(index==0)
			append!(degreeList, [[x,1]])
		end
	end
	return(	degreeList)
end

function findAllCompositions(amount, number)
	saverList = []
	for j in 1:number
		if(number%j==0)
			append!(saverList,[[j]])
		end
	end

	product = 1
	for k in (amount-1):-1:1
		helperList = []
		for entry in saverList
			append!(helperList, findDivisors(number/prod(entry), entry))
		end
		saverList = helperList
	end

	saverList = filter(p-> prod(p)==number,saverList)
	for list in saverList
		sort!(list)
	end
	saverSet = Set(saverList)
	saverList = collect(saverSet)
	return(saverList)
end

function findDivisors(number, currentList)
	if(number <= 0)
		return([])
	end
	saverList = []
	for k in 1:number
		if number%k==0
			helper = [list for list in currentList]
			append!(helper,[k])
			append!(saverList,[helper])
		end
	end
	return(saverList)
end

end
