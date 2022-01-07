module auxiliaryFunctions

import LinearAlgebra: zeros, Matrix, svd, pinv, transpose, det, norm, I, rank
import HomotopyContinuation: solve, randn, differentiate, solutions, real_solutions, System, @polyvar
import DynamicPolynomials: Polynomial, Term
import Combinatorics: binomial, powerset, multiexponents
import Suppressor: @suppress

export affineVeronese,
	projVeronese,
	addNoise,
	calculateMeanDistanceToVariety,
	findEqListOfDegrees,
	comparisonOfMethods,
	makeCombinations,
	fillUpWithZeros,
	sampsonDistance,
	weightedGradientDescent,
	cleanUp

function addNoise(points, variance)
	points = [[entry+randn(Float64) for entry in point] for point in points]
	return(points)
end

#Take the filledzero list of degrees and return, if the polynomial generators are minimal.
function cleanUp(intermediateValues, listOfDegrees, varlength)
	totalexponents = vcat(map(i -> collect(multiexponents(varlength,-i)), -listOfDegrees[end][1]:0)...)
	array=[]
	indecesOfBasis = []
	for q in 1:length(listOfDegrees)
		currentsum = listOfDegrees[q][1]==listOfDegrees[1][1] ? 0 : sum([en[2] for en in listOfDegrees[1:q-1]])
		for i in 1:listOfDegrees[q][2]
			push!(array,intermediateValues[i+currentsum])
			push!(indecesOfBasis, length(array))
		end
		currentexponents = vcat(map(i -> collect(multiexponents(varlength,-i)), -listOfDegrees[q][1]:0)...)
		for val in 1:sum(totalexponents[1])-sum(currentexponents[1])
			indexarray = [[findfirst(t->t==adder+entry, totalexponents) for entry in currentexponents] for adder in collect(multiexponents(varlength,val)) ]
			for i in 1:listOfDegrees[q][2]
				for j in 1:length(indexarray)
					helper = [0. for _ in 1:length(totalexponents)]
					for k in 1:length(indexarray[j])
						helper[indexarray[j][k]] = intermediateValues[currentsum+i][length(totalexponents)-length(collect(multiexponents(varlength,listOfDegrees[q][1])))+k-1]
					end
					push!(array,helper)
				end
			end
		end
	end
	#Return true if the rank does not drop (ideal is minimal) and return the reduced ideal generators. Else return false
	if rank(hcat(array...))==length(array)
		redarray = reduce([array[ind] for ind in indecesOfBasis], array)
		return(true, [ar./norm(ar) for ar in redarray])
	else
		return(false, [])
	end
end

#Calculate f mod I for f not in I
function reduce(truebasis, array)
	for i in 1:length(truebasis)
		element = Base.copy(truebasis[i])
		fil = sort(filter(t->element!=t, array), rev=true)
		indeces = [findfirst(t->norm(t)>1e-15, el) for el in fil]

		#Gauss elimination algorithm
		for j in 1:length(fil)
			filind = findfirst(t->norm(t)>1e-15, fil[j])
			if norm(element[filind])>1e-15
				c1 = fil[j][filind]
				c2 = element[filind]
				element = element - fil[j]./c1.*c2
			end
			truebasis[i] = element
		end
	end
	return(truebasis)
end

function jacobianProd(veronese, var)
	d = length(var)
	dMatrix = Array{Polynomial, 2}(undef, length(veronese), length(veronese))
	dArray = [collect(differentiate(poly, var) for poly in veronese)][1]
	for i in 1:size(dMatrix)[1]
		for j in 1:size(dMatrix)[2]
			dMatrix[i,j] = Polynomial(differentiate(veronese[i], var)'*differentiate(veronese[j], var))
		end
	end
	return(dMatrix)
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
	exponents = vcat(map(i -> collect(multiexponents(length(var)-1,-i)), -n:0)...)
	output = [prod(var[1:length(var)-1].^exp)*var[length(var)]^(n-sum(exp)) for exp in exponents]
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
		systemArray = filter!(t->t!=0,systemArray)
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
		systemArray = filter!(t->t!=0,systemArray)
		F_u = System(systemArray, variables = var, parameters = u)
	else
		throw(error("The method is not yet supported for non-complete intersections!"))
	end
	p = randn(ComplexF64, length(points[1]))
	#@suppress begin
		try
			result_p = solve(F_u, target_parameters = p)
			realSolutions = solve(
								F_u,
								solutions(result_p);
								start_parameters =  p,
								target_parameters = points,
								transform_result = (r,u) -> minimum([norm(rel-u) for rel in solutions(r)])
			)
			realSolutions = filter(p -> p!=Inf, realSolutions)
			if !isempty(realSolutions)
				return(sum(realSolutions)/length(realSolutions))
			else
				return("dim>0")
			end
		catch e
			return("dim>0")
		end
	#end
end

#=
 Output is a good estimate for the starting vector of our Least Square Iteration
 =#
function comparisonOfMethods(n,points,numEq,tau)
	timer = round(Int64, time() * 1000)
	@polyvar var[1:length(points[1])]
	projPoints = [vcat(point,[1]) for point in points]
	@polyvar projVar[1:length(projPoints[1])]

	veroneseProj = projVeronese(n,projVar)
	veroProdProj = sum([projVeronese(n,point)*projVeronese(n,point)' for point in projPoints])/length(projPoints)
	jacoProdProj = evaluationOfMatrix(jacobianProd(veroneseProj,projVar), projPoints, projVar)
	svdSingular = svd(pinv(jacoProdProj)*veroProdProj)
	firstS = [entry / maximum(svdSingular.S) for entry in svdSingular.S]
	smallestS = firstS[length(firstS)-numEq]*tau
	numberOfSmallSingularValues = length(filter(p-> p<=smallestS, firstS))
	firstV = [svdSingular.V[:,i] for i in (length(veroneseProj)-numberOfSmallSingularValues+1):length(veroneseProj)]
	timer2 = round(Int64, time() * 1000)
	try
		Vandermonde = vandermonde(length(var),n,points,false)
		svdVander = svd(Vandermonde)
		secondS = [entry / maximum(svdVander.S) for entry in svdVander.S]
		secondV = svdVander.V[:,(length(veroneseProj)-numberOfSmallSingularValues)+1:length(veroneseProj)]
		secondV = [secondV[:,i] for i in 1:size(secondV)[2]]
		return(firstV, secondV)
	catch e
		return(firstV, firstV)
		prinln("Error caught",e)
	end
end

function weightedGradientDescent(points, n, var, curw0, nEq, maxIter, saverArray, zeroEntries)

	w0Matrix = Array{Float64,2}(undef,length(curw0[1]),length(curw0))
	for i in 1:length(curw0)
		w0Matrix[:,i] = curw0[i]
	end
	function zeroMatrix(w0Matrix)
		mat = zeros(Float64,length(curw0[1]),length(curw0))
		for zero in zeroEntries
			mat[zero[2],zero[1]] = w0Matrix[zero[2],zero[1]]
		end
		return(mat)
	end

	lossFct = w0Matrix -> sum([(projVeronese(n,vcat(points[j],1))'*w0Matrix)*saverArray[j]*w0Matrix'*projVeronese(n,vcat(points[j],1)) for j in 1:length(points)])/length(points)+100*sum([entry.^2 for entry in zeroMatrix(w0Matrix)])
	curLoss = lossFct(w0Matrix)
	dLossFct = w0Matrix -> 2*sum([(projVeronese(n,vcat(points[j],1))*projVeronese(n,vcat(points[j],1))')*w0Matrix*saverArray[j] for j in 1:length(points)])./length(points)+200*zeroMatrix(w0Matrix)
	i = 1
	while  i < maxIter && norm(dLossFct(w0Matrix)) > 1e-2
		w0Matrix, curLoss = backtracking_line_search(w0Matrix, dLossFct, lossFct)
		i=i+1
	end
	return([w0Matrix[:,i]./norm(w0Matrix[:,i]) for i in 1:size(w0Matrix)[2]], curLoss)
end

function backtracking_line_search(w0Matrix, dLossFct, lossFct; r=1e-3, s=0.7)
	dLoss = dLossFct(w0Matrix)./norm(dLossFct(w0Matrix))
	α = 0; t = 0.1; β = 10000
	while norm(β-α)>1e-15
		if lossFct(w0Matrix)-lossFct(w0Matrix - t * dLoss) < r*t*norm(dLoss)^2
			β = t
			t = (α+β)/2
		elseif s*norm(dLoss'*dLoss)<norm(dLoss'*dLossFct(w0Matrix - t*dLoss)./norm(dLossFct(w0Matrix - t*dLoss)))
			α = t
			t = (α+β)/2
		else
			break
		end
	end
	return(w0Matrix - t * dLoss, lossFct(w0Matrix - t * dLoss))
end

function sampsonDistance(points, nEq, n, var, startValues)
	@polyvar zed[1:length(points[1])+1]
	veronese = projVeronese(n, zed)
	Qstart = [start'*veronese for start in startValues]
	matrix = Array{Polynomial,2}(undef, nEq, length(points[1]))
	for i in 1:nEq, j in 1:nEq
		matrix[i,j] = differentiate(Qstart[i],zed)'*differentiate(Qstart[j],zed)
	end
	#TODO Whatever needs to be done here.
	helper = Array{Float64,2}(undef, nEq, nEq)
	saverArray = []

	for point in points
		for i in 1:nEq
			for j in 1:nEq
				helper[i,j] = matrix[i,j](zed=>vcat(point,1))
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
		output = Array{Float64, 1}(undef, length(combination[end]))
		output[1:length(output)-length(combination[i])] .= 0
		output[length(output)-length(combination[i])+1:end] = combination[i]
		append!(zeroEntries,[[i,j] for j in 1:length(output)-length(combination[i])])
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
