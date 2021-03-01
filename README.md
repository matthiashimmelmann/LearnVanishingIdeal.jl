# LearnVanishingIdeal
 
This `Julia` package aims at learning an algebraic variety from a list of potentially noisy data points. In doing so, it uses statistical techniques inspired by the *Generalized Principal Component Analysis* by Ma et al. 

It runs under the assumption that the underlying variety is a complete intersection. At the moment, either a maximum occuring degree among the generators of the vanishing ideal or a list of degrees is required as input. In addition, we assume that the codimension of the variety is known. 

# Instructions

In the following example, we are given the points (0,1), (-1,2) and (1,2) that lie on a parabola. Hence, we want to approximate these data points with one degree 2 curve. 

```
julia> @polyvar x y
julia> result, sampserror = approximateVanishingIdeal([[-2,5],[2,5],[0,1],[1,2],[-1,2]], [2])
julia> [[round(entry,digits=2) for entry in value]'*affineVeronese(2,[x,y]) for value in result]
1-element Array{Polynomial{true,Float64},1}:
 -0.58x² + 0.58y - 0.58
```

The output of this method is a normalized version of the classic parabola ![ y=x^2+1](https://latex.codecogs.com/svg.latex?y=x^2+1). The general input format of this method is `approximateVanishingIdeal(points::Array{Array{Float64,1},1}, listOfDegrees::Array{Int64,1})`.
