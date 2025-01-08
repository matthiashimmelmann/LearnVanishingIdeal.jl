using HomotopyContinuation, Plots
totalarray=[]
open("dataCSVdeltan4d10qn4d20complexPointsZ1Z2Z3.txt") do f
    # read till end of file
    while ! eof(f)  
       s = readline(f)
       ar = []
       for st in split(s, ",")
            cs=collect(st)
            cs = [x for x in cs if x!='(' && x!=')']
            result = join(cs)
            if length(split(result, "+"))==2
                push!(ar, complex(parse(Float64, split(result, "+")[1]), parse(Float64, split(result, "+")[2][1:end-1])))
            elseif length(split(result, "-"))==2
                push!(ar, complex(parse(Float64, split(result, "-")[1]), -parse(Float64, split(result, "-")[2][1:end-1])))
            elseif result[1]=="-" && length(split(result, "+"))==2
                push!(ar, complex(-parse(Float64, split(result, "+")[1]), parse(Float64, split(result, "+")[2][1:end-1])))
            else
                push!(ar, complex(-parse(Float64, split(result, "-")[2]), -parse(Float64, split(result, "-")[3][1:end-1])))
            end
        end
        push!(totalarray,ar)
    end
end
include("LearnVanishingIdeal.jl")
result, sampserror = LearnVanishingIdeal.approximateVanishingIdeal(totalarray, [4], quick=false, affine=false)
@var x,y,z
result = [[round(entry,digits=3) for entry in value]'*LearnVanishingIdeal.projVeronese(4,[x,y,z]) for value in result][1]

@var a[1:3]
G = [result, a'*[x,y,z]]
a₀ = randn(ComplexF64, 3)
F = System(G; variables=[x,y,z], parameters=a)
start = solve(F; target_parameters=a₀)
start_sols = solutions(start)
k = 10^4

function map_step(R)
    R = R ./ norm(R)
    return [imag( (2*R[1]-R[2]-R[3]) / (R[1]+R[2]+R[3]) ), imag( sqrt(3)*(R[2]-R[3])/(R[1]+R[2]+R[3]) )]
end

#track towards k random linear spaces
points = solve(
    F,
    start_sols;
    start_parameters =  a₀,
    target_parameters = [randn(ComplexF64, 3) for _ in 1:k],
    transform_result = (R,p) -> map_step(p)
)
filter!(p -> !(Inf in p), points)
orig_points = [map_step(ar) for ar in totalarray]

sample_matrix = hcat(points...)
orig_sample_matrix = hcat(orig_points...)

plt = scatter(sample_matrix[1,:], sample_matrix[2,:],
        legend = false, grid = false, framestyle = :origin, markersize=1, markercolor = :grey, ylims = (-3,3), xlims = (-3,3), size=(800,800))
scatter!(plt, orig_sample_matrix[1,:], orig_sample_matrix[2,:],
        markersize=3, markercolor = :steelblue)
