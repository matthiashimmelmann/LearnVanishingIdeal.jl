include("LearnVanishingIdeal.jl")
using HomotopyContinuation
@var x y z
points = []
open("prestress_stable_points.poly") do f
    # line_number
    line = 0  
    s = readline(f)          
    # read till end of file
    while ! eof(f)  
        s = readline(f)          
        # read a new / next line for every iteration           
        line += 1
        if s == "POLYS"
            break
        end
        
        L = split(split(s, ": ")[2], " ")
        push!(points, [parse(Float64, L[1]), parse(Float64, L[2]), parse(Float64, L[3])])
    end
end

result, sampserror = LearnVanishingIdeal.approximateVanishingIdeal(points, [2])
[[real(round(entry,digits=3)) for entry in value]'*LearnVanishingIdeal.affineVeronese(2,[x,y,z]) for value in result]