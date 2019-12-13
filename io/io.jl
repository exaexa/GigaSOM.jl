
using GigaSOM, DataFrames, XLSX, Test, Random, Distributed, SHA, JSON

checkDir()
cwd = pwd()

# dataPath = "/Users/ohunewald/work/data_felD1/"
dataPath = "artificial_data_cytof/"
cd(dataPath)
md = DataFrame(XLSX.readtable("metadata.xlsx", "Sheet1", infer_eltypes=true)...)
panel = DataFrame(XLSX.readtable("panel.xlsx", "Sheet1", infer_eltypes=true)...)

lineageMarkers, functionalMarkers = getMarkers(panel)


nWorkers = 2
addprocs(nWorkers, topology=:master_worker)
@everywhere using GigaSOM, FCSFiles

@info "processes added"

# read directory
content = md.file_name
L = length(content)

R =  Vector{Any}(undef,L)

N = convert(Int64, (length(md.file_name)/nWorkers))

#R[1] = loadData(1, content[1], md, panel)

@time @sync for (idx, pid) in enumerate(workers())
    @async for k in (idx-1)*N+1:idx*N
         R[k] = fetch(@spawnat pid loadData(idx, content[k], md, panel))
    end
end

workerIDs = workers()

#Rc = Vector{Any}(undef, nWorkers)
Rc = [Ref[], Ref[]]
Rc1 = fill(Ref[],nWorkers)
#Rc2 = Array{Array{Ref,1},1}
Rc == Rc1
for k in 1:L
    id = R[k][2]
    localID = findall(isequal(id), workerIDs)
    push!(Rc[localID[1]], R[k][1])
    push!(Rc1[localID[1]], R[k][1])
    #push!(Rc2[localID[1]], R[k][1])
end
Rc == Rc1

#=
#@time @sync for (idx, pid) in enumerate(workers())
#    @async R[idx] = fetch(@spawnat pid begin loadData(md.file_name[(idx-1)*N+1:idx*N],md, panel) end)
#end

# Get n random samples from m workers in R:
samplesPerWorker = Int(length(R))
# sampleList defines wich data in R ref has to return one random sample
# TODO: put this into core.jl
sampleList = rand(1:samplesPerWorker, 100)
X = zeros(100, length(lineageMarkers))

for i in 1:length(sampleList)
    element = sampleList[i]
    # dereference and get one random sample from matrix
    Y = R[element].x[rand(1:size(R[element].x, 1), 1),:]
    # convert Y into vector
    X[i, :] = vec(Y)
end

som = initGigaSOM(X, 10, 10)

#------ trainGigaSOM() -------------------------
# define the columns to be used for som training
cc = map(Symbol, lineageMarkers)
# R holds the reference to the dataset for each worker
#@time som = trainGigaSOM(som, R, cc)
=#
rmprocs(workers())
