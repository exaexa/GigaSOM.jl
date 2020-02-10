"""
    save_at(worker, sym, val)

Saves value `val` to symbol `sym` at `worker`. `sym` should be quoted (or
contain a symbol). `val` gets unquoted in the processing and evaluated at the
worker, quote it if you want to pass exact command to the worker.

Examples:
    addprocs(1)
    save_at(2,:x,123)       # saves 123
    save_at(2,:x,myid())    # saves 1
    save_at(2,:x,:(myid())) # saves 2
    save_at(2,:x,:(:x))     # saves the symbol :x (just :x won't work because of unquoting)
"""
function save_at(worker, sym::Symbol, val)
    remote_do(()->eval(:($sym = $val)), worker)
end

"""
    get_from(worker,val)

Get a value `val` from a remote `worker`; quoting of `val` works just as with
`save_at`. Returns a future.
"""
function get_from(worker, val)
    remotecall(()->eval(:($val)), worker)
end

"""
    get_val_from(worker,val)

Shortcut for instantly fetching the future from `get_from`.
"""
function get_val_from(worker, val)
    fetch(get_from(worker, val))
end

"""
    remove_from(worker,sym)

Sets symbol `sym` on `worker` to `nothing`, effectively freeing the data.
"""
function remove_from(worker, sym::Symbol)
    save_at(worker, sym, nothing)
end

"""
    distribute(sym, dd::DArray)

Distribute the distributed array parts from `dd` into worker-local variable
`sym`.

Requires @everywhere import DistributedArrays.
"""
function distribute_data(sym::Symbol, dd::DArray)
    for pid in dd.pids
        save_at(pid, sym, :(DistributedArrays.localpart($dd)))
    end
end

"""
    undistribute(sym, dd::DArray)

Removes the worker-local data created by the corresponding call of `distribute`
"""
function undistribute_data(sym::Symbol, dd::Darray)
    for pid in dd.pids
        remove_from(pid, sym)
    end
end

"""
Load filenames to each referenced worker.

TODO: reduce the arguments a bit (we have transform now!)
TODO: remove the Ref when it's not needed
"""
function distribute_jls_data(sym::Symbol, fns::Array{String}, workers;
    panel=Nothing(), transform="asinh", cofactor=5, reduce=false, sort=false, transform=false)
    for (i, pid) in workers
        fn = fns[i]
        save_at(pid, sym, :(
            loadDataFile($i, $fn,
                         $panel, $method, $cofactor,
                         $reduce, $sort, $transform).x
            ))
    end
end

"""
    undistribute_data(sym, workers)

Remove the loaded data from workers.
"""
function undistribute_data(sym, workers)
    for pid in workers
        remove_from(pid,sym)
    end
end

"""
    distributed_transform(sym::Symbol, fn::Function, workers)

Transform the worker-local distributed data stored as `sym` on `workers`
in-place, by a function `fn`.

# Example
    
    distributed_transform(:myData, (d)->(2*d), workers())
"""
function distributed_transform(sym::Symbol, fn, workers)
    futures = [ remote_do(()->eval(:(begin; $sym = $fn($sym); nothing; end)), pid)
        for pid in workers ]
    fetch.(futures)  #wait until all threads finish
    nothing
end

"""
    distributed_mapreduce(val, fn::Function, workers)

Run `map`s (non-modifying transforms on the data) and `fold`s (2-to-1
reductions) on the worker-local data (in `val`s) distributed on `workers` and
return the final reduced result.

It is assumed that the fold operation is associative, but not commutative (as
in semigroups). If there are no workers, operation returns `nothing` (we don't
have a monoid to magically conjure zero elements :[ ).

In current version, the reduce step is a sequential left fold, executed in the
main process.

# Example
    # compute a mean of the distributed array
    sum,len = distributed_mapreduce(:myData,
        (d) -> (sum(d),length(d)),
        ((s1, l1), (s2, l2)) -> (s1+s2, l1+l2),
        workers())
    println(sum/len)
"""
function distributed_mapreduce(val, map, fold, workers)
    if isempty(workers)
        return nothing
    end

    futures = [remotecall(()->eval(:($fn($sym))), pid) for pid in workers ]
    res = fetch(futures[1])
    for i in 2:length(futures)
        res = fold(res, fetch(futures[i]))
    end
    res
end