function mydot(a, b)
    acc = zero(eltype(a))
    for i in eachindex(a)
        acc += a[i]*b[i]
    end
    acc
end

function matmul!(c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where T
    for i in eachindex(c)
        c[i] = 0
    end
    m, n = size(A)
    @inbounds begin
        for j in 1:n
            for i in 1:m
                c[i] += A[i, j] * b[j]
            end
        end
    end
    c
end

function matmul(A::AbstractMatrix{T}, b::AbstractVector{T}) where T
    m, n = size(A)
    c = similar(b, m)
    matmul!(c, A, b)
end

function matmul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    m, n = size(A)
    p = size(B, 2)
    @inbounds for i in eachindex(C)
        C[i] = 0
    end
    @inbounds for i in 1:m
        for k in 1:p
            for j in 1:n
                C[i, k] += A[i, j] * B[j, k]
            end
        end
    end
    C
end

function matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    m = size(A, 1)
    p = size(B, 2)
    C = similar(A, m, p)
    matmul!(C, A, B)
end

function mylu!(A::AbstractMatrix{T}) where T
    m, n = size(A)
    minmn = min(m, n)
    pidx = Vector{Int}(minmn)
    for k in 1:minmn
        # find the largest pivot
        maxidx = k
        maxval = abs(A[k, k])
        for j in k:m
            val = abs(A[j, k])
            val > maxval && (maxval = val; maxidx = j)
        end
        pidx[k] = maxidx
        # swap rows if necessary
        if maxidx != k
            for j in 1:n
                A[maxidx, j], A[k, j] = A[k, j], A[maxidx, j]
            end
        end
        pinv = inv(A[k, k]) # inverse of the pivot
        # zero out the k-th col
        for j in k+1:m
            A[j, k] *= pinv
        end
        # update
        for j in k+1:m
            for i in k+1:n
                A[j, i] -= A[j, k] * A[k, i]
            end
            
        end
    end
    LinAlg.UnitLowerTriangular(A), UpperTriangular(A), pidx
end

function pidx2perm(pidx)
    n = length(pidx)
    perm = collect(1:n)
    for i in eachindex(pidx)
        perm[pidx[i]], perm[i] = perm[i], perm[pidx[i]]
    end
    perm
end

function chol_!(A::AbstractMatrix{T}) where T
    @views @inbounds begin
        m, n = size(A)
        @assert m==n
        alpha = A[1, 1] = sqrt(A[1, 1])
        if n > 1
            w = scale!(A[2:n, 1], inv(alpha))
            K = A[2:n, 2:n]
            for j in eachindex(w), i in j:n-1
                K[i, j] -= w[i]*w[j]'
            end
            chol_!(K)
        end
        return LowerTriangular(A)
    end
end
chol_(A) = chol_!(copy(A))
