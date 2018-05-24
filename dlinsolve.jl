function mydot(a, b)
    acc = zero(eltype(a))
    for i in eachindex(a)
        acc += a[i]*b[i]
    end
    acc
end

function forwardsub!(L::AbstractMatrix{T}, b, x = b) where T
    n = LinAlg.checksquare(L)
    @assert n == length(b)
    for i in 1:n
        invli = inv(L[i, i])
        x[i] = b[i]
        for j in 1:i-1
            x[i] -= L[i, j]*x[j]
        end
        x[i] *= invli
    end
    x
end
forwardsub(L, b) = forwardsub!(L, copy(b))

function backwardsub!(U::AbstractMatrix{T}, b, x = b) where T
    n = LinAlg.checksquare(U)
    @assert n == length(b)
    for i in n:-1:1
        invui = inv(U[i, i])
        x[i] = b[i]
        for j in i+1:n
            x[i] -= U[i, j]*x[j]
        end
        x[i] *= invui
    end
    x
end
backwardsub(U, b) = backwardsub!(U, copy(b))

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
        p_inv = inv(A[k, k])
        for j in k+1:m   # i, j sequence?
            A[j, k] *= p_inv
        end
        for j in k+1:m
            for i in k+1:n
                A[j, i] -= A[j, k]*A[k, i]
            end
        end
    end
    LinAlg.UnitLowerTriangular(A), UpperTriangular(A), pidx
end

function pidx2perm!(dest, pidx)
    @assert length(dest) == length(pidx)
    for i in eachindex(pidx)
        dest[i], dest[pidx[i]] = dest[pidx[i]], dest[i]
    end
    dest
end
pidx2perm(pidx) = pidx2perm!(collect(1:endof(pidx)), pidx)

lu_(A) = mylu!(copy(A))

function linsolve!(A, b)
    L, U, p = mylu!(A)
    pidx2perm!(b, p)
    forwardsub!(L, b)
    backwardsub!(U, b)
end
linsolve(A, b) = linsolve!(copy(A), copy(b))

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

#######################
# Tests
#######################
using Base.Test

@testset "Substitution Tests" begin
    A = rand(5,5)
    L = LowerTriangular(A)
    U = UpperTriangular(A)
    b = rand(5)
    @test b ≈ L*forwardsub(L, b)
    @test b ≈ U*backwardsub(U, b)
end

@testset "LU Factorization Tests" begin
    A = rand(5,5)
    L, U, p = lu_(A)
    perm = pidx2perm(p)
    #println("---------- L -----------")
    #display(LowerTriangular(L))
    #println("---------- U -----------")
    #display(U)
    #println("----- pidx & perm ------")
    #display(hcat(p, perm))
    @test L*U ≈ A[perm, :]
end

@testset "Cholesky Factorization Tests" begin
    A = rand(5,5)
    A = A'A
    L = chol_(A)
    #println("---------- L -----------")
    #display(L)
    @test A ≈ L*L'
end

@testset "Linear Solve Tests" begin
    A = rand(5, 5)
    b = rand(5)
    x = linsolve(A, b)
    @test A*x ≈ b
end
