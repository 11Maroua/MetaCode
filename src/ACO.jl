using LinearAlgebra
using Random

include("loadSPP.jl")
include("Glouton.jl")     
include("Exploration.jl") 


##############################
# Roulette biaisée
##############################
function roulette_biaise(weights::AbstractVector{<:Real})
    total = sum(weights)
    if total <= 0
        return 1
    end
    r = rand() * total
    s = 0.0
    @inbounds for (i,w) in enumerate(weights)
        s += w
        if s >= r
            return i
        end
    end
    return length(weights)
end


##############################
# Mise à jour des phéromones
##############################
function update_pheromones!(pher::Vector{Float64},
                            best_iter_sol::Vector{Int},
                            rhoE::Float64,
                            rhoD::Float64)
    @inbounds for i in 1:length(pher)
        pher[i] *= rhoE
        if best_iter_sol[i] == 1
            pher[i] += rhoD
        end
    end
end


##############################
# Perturbation des phéromones
##############################
function disturb_pheromones!(pher::Vector{Float64},
                             iter::Int,
                             maxIter::Int)

    L = length(pher)
    factor = 0.95 * log10(iter) / log10(maxIter)
    lo = 0.05
    hi_base = 0.5 * (1 - iter / maxIter)

    @inbounds for i in 1:L
        pher[i] *= factor
    end

    K = rand(0:round(Int, 0.1 * L))
    @inbounds for _ in 1:K
        idx = rand(1:L)
        hi = hi_base
        pher[idx] = lo + rand() * (hi - lo)
    end

    @inbounds for i in 1:L
        if pher[i] < 0.1
            hi = hi_base
            pher[i] += lo + rand() * (hi - lo)
        end
    end
end


##############################
# Pré-calcul des conflits
# conf[j] = liste des variables i en conflit avec j
##############################
function precompute_conflicts(A::AbstractMatrix{<:Integer})
    m, n = size(A)
    conflicts = Vector{Vector{Int}}(undef, n)
    @inbounds for j in 1:n
        list = Int[]
        for i in 1:n
            i == j && continue
            for k in 1:m
                if A[k,j] == 1 && A[k,i] == 1
                    push!(list, i)
                    break
                end
            end
        end
        conflicts[j] = list
    end
    return conflicts
end


##############################
# Construction - Exploration
# (in-place, zéro allocation dans la boucle)
##############################
function build_solution_exploration!(
    x::Vector{Int},
    C::AbstractVector,
    A::AbstractMatrix{<:Integer},
    pher::Vector{Float64},
    conf::Vector{Vector{Int}},
    eta::AbstractVector,
    I::BitVector,
    candidats::Vector{Int},
    attractivites::Vector{Float64},
    iter::Int,
    maxIter::Int;
    alpha::Float64 = 1.0,
    beta::Float64 = 2.0
)
    m, n = size(A)

    @inbounds for j in 1:n
        x[j] = 0
        I[j] = true
    end

    P = log10(iter) / log10(maxIter)

    while true
        nbC = 0
        best_j = -1
        best_val = -Inf

        @inbounds for j in 1:n
            if I[j]
                val = (pher[j]^alpha) * (eta[j]^beta)
                nbC += 1
                candidats[nbC] = j
                attractivites[nbC] = val
                if val > best_val
                    best_val = val
                    best_j = j
                end
            end
        end

        nbC == 0 && break

        j_choisi = if rand() < P
            best_j
        else
            idx = roulette_biaise(@view attractivites[1:nbC])
            candidats[idx]
        end

        x[j_choisi] = 1
        I[j_choisi] = false

        @inbounds for t in eachindex(conf[j_choisi])
            j_conf = conf[j_choisi][t]
            I[j_conf] = false
        end
    end

    return nothing
end


##############################
# Construction - Exploitation
# (in-place, zéro allocation dans la boucle)
##############################
function build_solution_exploitation!(
    x::Vector{Int},
    C::AbstractVector,
    A::AbstractMatrix{<:Integer},
    pher::Vector{Float64},
    conf::Vector{Vector{Int}},
    I::BitVector;
    alpha::Float64 = 1.0
)
    m, n = size(A)

    @inbounds for j in 1:n
        x[j] = 0
        I[j] = true
    end

    while true
        best_j = -1
        best_val = -Inf

        @inbounds for j in 1:n
            if I[j]
                val = pher[j]^alpha
                if val > best_val
                    best_val = val
                    best_j = j
                end
            end
        end

        best_j == -1 && break

        x[best_j] = 1
        I[best_j] = false

        @inbounds for t in eachindex(conf[best_j])
            j_conf = conf[best_j][t]
            I[j_conf] = false
        end
    end

    return nothing
end


function ACO_SPP(C, A, conf;
                 maxIter::Int = 200,
                 maxAnt::Int = 15,
                 rhoE::Float64 = 0.8,
                 phiInit::Float64 = 1.0,
                 iterOnExploit::Float64 = 0.75,
                 iterStagnant::Int = 8,
                 do_local_search::Bool = true,
                 verbose::Bool = true)

    n = length(C)
    pher = fill(Float64(phiInit), n)

    x = zeros(Int, n)
    I = trues(n)
    candidats = Vector{Int}(undef, n)
    attractivites = Vector{Float64}(undef, n)
    eta = [utilite_variable(C, A, j) for j in 1:n]

    best_global = zeros(Int, n)
    best_global_val = -Inf
    last_improvement = 0

    if verbose
        println("=======================================================")
        println("               ACO pour SPP (version optimisée)")
        println("=======================================================")
        println("Variables     : $n")
        println("Fourmis       : $maxAnt")
        println("Itérations    : $maxIter")
        println("(evap)     : $rhoE")
        println("(init)     : $phiInit")
        println("Exploitation  : toutes les $(round(Int,iterOnExploit*maxIter)) itérations")
        println("Descente locale : $(do_local_search ? "Activée" : "Désactivée")")
        println("-------------------------------------------------------")
        println("Phéromones initiales : min=$(minimum(pher)), max=$(maximum(pher))")
    end

    for iter in 1:maxIter

        verbose && println("\n================ ITERATION $iter / $maxIter ================")

        best_iter = zeros(Int, n)
        best_iter_val = -Inf

        for ant in 1:maxAnt
            exploit = (ant == 1) ||
                      (iter % round(Int, iterOnExploit * maxIter) == 0)

            if verbose
                print("  Fourmi $ant : ")
                println(exploit ? "EXPLÉOITATION" : "exploration")
            end

            # Construction
            if exploit
                build_solution_exploitation!(x, C, A, pher, conf, I)
            else
                build_solution_exploration!(x, C, A, pher, conf, eta,
                                            I, candidats, attractivites,
                                            iter, maxIter)
            end

            # Descente locale
            if do_local_search
                val_before = dot(C, x)
                x = descente_simple(x, C, A,verbose=false)
                val_after = dot(C, x)
                if verbose && val_after != val_before
                    println("     Descente locale : $val_before → $val_after")
                end
            end

            # Evaluation
            val = dot(C, x)
            verbose && println("     Valeur solution = $val")

            # Best iteration
            if val > best_iter_val
                best_iter_val = val
                @inbounds best_iter .= x
            end

            # Best global
            if val > best_global_val
                best_global_val = val
                @inbounds best_global .= x
                last_improvement = iter
                verbose && println("     >>> Nouvelle meilleure globale = $val")
            end
        end

        verbose && println("  → Best iteration $iter = $best_iter_val")

        # Mise à jour phéromones
        rhoD = phiInit * (1 - rhoE)
        update_pheromones!(pher, best_iter, rhoE, rhoD)

        verbose && println("  Phéromones : min=$(minimum(pher)), max=$(maximum(pher))")

        # Perturbation 
        stagnant = (iter - last_improvement >= iterStagnant)
        exists_zero = any(p -> p < 0.001, pher)
        enough_time = iter < 0.9 * maxIter

        if stagnant && exists_zero && enough_time
            verbose && println("  *** Perturbation activée (stagnation détectée) ***")
            pher_before_min = minimum(pher)
            pher_before_max = maximum(pher)

            disturb_pheromones!(pher, iter, maxIter)

            verbose && println("  Phéromones avant : min=$pher_before_min, max=$pher_before_max")
            verbose && println("  Phéromones après : min=$(minimum(pher)), max=$(maximum(pher))")
        end
    end

    verbose && println("\n====================== FIN ACO ======================")
    verbose && println("Meilleure valeur globale = $best_global_val")
    verbose && println("======================================================")

    return best_global, best_global_val
end



#= C, A = loadSPP("../dat/pb_2000rnd0100.dat")
 conf = precompute_conflicts(A)

 best_sol, best_val =
     @time ACO_SPP(C, A, conf;
                   maxIter = 20,
                   maxAnt = 10,
                   rhoE = 0.85,
                  phiInit = 1.0,
                  iterOnExploit = 0.75,
                  iterStagnant = 10,
                  do_local_search = true,
                  verbose = false)

 println("Best value = ", best_val) =#


function experimentation_ACO(dir::String)

    # Liste tous les fichiers du dossier
    files = readdir(dir)

    println("===================================================")
    println(" Lancement des expérimentations ACO sur le dossier : $dir")
    println(" Fichiers détectés : ", length(files))
    println("===================================================\n")

    for file in files
        
        # Ignore les fichiers qui ne sont pas des .dat (optionnel)
        endswith(file, ".dat") || continue

        path = joinpath(dir, file)

        println("\n---------------- FICHIER : $file ----------------")

        try
            C, A = loadSPP(path)
            conf = precompute_conflicts(A)

            @elapsed best_sol, best_val = ACO_SPP(
                C, A, conf;
                maxIter = 30,
                maxAnt = 15,
                rhoE = 0.8,
                phiInit = 1.0,
                iterOnExploit = 0.75,
                iterStagnant = 8,
                do_local_search = true,
                verbose = false
            )

            println("Résultat pour $file : Best value = $best_val")

        catch e
            println("   ERREUR lors du traitement de $file :")
            println(e)
            continue
        end
    end

    println("\n===================================================")
    println("       EXPÉRIMENTATION ACO TERMINÉE")
    println("===================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    experimentation_ACO("../dat/")
end
