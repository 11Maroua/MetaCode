using LinearAlgebra
using Statistics
include("loadSPP.jl")
include("Exploration.jl")
include("Glouton.jl")


############################################
# Utilité d'une variable
############################################
function utilite_variable(C, A, j)
    s = sum(A[:, j])
    return s > 0 ? C[j] / s : C[j]
end

############################################
# Roulette biaisée
############################################
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

############################################
# Mise à jour des phéromones
############################################
function update_pheromones!(pher, best_iter_sol, rhoE, rhoD)
    @inbounds for i in 1:length(pher)
        pher[i] *= rhoE
        if best_iter_sol[i] == 1
            pher[i] += rhoD
        end
    end
end

############################################
# Perturbation
############################################
function disturb_pheromones!(pher, iter, maxIter)
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

############################################
# Pré-calcul des conflits
############################################
function precompute_conflicts(A)
    m, n = size(A)
    conflicts = Vector{Vector{Int}}(undef, n)
    @inbounds for j in 1:n
        list = Int[]
        for i in 1:n
            if i != j
                for k in 1:m
                    if A[k,j] == 1 && A[k,i] == 1
                        push!(list, i)
                        break
                    end
                end
            end
        end
        conflicts[j] = list
    end
    return conflicts
end

############################################
# Construction exploration
############################################
function build_solution_exploration!(
    x, C, A, pher, conf, eta,
    I, candidats, attractivites,
    iter, maxIter;
    alpha=1.0,
    beta=2.0
)

    n = length(C)
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

        j_sel = if rand() < P
            best_j
        else
            idx = roulette_biaise(@view attractivites[1:nbC])
            candidats[idx]
        end

        x[j_sel] = 1
        I[j_sel] = false

        @inbounds for j_conf in conf[j_sel]
            I[j_conf] = false
        end
    end

    return nothing
end

############################################
# Construction exploitation
############################################
function build_solution_exploitation!(
    x, C, A, pher, conf, I;
    alpha=1.0
)
    n = length(C)

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
        @inbounds for j_conf in conf[best_j]
            I[j_conf] = false
        end
    end

    return nothing
end

############################################
# ACO COMPLET AVEC MESURE PAR ITERATIONS
############################################
function ACO_SPP(C, A, conf;
                 maxIter=30,
                 maxAnt=15,
                 rhoE=0.8,
                 phiInit=1.0,
                 iterOnExploit=0.75,
                 iterStagnant=8,
                 do_local_search=true,
                 verbose=true,
                 measure_every=5)

    n = length(C)
    pher = fill(phiInit, n)

    x = zeros(Int, n)
    I = trues(n)
    candidats = Vector{Int}(undef, n)
    attractivites = Vector{Float64}(undef, n)
    eta = [utilite_variable(C, A, j) for j in 1:n]

    vals_fourmis = Vector{Float64}(undef, maxAnt)
    history = Vector{Tuple{Int,Float64,Float64,Float64}}()

    best_global = zeros(Int, n)
    best_global_val = -Inf
    last_improvement = 0

    for iter in 1:maxIter

        best_iter_val = -Inf
        best_iter = zeros(Int, n)

        for ant in 1:maxAnt
            exploit = (ant == 1) ||
                      (iter % round(Int, iterOnExploit * maxIter) == 0)

            if exploit
                build_solution_exploitation!(x, C, A, pher, conf, I)
            else
                build_solution_exploration!(x, C, A, pher, conf, eta,
                                            I, candidats, attractivites,
                                            iter, maxIter)
            end

            if do_local_search
                x = descente_simple(x, C, A, verbose=false)
            end

            val = dot(C, x)
            vals_fourmis[ant] = val

            if val > best_iter_val
                best_iter_val = val
                best_iter .= x
            end
            if val > best_global_val
                best_global_val = val
                best_global .= x
                last_improvement = iter
            end
        end

        rhoD = phiInit * (1 - rhoE)
        update_pheromones!(pher, best_iter, rhoE, rhoD)

        if iter - last_improvement >= iterStagnant &&
           iter < 0.9 * maxIter
            disturb_pheromones!(pher, iter, maxIter)
        end

        # --- Mesure basée sur les itérations ---
        if iter % measure_every == 0
            zmin = minimum(vals_fourmis)
            zmax = maximum(vals_fourmis)
            zmoy = mean(vals_fourmis)
            push!(history, (iter, zmin, zmax, zmoy))
        end
    end

    return best_global, best_global_val, history
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



using Plots

function plot_ACO_history(filename::String, history)
    if isempty(history)
        println("Aucun historique pour $filename — plot ignoré.")
        return
    end

    # Extraction
    iters = [h[1] for h in history]
    zmin  = [h[2] for h in history]
    zmax  = [h[3] for h in history]
    zmoy  = [h[4] for h in history]

    # === Courbes ===
    p = plot(
        iters, zmax;
        label = "zMax",
        color = :red,
        lw = 2,
    )
    plot!(
        p, iters, zmoy;
        label = "zMoy",
        color = :green,
        lw = 2,
    )
    plot!(
        p, iters, zmin;
        label = "zMin",
        color = :blue,
        lw = 2,
    )

    # Mise en forme
    xlabel!("Itérations ACO")
    ylabel!("Valeurs de z(x)")
    title!("ACO-SPP | z_min, z_moy, z_max | $(filename)")
    plot!(p, grid = true)
    plot!(p, legend = :bottomright)   # <-- correction ici

    # Sauvegarde
    output = "../doc/plot_$(filename).png"
    savefig(p, output)
    println("Plot sauvegardé : $output")
end





function experimentation_ACO(dir::String)

    files = readdir(dir)

    # Fichier CSV global unique
    out_file = "../doc/aco_results_all.csv"

    # Écrire l'en-tête une seule fois
    open(out_file, "w") do io
        println(io, "filename,time,z_best")
    end

    println("===================================================")
    println(" Lancement des expérimentations ACO sur le dossier : $dir")
    println("===================================================\n")

    for file in files

        endswith(file, ".dat") || continue
        path = joinpath(dir, file)

        println("\n---------------- FICHIER : $file ----------------")

        try
            C, A = loadSPP(path)
            conf = precompute_conflicts(A)

            # --- Exécution ACO ---
            time_ACO = @elapsed best_sol, best_val, _ = ACO_SPP(
                C, A, conf;
                maxIter = 50,
                maxAnt = 25,
                rhoE = 0.8,
                phiInit = 1.0,
                iterOnExploit = 0.75,
                iterStagnant = 8,
                do_local_search = true,
                verbose = false,
                measure_every = 10_000_000   # désactivation de l’history
            )

            println("Résultat pour $file : Best value = $best_val")

            # --- Ajouter une ligne dans le fichier global ---
            open(out_file, "a") do io
                println(io, "$file,$time_ACO,$best_val")
            end

        catch e
            println("   ERREUR lors du traitement de $file :")
            println(e)
        end
    end

    println("\n===================================================")
    println("   EXPÉRIMENTATION ACO TERMINÉE — CSV généré :")
    println("   $out_file")
    println("===================================================")
end


experimentation_ACO("../dat/")