using LinearAlgebra
using Random
include("loadSPP.jl")
include("Glouton.jl")
include("Exploration.jl")

# Valeur d'une solution (Z = C^T x)
function valeur_solution(C, x)
    return dot(C, x)
end

# Heuristique gloutonne pour la variable j (extraite de construction_gloutonne)
function utilite_variable(C, A, j)
    nb_contraintes = sum(A[:, j])
    if nb_contraintes > 0
        return C[j] / nb_contraintes
    else
        return C[j]  # même cas particulier que dans ton glouton
    end
end

# Construction d'une solution initiale par une heuristique gloutonne randomisée

function tirage_roulette(candidats::Vector{Int}, poids::Vector{Float64})
    s = sum(poids)
    if s <= 0
        # sécurité : si tout est nul/neg, choisir uniformément
        return rand(candidats)
    end
    r = rand() * s
    cumul = 0.0
    for (idx, p) in enumerate(poids)
        cumul += p
        if cumul >= r
            return candidats[idx]
        end
    end
    return candidats[end]  # sécurité numérique
end



function construction_par_fourmi(C, A, tau;
                                alpha::Float64 = 1.0,
                                beta::Float64  = 2.0,
                                q0::Float64    = 0.9)

    m, n = size(A)
    x = zeros(Int, n)
    variables_restantes = collect(1:n)

    while !isempty(variables_restantes)
        candidats = Int[]
        attractivites = Float64[]
        meilleur_j = -1
        meilleure_attractivite = -Inf

        # Évaluer toutes les variables restantes
        for j in variables_restantes
            if peut_ajouter(x, A, j)
                u = utilite_variable(C, A, j)

                # Si utilité ≤ 0, on peut choisir de l'ignorer (sauf si tu veux vraiment l'autoriser)
                if u <= 0
                    continue
                end

                # Heuristique η_j = utilité gloutonne
                eta = u

                # Attractivité combinant phéromone et heuristique
                val = (tau[j]^alpha) * (eta^beta)

                push!(candidats, j)
                push!(attractivites, val)

                if val > meilleure_attractivite
                    meilleure_attractivite = val
                    meilleur_j = j
                end
            end
        end

        if isempty(candidats)
            # plus aucune variable ajoutable sans violer les contraintes
            break
        end

        # --- Règle ACS : exploitation vs exploration ---
        if rand() < q0
            # exploitation : choisir le meilleur candidat
            j_choisi = meilleur_j
        else
            # exploration : tirage roulette sur les attractivités
            j_choisi = tirage_roulette(candidats, attractivites)
        end

        # Ajout de la variable choisie
        x[j_choisi] = 1

        # Retirer j_choisi des candidats
        idx_supprimer = findfirst(==(j_choisi), variables_restantes)
        deleteat!(variables_restantes, idx_supprimer)
    end

    return x
end


function ACO_SPP(C, A;
                 nb_iterations::Int = 20,
                 nb_fourmis::Int    = 10,
                 alpha::Float64     = 1.0,
                 beta::Float64      = 2.0,
                 rho::Float64       = 0.1,    # évaporation
                 Q::Float64         = 0.01,   # intensité dépôt phéromone
                 tau0::Float64      = 1.0,
                 tau_min::Float64   = 1e-6,
                 tau_max::Float64   = 10.0,
                 max_restarts_ls::Int = 1)    # pour descente_multi_start

    m, n = size(A)

    # --- Initialisation des phéromones ---
    tau = fill(tau0, n)

    # Optionnel : initialiser une solution gloutonne pour avoir une bonne base
    x_glouton = construction_gloutonne(C, A)
    z_glouton = valeur_solution(C, x_glouton)

    meilleure_solution_globale = copy(x_glouton)
    meilleure_valeur_globale = z_glouton

    println("Initialisation ACO : Z_glouton = ", z_glouton)

    for it in 1:nb_iterations
        println("\n==================== ITERATION ACO $it / $nb_iterations ====================")
        solutions = Vector{Vector{Int}}()
        valeurs = Float64[]

        for k in 1:nb_fourmis
            println("\n--- Fourmi $k / $nb_fourmis ---")

            # 1) Construction par fourmi
            x = construction_par_fourmi(C, A, tau; alpha=alpha, beta=beta)

            # 2) Descente locale multi-start (hybridation ACO + LS)
            #    Si tu veux limiter le temps, mets max_restarts_ls=1 ou 2
            x = descente_multi_start(x, C, A; max_restarts=max_restarts_ls)

            v = valeur_solution(C, x)
            push!(solutions, x)
            push!(valeurs, v)

            if v > meilleure_valeur_globale
                meilleure_valeur_globale = v
                meilleure_solution_globale = copy(x)
                println(">>> NOUVELLE MEILLEURE SOLUTION GLOBALE : Z = ", v)
            end
        end

        # --- Évaporation globale des phéromones ---
        for j in 1:n
            tau[j] = (1.0 - rho) * tau[j]
        end

        # --- Dépôt de phéromones par la meilleure solution globale (schéma élitiste) ---
        for j in findall(==(1), meilleure_solution_globale)
            # Comme on MAXIMISE, déposer proportionnellement à Z
            tau[j] += Q * max(meilleure_valeur_globale, 0.0)
        end

        # --- Bornes sur tau (pour éviter explosion / disparition) ---
        for j in 1:n
            if tau[j] < tau_min
                tau[j] = tau_min
            elseif tau[j] > tau_max
                tau[j] = tau_max
            end
        end

        println("Fin itération $it : meilleure valeur globale = ", meilleure_valeur_globale)
    end

    return meilleure_solution_globale, meilleure_valeur_globale, tau
end


C, A = loadSPP("Data/pb_200rnd0100.dat")
meilleure_sol, meilleure_valeur, tau_final =
    @time ACO_SPP(C, A;
            nb_iterations=10,
            nb_fourmis=5,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            Q=0.01,
            max_restarts_ls=1) 
