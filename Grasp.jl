using LinearAlgebra
using Statistics

include("loadSPP.jl")
include("Glouton.jl")
include("Exploration.jl")


# ==================== GRASP IMPLÉMENTATION ====================

function procedure_grasp(C, A, alpha)
    n = length(C)
    x = zeros(Int, n)
    candidate_set = collect(1:n)
    
    println("=== CONSTRUCTION GRASP (alpha = $(alpha)) ===")
    
    while !isempty(candidate_set)
        # Évaluer l'utilité de tous les candidats
        utilities = Float64[]
        valid_candidates = Int[]
        
        for j in candidate_set
            if peut_ajouter(x, A, j)
                nb_contraintes = sum(A[:, j])
                if nb_contraintes > 0
                    utilite = C[j] / nb_contraintes
                else
                    utilite = C[j]
                end
                push!(utilities, utilite)
                push!(valid_candidates, j)
            end
        end
        
        if isempty(valid_candidates)
            break
        end
        
        # Construire la RCL
        u_min = minimum(utilities)
        u_max = maximum(utilities)
        
        if u_min == u_max
            # Gestion du cas où toutes les utilités sont égales
            RCL = valid_candidates
        else
            u_limit = u_min + alpha * (u_max - u_min)
            RCL = valid_candidates[utilities .>= u_limit]
        end
        
        # On selectionne aléatoirement une variable de la RCL
        if !isempty(RCL)
            selected_idx = rand(1:length(RCL))
            j_selected = RCL[selected_idx]
            
            x[j_selected] = 1
            
            # Retrait de la variable selectionnée  de l'ensemble des candidats
            idx_to_remove = findfirst(==(j_selected), candidate_set)
            deleteat!(candidate_set, idx_to_remove)
            
            println("  Sélection: variable $(j_selected), utilité = $(utilities[findfirst(==(j_selected), valid_candidates)])")
        else
            break
        end
    end
    
    valeur_totale = dot(C, x)
    println("Solution construite: Z = $(valeur_totale), $(sum(x)) variables")
    
    return x
end

function recherche_locale(x, C, A; max_iter=100)
   
    println("=== RECHERCHE LOCALE ===")
    
    x_current = copy(x)
    z_current = dot(C, x_current)
    iteration = 0
    improved = true
    
    while improved && iteration < max_iter
        iteration += 1
        improved = false
        
        # Utilisation des fcts generer_voisinage du devoir1
        voisinages = [
            ("1-1", () -> generer_voisinage_1_1(x_current, C, A)),
            ("2-1", () -> generer_voisinage_2_1(x_current, C, A)),
            ("3-1", () -> generer_voisinage_3_1(x_current, C, A))
        ]
        
        for (nom_voisinage, generer_voisin) in voisinages
            x_voisin, gain, amelioration, nb_voisins = generer_voisin()
            
            if amelioration && gain > 0
                x_current = x_voisin
                z_current += gain
                improved = true
                println("  Itération $(iteration) - $(nom_voisinage): Z = $(z_current) (+$(round(gain, digits=2)))")
                break  # Recommencer avec la nouvelle solution
            end
        end
    end
    
    println("Recherche locale terminée: Z = $(z_current) (amélioration: $(round(z_current - dot(C, x), digits=2)))")
    
    return x_current
end

function grasp_complet(C, A; alpha=0.7, n_iter=100)
   
    
    println("\n============ Résolution SPP avec GRASP  =========")
    println("Paramètres: alpha=$(alpha), n_iter=$(n_iter)")
   
    
    x_best = zeros(Int, length(C))
    z_best = -Inf
    solutions_history = Float64[]
    improved_history = Float64[]
    best_history = Float64[]
    
    for iter in 1:n_iter
        if iter == 1
            println("Itération $(iter)/$(n_iter)")
        end
        
        # Phase de construction : calcul Ulimit + RCL
        x_constructed = procedure_grasp(C, A, alpha)
        z_constructed = dot(C, x_constructed)
        
        # Phase d'amélioration :recherche Locale
        x_improved = recherche_locale(x_constructed, C, A)
        z_improved = dot(C, x_improved)
        
        # Mise à jour de la meilleure solution
        if z_improved > z_best
            x_best = copy(x_improved)
            z_best = z_improved
            println("Nouvelle meilleure solution à l'itération $(iter): Z = $(z_best)")
            
        end
        
        push!(solutions_history, z_constructed)
        push!(improved_history, z_improved)
        push!(best_history, z_best)
    end
    
    
    println("\n========= MEILLEURE SOLUTION TROUVEE AVEC GRASP ==========")
    println("Meilleure solution: Z = $(z_best)")
    println("Variables sélectionnées: $(sum(x_best))")
    println("Paramètre alpha: $(alpha)")
   
    
    return x_best, solutions_history, improved_history, best_history
end

# ==================== REACTIVE GRASP ====================

mutable struct initialisation_ReactiveGRASP
    #vecteur de alpha 
    alpha_values::Vector{Float64}

    #probas de chaque alpha
    probabilities::Vector{Float64}

    #moyenne des capacités obtenues avec alpha choisi lors de la recherche locale
    avg_values::Vector{Float64}

    #nombre de fois où chaque alpha a été utilisé
    counts::Vector{Int}

    #frequence de mise à jour des probabilités
    update_frequency::Int
end

function initialisation_ReactiveGRASP(alpha_values::Vector{Float64}, update_frequency::Int=50)
    m = length(alpha_values)
    probabilities = ones(m) / m
    avg_values = zeros(m)
    counts = zeros(Int, m)
    
    return initialisation_ReactiveGRASP(alpha_values, probabilities, avg_values, counts, update_frequency)
end

function select_Random_alpha(reactive_grasp)
    
    cum_probs = cumsum(reactive_grasp.probabilities)
    r = rand()
    
    for i in 1:length(cum_probs)
        if r <= cum_probs[i]
            return reactive_grasp.alpha_values[i], i
        end
    end
    
    return reactive_grasp.alpha_values[end], length(cum_probs)
end

function update_probabilities(reactive_grasp, z_best, z_worst)
   
    m = length(reactive_grasp.alpha_values)
    q_values = zeros(m)
    
    for k in 1:m
        if reactive_grasp.counts[k] > 0
            if z_best != z_worst
                q_values[k] = (reactive_grasp.avg_values[k] - z_worst) / (z_best - z_worst)
            else
                q_values[k] = 1.0  
            end
        else
            q_values[k] = 0.0
        end
    end
    
    #calcul au fur et à mesure de somme des coeff de qualité q
    sum_q = sum(q_values)
    if sum_q > 0
        reactive_grasp.probabilities = q_values / sum_q
    else
        reactive_grasp.probabilities = ones(m) / m
    end
    
    reactive_grasp.avg_values .= 0.0
    reactive_grasp.counts .= 0
end

function reactive_grasp(C, A; alphas=[0.2, 0.5, 0.75, 0.9, 1.0], 
                                    n_iter=1000, update_frequency=50)
   
    println("\n========== REACTIVE GRASP =========")
    println("Valeurs alpha: $(alphas)")
    println("Itérations: $(n_iter), Fréquence mise à jour: $(update_frequency)")
    
    
    reactive_grasp = initialisation_ReactiveGRASP(alphas, update_frequency)
    
    x_best = zeros(Int, length(C))
    z_best = -Inf
    z_worst = Inf
    
    solutions_history = Float64[]
    improved_history = Float64[]
    best_history = Float64[]
    alpha_history = Float64[]
    
    for iter in 1:n_iter
        if iter % 100 == 0
            println("Itération $(iter)/$(n_iter)")
        end
        
        #selection random d'un alpha et calcul de ses stats

        alpha, alpha_index = select_Random_alpha(reactive_grasp)
        
        # Solution crée avec Grasp
        x_constructed = procedure_grasp(C, A, alpha)
        z_constructed = dot(C, x_constructed)
        
        x_improved = recherche_locale(x_constructed, C, A)
        z_improved = dot(C, x_improved)
        
        reactive_grasp.avg_values[alpha_index] = (reactive_grasp.avg_values[alpha_index] * reactive_grasp.counts[alpha_index] + z_improved) / 
                                            (reactive_grasp.counts[alpha_index] + 1)
        reactive_grasp.counts[alpha_index] += 1
        
        # Mise à jour des meilleures/pires valeurs
        if z_improved > z_best
            x_best = copy(x_improved)
            z_best = z_improved
        end
        
        if z_improved < z_worst
            z_worst = z_improved
        end
        
        # Mise à jour des probabilités
        if iter % update_frequency == 0
            update_probabilities(reactive_grasp, z_best, z_worst)
            println("\n--- Mise à jour Reactive GRASP (itération $(iter)) ---")
            println("Probabilités: $(round.(reactive_grasp.probabilities, digits=4))")
            
        end
        
        #creation d'un historique pour chaque alpha
        push!(solutions_history, z_constructed)
        push!(improved_history, z_improved)
        push!(best_history, z_best)
        push!(alpha_history, alpha)
    end
    
  
    println("\n========== RÉSULTATS FINAUX REACTIVE GRASP ===========")
    println("Meilleure solution: Z = $(z_best)")
    println("Variables sélectionnées: $(sum(x_best))")
    println("Distribution finale alphas values: $(round.(reactive_grasp.probabilities, digits=4))")
    
    
    return x_best, solutions_history, improved_history, best_history, alpha_history, reactive_grasp.probabilities
end

# ==================== INTERFACE UTILISATEUR ====================

function resoudreSPP_GRASP(fname; methode="grasp", alpha=0.7, n_iter=100, alphas=[0.2, 0.5, 0.75, 0.9, 1.0])
    """
    Fonction principale pour résoudre SPP avec GRASP
    """
    println("RÉSOLUTION GRASP: ", fname)
    println("Méthode: ", methode)
    println("="^70)
    
    C, A = loadSPP(fname)
    println("Dimensions: ", size(A, 1), " contraintes × ", size(A, 2), " variables\n")
    
    if methode == "grasp"
        return grasp_complet(C, A, alpha=alpha, n_iter=n_iter)
    elseif methode == "reactive-grasp"
        return reactive_grasp(C, A, alphas=alphas, n_iter=n_iter)
    else
        error("Méthode inconnue: ", methode)
    end
end

function comparer_methodes(fname; n_iter=100)
    """
    Compare GRASP standard vs Reactive GRASP
    """
    println("\n" * "="^70)
    println("COMPARAISON DES MÉTHODES: ", fname)
    println("="^70)
    
    C, A = loadSPP(fname)
    
    # GRASP standard
    println("\n>>> GRASP Standard (α=0.7)")
    @time x_grasp, const_grasp, impr_grasp, best_grasp = grasp_complet(C, A, α=0.7, n_iter=n_iter, verbose=false)
    
    # Reactive GRASP
    println("\n>>> Reactive GRASP")
    @time x_reactive, const_react, impr_react, best_react, α_hist, probs = reactive_grasp(C, A, n_iter=n_iter, verbose=false)
    
    # Résultats
    println("\n" * "="^70)
    println("RÉSULTATS DE COMPARAISON")
    println("GRASP Standard:    Z = $(dot(C, x_grasp))")
    println("Reactive GRASP:    Z = $(dot(C, x_reactive))")
    println("Distribution α finale: $(round.(probs, digits=4))")
    println("="^70)
    
    return (x_grasp, best_grasp), (x_reactive, best_react)
end

# ==================== EXÉCUTION ====================

if abspath(PROGRAM_FILE) == @__FILE__
   
    # GRASP standard
    @time resoudreSPP_GRASP("Data/pb_100rnd0100.dat", methode="grasp", n_iter=50)
    
    #REACTIVE Grasp
    #@time resoudreSPP_GRASP("Data/pb_200rnd0100.dat", methode="reactive-grasp", n_iter=100)
    
    # Comparaison
    # comparer_methodes("Data/pb_100rnd0100.dat", n_iter=100)
end