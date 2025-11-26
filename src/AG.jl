using LinearAlgebra
using Random
include("loadSPP.jl")
include("Glouton.jl")

function est_realisable(x::Vector{Int}, A::Matrix{Int})
    m, n = size(A)
    for i in 1:m
        count = 0
        for j in 1:n
            if x[j] == 1 && A[i, j] == 1
                count += 1
                if count > 1
                    return false
                end
            end
        end
    end
    return true
end

function evaluer_fitness(x::Vector{Int}, C::Vector{Int}, A::Matrix{Int})
    if est_realisable(x, A)
        return dot(C, x)
    else
        return -sum(C) #penaliser les solutions non-realisables
    end
end

function reparer_solution(x::Vector{Int}, C::Vector{Int}, A::Matrix{Int})
    x_repare = copy(x)
    m, n = size(A)
    
    for i in 1:m
        variables_conflit = Int[]
        for j in 1:n
            if x_repare[j] == 1 && A[i, j] == 1
                push!(variables_conflit, j)
            end
        end
        
        #on repare si necessaire seulement
        if length(variables_conflit) > 1
            # Garder la meilleure variable
            meilleur_j = variables_conflit[argmax([C[j] for j in variables_conflit])]
            for j in variables_conflit
                if j != meilleur_j
                    x_repare[j] = 0
                end
            end
        end
    end
    return x_repare
end

# Recherche locale comme mecanisme de mutation
function recherche_locale(x::Vector{Int}, C::Vector{Int}, A::Matrix{Int})
    meilleure_sol = copy(x)
    meilleure_fitness = evaluer_fitness(x, C, A)
    n = length(x)
    amelioration = true
    
    while amelioration
        amelioration = false
        
        #essaie differentes techniques pour ameliorer solution
        # Reduction de variable: flip bit 1->0
        for i in 1:n
            if meilleure_sol[i] == 1
                x_test = copy(meilleure_sol)
                x_test[i] = 0
                fitness_test = evaluer_fitness(x_test, C, A)
                if fitness_test > meilleure_fitness
                    meilleure_sol = x_test
                    meilleure_fitness = fitness_test
                    amelioration = true
                end
            end
        end
        
        # Ajout de variable: flip bit 0->1
        for i in 1:n
            if meilleure_sol[i] == 0
                x_test = copy(meilleure_sol)
                x_test[i] = 1
                x_test = reparer_solution(x_test, C, A)
                fitness_test = evaluer_fitness(x_test, C, A)
                if fitness_test > meilleure_fitness
                    meilleure_sol = x_test
                    meilleure_fitness = fitness_test
                    amelioration = true
                end
            end
        end
        
        # Swap variables
        for i in 1:n
            if meilleure_sol[i] == 1
                for j in 1:n
                    if meilleure_sol[j] == 0
                        x_test = copy(meilleure_sol)
                        x_test[i] = 0
                        x_test[j] = 1
                        x_test = reparer_solution(x_test, C, A)
                        fitness_test = evaluer_fitness(x_test, C, A)
                        if fitness_test > meilleure_fitness
                            meilleure_sol = x_test
                            meilleure_fitness = fitness_test
                            amelioration = true
                            break
                        end
                    end
                end
                if amelioration
                    break
                end
            end
        end
    end
    
    return meilleure_sol, meilleure_fitness
end

function generer_solution_aleatoire(n::Int)
    return [rand() < 0.25 ? 1 : 0 for _ in 1:n]
end


#selection parents avec methode tournoi à parametre k=3
function selection_tournoi(population::Vector{Vector{Int}}, fitness::Vector{Float64})
    i1, i2, i3 = rand(1:length(population), 3)
    if fitness[i1] >= fitness[i2] && fitness[i1] >= fitness[i3]
        return population[i1]
    elseif fitness[i2] >= fitness[i3]
        return population[i2]
    else
        return population[i3]
    end
end

# Utilisation méthode Croisement à 2 points (meilleur que 1 point)
function croisement_2_points(p1::Vector{Int}, p2::Vector{Int})
    n = length(p1)
    point1 = rand(1:n-2)
    point2 = rand(point1+1:n-1)
    
    e1 = vcat(p1[1:point1], p2[point1+1:point2], p1[point2+1:end])
    e2 = vcat(p2[1:point1], p1[point1+1:point2], p2[point2+1:end])
    
    return e1, e2
end

function mutation(x::Vector{Int}, prob_mutation::Float64)
    x_mute = copy(x)
    n = length(x)
    for i in 1:n
        if rand() < prob_mutation
            x_mute[i] = 1 - x_mute[i]
        end
    end
    return x_mute
end

function algorithme_genetique_simple(C::Vector{Int}, A::Matrix{Int}; 
                                     taille_pop=100,
                                     generations=300,
                                     prob_croisement=0.85,
                                     prob_mutation=0.02,
                                     freq_recherche_locale=50,
                                     verbose=false)
   
    n = length(C)
    
    println("================= ALGORITHME GÉNÉTIQUE SIMPLE ===============")
    println("Variables: $n, Contraintes: $(size(A,1))")
    println("Population: $taille_pop, Générations: $generations")
    println()
    
    # Initialisation population
    population = Vector{Vector{Int}}()
    fitness_pop = Vector{Float64}()
    
    # Ajouter solution gloutonne
    sol_glouton = construction_gloutonne(C, A)
    push!(population, sol_glouton)
    push!(fitness_pop, evaluer_fitness(sol_glouton, C, A))
    
    # Générer reste de la population aléatoirement
    for _ in 2:taille_pop
        sol = generer_solution_aleatoire(n)
        sol_repare = reparer_solution(sol, C, A)
        push!(population, sol_repare)
        push!(fitness_pop, evaluer_fitness(sol_repare, C, A))
    end
    
    meilleure_sol = copy(population[argmax(fitness_pop)])
    meilleure_fitness = maximum(fitness_pop)
    
    println("Fitness initiale: $meilleure_fitness")


    #critere d'arret: nb de générations
    for gen in 1:generations
        nouvelle_population = Vector{Vector{Int}}()
        nouvelle_fitness = Vector{Float64}()
        
        # Élitisme: on prend top3 inidividus
        indices_tries = sortperm(fitness_pop, rev=true)
        for i in 1:3
            push!(nouvelle_population, copy(population[indices_tries[i]]))
            push!(nouvelle_fitness, fitness_pop[indices_tries[i]])
        end
        
        # Génération nouveaux individus
        while length(nouvelle_population) < taille_pop
            p1 = selection_tournoi(population, fitness_pop)
            p2 = selection_tournoi(population, fitness_pop)
            
            if rand() < prob_croisement
                e1, e2 = croisement_2_points(p1, p2)
            else
                e1, e2 = copy(p1), copy(p2)
            end
            
            e1 = mutation(e1, prob_mutation)
            e2 = mutation(e2, prob_mutation)
            
            for enfant in [e1, e2]
                if length(nouvelle_population) < taille_pop
                    enfant_repare = reparer_solution(enfant, C, A)
                    fit = evaluer_fitness(enfant_repare, C, A)
                    push!(nouvelle_population, enfant_repare)
                    push!(nouvelle_fitness, fit)
                end
            end
        end
        
        population = nouvelle_population
        fitness_pop = nouvelle_fitness
        
        #Recherche locale sur meilleure solution pour utiliser travail precedent et
        #fournir de bons elements
        if gen % freq_recherche_locale == 0
            best_idx = argmax(fitness_pop)
            sol_amelioree, fit_amelioree = recherche_locale(population[best_idx], C, A)
            if fit_amelioree > fitness_pop[best_idx]
                population[best_idx] = sol_amelioree
                fitness_pop[best_idx] = fit_amelioree
            end
        end
        
        # Mise à jour meilleure solution
        best_idx = argmax(fitness_pop)
        if fitness_pop[best_idx] > meilleure_fitness
            ancienne = meilleure_fitness
            meilleure_sol = copy(population[best_idx])
            meilleure_fitness = fitness_pop[best_idx]
            println("Generation $gen: AMÉLIORATION = $meilleure_fitness (+$(meilleure_fitness - ancienne))")
        end
        
      
        if gen % 50 == 0
            println("Gen $gen: Meilleure fitness = $meilleure_fitness")
        end
    end
    
    
    println("\n============= RÉSULTAT FINAL =============")
    println("Fitness: $meilleure_fitness")
    println("Variables sélectionnées: $(sum(meilleure_sol))")
    println("Solution réalisable: $(est_realisable(meilleure_sol, A))")
    
    return meilleure_sol, meilleure_fitness
end

function tester_AG()
    C, A = loadSPP("dat/pb_200rnd0100.dat")
    
    println("Instance: $(length(C)) variables, $(size(A,1)) contraintes\n")
    
    solution, fitness = algorithme_genetique_simple(C, A, 
                                                    taille_pop=80,
                                                    generations=250,
                                                    prob_croisement=0.9,
                                                    prob_mutation=0.02,
                                                    freq_recherche_locale=50)
 
    println("\n=========== COMPARAISON GLOUTON =========")
    
    sol_glouton = construction_gloutonne(C, A)
    fitness_glouton = evaluer_fitness(sol_glouton, C, A)
    
    println("Glouton: $fitness_glouton")
    println("AG:      $fitness")
    
    if fitness > fitness_glouton
        amelioration = fitness - fitness_glouton
        println("\n AMÉLIORATION: +$amelioration )")
    else
        println("\n Même performance")
    end
    
    return solution, fitness
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time solution, fitness = tester_AG()
end