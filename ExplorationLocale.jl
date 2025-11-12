using LinearAlgebra

include("loadSPP.jl")
include("Glouton.jl")

# ==================== CONSTRUCTION VOISINAGES ====================

function generer_voisinage_2_1(x, C, A)
    n = length(x)
    vars_in = findall(x .== 1)
    vars_out = findall(x .== 0)
    
    z_current = dot(C, x)
    meilleur_voisin = copy(x)
    meilleur_z = z_current
    amelioration = false
    compteur = 0
    
    for idx1 in 1:length(vars_in)-1
        for idx2 in idx1+1:length(vars_in)
            drop1 = vars_in[idx1]
            drop2 = vars_in[idx2]
            
            x_temp = copy(x)
            x_temp[drop1] = 0
            x_temp[drop2] = 0
            
            for add in vars_out
                compteur += 1
                
                if peut_ajouter(x_temp, A, add)
                    x_voisin = copy(x_temp)
                    x_voisin[add] = 1
                    z_voisin = dot(C, x_voisin)
                    
                    if z_voisin > meilleur_z
                        meilleur_z = z_voisin
                        meilleur_voisin = copy(x_voisin)
                        amelioration = true
                    end
                end
            end
        end
    end
    
    gain = meilleur_z - z_current
    return meilleur_voisin, gain, amelioration, compteur
end

function generer_voisinage_1_1(x, C, A)
    n = length(x)
    vars_in = findall(x .== 1)
    vars_out = findall(x .== 0)
    
    z_current = dot(C, x)
    meilleur_voisin = copy(x)
    meilleur_z = z_current
    amelioration = false
    compteur = 0
    
    for drop in vars_in
        x_temp = copy(x)
        x_temp[drop] = 0
        
        for add in vars_out
            compteur += 1
            
            if peut_ajouter(x_temp, A, add)
                x_voisin = copy(x_temp)
                x_voisin[add] = 1
                z_voisin = dot(C, x_voisin)
                
                if z_voisin > meilleur_z
                    meilleur_z = z_voisin
                    meilleur_voisin = copy(x_voisin)
                    amelioration = true
                end
            end
        end
    end
    
    gain = meilleur_z - z_current
    return meilleur_voisin, gain, amelioration, compteur
end

function generer_voisinage_3_1(x, C, A)
    n = length(x)
    vars_in = findall(x .== 1)
    vars_out = findall(x .== 0)
    
    if length(vars_in) < 3
        return copy(x), 0.0, false, 0
    end
    
    z_current = dot(C, x)
    meilleur_voisin = copy(x)
    meilleur_z = z_current
    amelioration = false
    compteur = 0
    
    for idx1 in 1:length(vars_in)-2
        for idx2 in idx1+1:length(vars_in)-1
            for idx3 in idx2+1:length(vars_in)
                drop1 = vars_in[idx1]
                drop2 = vars_in[idx2]
                drop3 = vars_in[idx3]
                
                x_temp = copy(x)
                x_temp[drop1] = 0
                x_temp[drop2] = 0
                x_temp[drop3] = 0
                
                for add in vars_out
                    compteur += 1
                    
                    if peut_ajouter(x_temp, A, add)
                        x_voisin = copy(x_temp)
                        x_voisin[add] = 1
                        z_voisin = dot(C, x_voisin)
                        
                        if z_voisin > meilleur_z
                            meilleur_z = z_voisin
                            meilleur_voisin = copy(x_voisin)
                            amelioration = true
                        end
                    end
                end
            end
        end
    end
    
    gain = meilleur_z - z_current
    return meilleur_voisin, gain, amelioration, compteur
end

# ==================== PERTURBATION SIMPLE ====================

function perturber_solution(x, C, A)
    x_perturbe = copy(x)
    vars_in = findall(x .== 1)
    
    if length(vars_in) >= 3
        to_remove = vars_in[1:3]
        for var in to_remove
            x_perturbe[var] = 0
        end
    end
    
    # Réparer la solution de manière gloutonne
    vars_out = findall(x_perturbe .== 0)
    for j in vars_out
        if peut_ajouter(x_perturbe, A, j)
            x_perturbe[j] = 1
        end
    end
    
    return x_perturbe
end

# ==================== DESCENTE AVEC MULTI-START ====================

function descente_multi_start(x_initial, C, A; max_restarts=5)
    
    println("\n" * "="^70)
    println("=== DESCENTE MULTI-START (", max_restarts, " redémarrages) ===")
    println("="^70)
    
    x_best = copy(x_initial)
    z_best = dot(C, x_initial)
    
    println("Solution initiale: Z = ", z_best, " (", sum(x_best), " variables)")
    
    for restart in 1:max_restarts
        println("\n---- RESTART ", restart, "/", max_restarts, " ----")
        
        # Partir de la meilleure solution actuelle
        x = copy(x_best)
        
        # Si ce n'est pas le premier restart, perturber
        if restart > 1
            println("Perturbation de la solution...")
            x = perturber_solution(x, C, A)
            z_perturbe = dot(C, x)
            println("Après perturbation: Z = ", z_perturbe, " (", sum(x), " variables)")
        end
        
        # Descente profonde
        iteration = 0
        ameliorations_locales = 0
        
        while true
            iteration += 1
            
            # Essayer d'abord le voisinage 2-1
            x_nouveau, gain, amelioration, nb_voisins = generer_voisinage_2_1(x, C, A)
            
            # Si bloqué, essayer le voisinage 1-1
            if !amelioration || gain <= 0
                x_nouveau, gain, amelioration, nb_voisins = generer_voisinage_1_1(x, C, A)
            end
            
            # Si toujours bloqué, essayer 3-1
            if !amelioration || gain <= 0
                x_nouveau, gain, amelioration, nb_voisins = generer_voisinage_3_1(x, C, A)
            end
            
            if !amelioration || gain <= 0
                println("  Itération ", iteration, ": Optimum local atteint")
                break
            end
            
            x = x_nouveau
            z_new = dot(C, x)
            ameliorations_locales += 1
            println("  Itération ", iteration, ": Z = ", z_new, " (+", round(gain, digits=2), 
                    ") - ", sum(x), " vars")
        end
        
        # Vérifier si on a trouvé une meilleure solution globale
        z_current = dot(C, x)
        if z_current > z_best
            improvement = z_current - z_best
            x_best = copy(x)
            z_best = z_current
            println("\n✓ NOUVELLE MEILLEURE SOLUTION: Z = ", z_best, " (+", 
                    round(improvement, digits=2), ")")
        else
            println("\nPas d'amélioration sur ce restart")
        end
    end
    
    println("\n" * "="^70)
    println("=== SOLUTION FINALE ===")
    println("Z initiale: ", dot(C, x_initial))
    println("Z finale:   ", z_best, " (+", round(z_best - dot(C, x_initial), digits=2), ")")
    println("Variables:  ", sum(x_best))
    println("Redémarrages: ", max_restarts)
    println("="^70)
    
    return x_best
end

# ==================== INTERFACE ====================

function resoudreSPP(fname; methode="multi-start")
   
    println("RÉSOLUTION: ", fname)
    println("Méthode: ", methode)
    println("="^70)
    
    C, A = loadSPP(fname)
    println("Dimensions: ", size(A, 1), " contraintes × ", size(A, 2), " variables\n")
    
    # Construction gloutonne
    x_init = construction_gloutonne(C, A)
    
    # Amélioration selon la méthode choisie
    if methode == "multi-start"
        x_final = descente_multi_start(x_init, C, A, max_restarts=5)
    else
        error("Méthode inconnue: ", methode)
    end
    
    return x_final
end

function tester_recherche_locale()
    println("\n" * "="^70)
    println("TEST RECHERCHE LOCALE SPP")
    println("="^70)
    
    fname = "Data/pb_100rnd0100.dat"
    C, A = loadSPP(fname)
    # Construction gloutonne
    x_init = construction_gloutonne(C, A)

    z_init = dot(C, x_init)
    println("\nSolution initiale: Z = ", z_init, " (", sum(x_init), " variables)")
    
    # Test avec multi-start
    println("\n>>> Test: MULTI-START")
    x_final = descente_multi_start(x_init, C, A, max_restarts=5)
    
    println("\n✓ Tests terminés!")
end

# ==================== EXÉCUTION ====================

if abspath(PROGRAM_FILE) == @__FILE__
    # Lancer le test
    @time tester_recherche_locale()
    
    # Ou tester différentes méthodes:
    # resoudreSPP("Data/pb_500rnd0100.dat", methode="multi-start")
end