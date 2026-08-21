using LinearAlgebra

include("loadSPP.jl")
include("Glouton.jl")


function conflit(A, j, k)
   
    for i in 1:size(A, 1)
        if A[i, j] == 1 && A[i, k] == 1
            return true
        end
    end
    return false
end

# ==================== DESCENTE PROFONDE ====================

function descente_profonde(x_initial, C, A; max_iter=1000, verbose=true)
   
    if verbose
        println("\n=== DESCENTE PROFONDE ===")
        println("Solution initiale: Z = $(dot(C, x_initial)) ($(sum(x_initial)) variables)")
    end
    
    x_current = copy(x_initial)
    z_current = dot(C, x_current)
    iteration = 0
    ameliorations = 0
    
    while iteration < max_iter
        iteration += 1
        amelioration_trouvee = false
        meilleur_gain = 0.0
        meilleur_voisin = nothing
        
        vars_in = findall(x_current .== 1)
        vars_out = findall(x_current .== 0)
        
        vars_out_sorted = vars_out[sortperm(C[vars_out], rev=true)]
        
        # ===== Stratégie 1: Échange 1-1 (drop + add) =====
        for drop in vars_in
            x_temp = copy(x_current)
            x_temp[drop] = 0
            
            # Explorer seulement les meilleures variables candidates
            for add in vars_out_sorted[1:min(50, length(vars_out_sorted))]
                if peut_ajouter(x_temp, A, add)
                    gain = C[add] - C[drop]
                    
                    if gain > meilleur_gain
                        meilleur_gain = gain
                        meilleur_voisin = copy(x_temp)
                        meilleur_voisin[add] = 1
                        amelioration_trouvee = true
                    end
                end
            end
        end
        
        # ===== Stratégie 2: Échange 2-1 (si pas d'amélioration en 1-1) =====
        if !amelioration_trouvee && length(vars_in) >= 2
            x_voisin, gain = chercher_voisin_2_1(x_current, C, A, vars_in, vars_out_sorted)
            if gain > meilleur_gain
                meilleur_gain = gain
                meilleur_voisin = x_voisin
                amelioration_trouvee = true
            end
        end
        
        # ===== Appliquer la meilleure amélioration ou arrêter =====
        if !amelioration_trouvee
            if verbose
                println("Optimum local atteint après $iteration itérations")
            end
            break
        end
        
        x_current = meilleur_voisin
        z_current += meilleur_gain
        ameliorations += 1
        
        if verbose && iteration % 10 == 0
            println("Itération $iteration: Z = $z_current (+$(round(meilleur_gain, digits=2)))")
        end
    end
    
    if verbose
        println("\n=== RÉSULTAT DESCENTE PROFONDE ===")
        println("Itérations: $iteration")
        println("Améliorations: $ameliorations")
        println("Solution finale: Z = $z_current ($(sum(x_current)) variables)")
    end
    
    return x_current
end

function chercher_voisin_2_1(x, C, A, vars_in, vars_out_sorted)
    
    meilleur_voisin = copy(x)
    meilleur_gain = 0.0
    
    # Explorer les paires de variables à retirer
    for i in 1:length(vars_in)-1
        for j in i+1:min(i+20, length(vars_in))  # Limiter les paires testées
            drop1 = vars_in[i]
            drop2 = vars_in[j]
            
            x_temp = copy(x)
            x_temp[drop1] = 0
            x_temp[drop2] = 0
            
            # Tester les meilleures variables candidates
            for add in vars_out_sorted[1:min(30, length(vars_out_sorted))]
                if peut_ajouter(x_temp, A, add)
                    gain = C[add] - C[drop1] - C[drop2]
                    
                    if gain > meilleur_gain
                        meilleur_gain = gain
                        meilleur_voisin = copy(x_temp)
                        meilleur_voisin[add] = 1
                    end
                end
            end
        end
    end
    
    return meilleur_voisin, meilleur_gain
end

# ==================== DESCENTE SIMPLE ====================

function descente_simple(x_initial, C, A; verbose=true)
    
    if verbose
        println("\n=== DESCENTE RAPIDE ===")
        println("Solution initiale: Z = $(dot(C, x_initial)) ($(sum(x_initial)) variables)")
    end
    
    x_current = copy(x_initial)
    z_current = dot(C, x_current)
    iteration = 0
    ameliorations = 0
    
    while true
        iteration += 1
        amelioration_trouvee = false
        
        vars_in = findall(x_current .== 1)
        vars_out = findall(x_current .== 0)
        vars_out_sorted = vars_out[sortperm(C[vars_out], rev=true)]
        
        # Chercher la première amélioration
        for drop in vars_in
            x_temp = copy(x_current)
            x_temp[drop] = 0
            
            for add in vars_out_sorted[1:min(30, length(vars_out_sorted))]
                if peut_ajouter(x_temp, A, add)
                    gain = C[add] - C[drop]
                    
                    if gain > 0
                        x_current = copy(x_temp)
                        x_current[add] = 1
                        z_current += gain
                        ameliorations += 1
                        amelioration_trouvee = true
                        
                        if verbose && ameliorations % 10 == 0
                            println("Amélioration $ameliorations: Z = $z_current")
                        end
                        break
                    end
                end
            end
            
            if amelioration_trouvee
                break
            end
        end
        
        if !amelioration_trouvee
            if verbose
                println("Optimum local atteint après $iteration itérations")
            end
            break
        end
    end
    
    if verbose
        println("\n=== RÉSULTAT DESCENTE RAPIDE ===")
        println("Améliorations: $ameliorations")
        println("Solution finale: Z = $z_current ($(sum(x_current)) variables)")
    end
    
    return x_current
end
