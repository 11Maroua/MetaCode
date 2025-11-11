using LinearAlgebra
include("loadSPP.jl")

# Structure pour maintenir l'état de couverture
mutable struct EtatCouverture
    nb_couverture::Vector{Int} #nb de variables qui couvrent une contrainte i 
    variables_actives::Set{Int}
    variables_inactives::Set{Int}
end

function creer_etat(x, A)
    m, n = size(A)
    nb_couverture = zeros(Int, m)
    
    for i in 1:m
        for j in 1:n
            if x[j] == 1 && A[i, j] == 1
                nb_couverture[i] += 1
            end
        end
    end
    
    variables_actives = Set(findall(x .== 1))
    variables_inactives = Set(findall(x .== 0))
    
    return EtatCouverture(nb_couverture, variables_actives, variables_inactives)
end

function est_faisable(x, A)
    m, n = size(A)
    for i in 1:m
        couvert = false
        for j in 1:n
            if x[j] == 1 && A[i, j] == 1
                couvert = true
                break
            end
        end
        if !couvert
            println("Contrainte ", i, " non couverte!")
            return false
        end
    end
    return true
end

function reparer_solution(x, A)
    println("Réparation de la solution...")
    x_reparé = copy(x)
    m, n = size(A)
    
    # Identifier les contraintes non couvertes
    contraintes_non_couvertes = []
    for i in 1:m
        couvert = false
        for j in 1:n
            if x_reparé[j] == 1 && A[i, j] == 1
                couvert = true
                break
            end
        end
        if !couvert
            push!(contraintes_non_couvertes, i)
        end
    end
    
    println("Nombre de contraintes non couvertes: ", length(contraintes_non_couvertes))
    
    # Pour chaque contrainte non couverte, ajouter la variable la moins chère qui la couvre
    for i in contraintes_non_couvertes
        # Trouver toutes les variables qui couvrent cette contrainte
        variables_couvrantes = findall(A[i, :] .== 1)
        if isempty(variables_couvrantes)
            error("Contrainte $i ne peut pas être couverte!")
        end
        
        # Choisir la variable la moins chère
        meilleure_var = variables_couvrantes[1]
        for j in variables_couvrantes
            if C[j] < C[meilleure_var]
                meilleure_var = j
            end
        end
        
        x_reparé[meilleure_var] = 1
        println("Ajout variable ", meilleure_var, " (coût ", C[meilleure_var], ") pour couvrir contrainte ", i)
    end
    
    # Vérifier que c'est maintenant faisable
    if est_faisable(x_reparé, A)
        println("✓ Solution réparée est faisable")
    else
        error("La réparation a échoué!")
    end
    
    return x_reparé
end

function peut_ajouter_retirer(etat, A, j_add, j_remove)
    m = size(A, 1)
    
    for i in 1:m
        nb_couv_actuel = etat.nb_couverture[i]
        nb_couv_futur = nb_couv_actuel
        
        if A[i, j_remove] == 1
            nb_couv_futur -= 1
        end
        if A[i, j_add] == 1
            nb_couv_futur += 1
        end
        
        # Vérifier que chaque contrainte reste couverte ET sans surcouverture
        if nb_couv_futur < 1 || nb_couv_futur > 1
            return false
        end
    end
    return true
end

function peut_ajouter_retirer_2(etat, A, j1_add, j2_add, j_remove)
    m = size(A, 1)
    
    for i in 1:m
        nb = etat.nb_couverture[i]
        
        if A[i, j_remove] == 1
            nb -= 1
        end
        if A[i, j1_add] == 1
            nb += 1
        end
        if A[i, j2_add] == 1
            nb += 1
        end
        
        if nb > 1 || nb < 1
            return false
        end
    end
    return true
end

function copier_etat(etat)
    return EtatCouverture(copy(etat.nb_couverture), copy(etat.variables_actives), copy(etat.variables_inactives))
end

function mettre_a_jour_etat!(etat, A, j_add, j_remove, j_add2, j_remove2)
    m = size(A, 1)
    
    if j_remove !== nothing
        delete!(etat.variables_actives, j_remove)
        push!(etat.variables_inactives, j_remove)
        for i in 1:m
            if A[i, j_remove] == 1
                etat.nb_couverture[i] -= 1
            end
        end
    end
    
    if j_remove2 !== nothing
        delete!(etat.variables_actives, j_remove2)
        push!(etat.variables_inactives, j_remove2)
        for i in 1:m
            if A[i, j_remove2] == 1
                etat.nb_couverture[i] -= 1
            end
        end
    end
    
    if j_add !== nothing
        push!(etat.variables_actives, j_add)
        delete!(etat.variables_inactives, j_add)
        for i in 1:m
            if A[i, j_add] == 1
                etat.nb_couverture[i] += 1
            end
        end
    end
    
    if j_add2 !== nothing
        push!(etat.variables_actives, j_add2)
        delete!(etat.variables_inactives, j_add2)
        for i in 1:m
            if A[i, j_add2] == 1
                etat.nb_couverture[i] += 1
            end
        end
    end
end

function chercher_amelioration_11(x, z_actuel, C, A, etat)
    inactives = collect(etat.variables_inactives)
    actives = collect(etat.variables_actives)
    
    # Trier par gain potentiel décroissant
    candidats = []
    for j_add in inactives
        for j_remove in actives
            gain = C[j_add] - C[j_remove]
            if gain >= 0 && peut_ajouter_retirer(etat, A, j_add, j_remove)
                push!(candidats, (gain, j_add, j_remove))
            end
        end
    end
    
    # Prendre le meilleur candidat
    if !isempty(candidats)
        sort!(candidats, by=x->x[1], rev=true)
        meilleur = candidats[1]
        gain, j_add, j_remove = meilleur
        
        x_voisin = copy(x)
        x_voisin[j_add] = 1
        x_voisin[j_remove] = 0
        
        nouvel_etat = copier_etat(etat)
        mettre_a_jour_etat!(nouvel_etat, A, j_add, j_remove, nothing, nothing)
        
        return x_voisin, z_actuel + gain, nouvel_etat, true
    end
    
    return x, z_actuel, etat, false
end

function chercher_amelioration_21(x, z_actuel, C, A, etat)
    inactives = collect(etat.variables_inactives)
    actives = collect(etat.variables_actives)
    
    # Trier les variables inactives par coût décroissant (les plus prometteuses en premier)
    sort!(inactives, by=j->C[j], rev=true)
    # Trier les variables actives par coût croissant (les moins chères en premier)
    sort!(actives, by=j->C[j])
    
    candidats = []
    
    for (idx1, j1_add) in enumerate(inactives[1:min(100, length(inactives))])
        for idx2 in (idx1+1):min(idx1+50, length(inactives))
            j2_add = inactives[idx2]
            for j_remove in actives[1:min(100, length(actives))]
                gain = C[j1_add] + C[j2_add] - C[j_remove]
                
                if gain >= 0 && peut_ajouter_retirer_2(etat, A, j1_add, j2_add, j_remove)
                    push!(candidats, (gain, j1_add, j2_add, j_remove))
                end
            end
        end
    end
    
    # Prendre le meilleur candidat
    if !isempty(candidats)
        sort!(candidats, by=x->x[1], rev=true)
        meilleur = candidats[1]
        gain, j1_add, j2_add, j_remove = meilleur
        
        x_voisin = copy(x)
        x_voisin[j1_add] = 1
        x_voisin[j2_add] = 1
        x_voisin[j_remove] = 0
        
        nouvel_etat = copier_etat(etat)
        mettre_a_jour_etat!(nouvel_etat, A, j1_add, j_remove, j2_add, nothing)
        
        return x_voisin, z_actuel + gain, nouvel_etat, true
    end
    
    return x, z_actuel, etat, false
end

function chercher_echange_neutre(x, z_actuel, C, A, etat)
    inactives = collect(etat.variables_inactives)
    actives = collect(etat.variables_actives)
    
    # Chercher un échange avec gain = 0 (pour diversifier)
    for j_add in inactives
        for j_remove in actives
            gain = C[j_add] - C[j_remove]
            
            if gain == 0 && peut_ajouter_retirer(etat, A, j_add, j_remove)
                x_voisin = copy(x)
                x_voisin[j_add] = 1
                x_voisin[j_remove] = 0
                
                nouvel_etat = copier_etat(etat)
                mettre_a_jour_etat!(nouvel_etat, A, j_add, j_remove, nothing, nothing)
                
                return x_voisin, z_actuel, nouvel_etat, true
            end
        end
    end
    
    return x, z_actuel, etat, false
end

function recherche_locale(x, C, A)
    x_courant = copy(x)
    z_courant = dot(C, x_courant)
    amelioration = true
    iterations = 0
    
    println("-----Début recherche locale améliorée...------")
    println("Solution initiale: z = ", z_courant)
    println()
    
    etat = creer_etat(x_courant, A)
    
    while amelioration && iterations < 50
        iterations += 1
        amelioration = false
        
        # ESSAYER D'ABORD LE VOISINAGE 2-1 (plus prometteur)
        x_ameliore, z_ameliore, etat_ameliore, trouve = chercher_amelioration_21(x_courant, z_courant, C, A, etat)
        if trouve
            x_courant = x_ameliore
            z_courant = z_ameliore
            etat = etat_ameliore
            amelioration = true
            println("Itération ", iterations, ": Amélioration 2-1 trouvée: z = ", z_courant)
            continue
        end
        
        # PUIS LE VOISINAGE 1-1
        x_ameliore, z_ameliore, etat_ameliore, trouve = chercher_amelioration_11(x_courant, z_courant, C, A, etat)
        if trouve
            x_courant = x_ameliore
            z_courant = z_ameliore
            etat = etat_ameliore
            amelioration = true
            println("Itération ", iterations, ": Amélioration 1-1 trouvée: z = ", z_courant)
            continue
        end
        
        # ENSUITE ESSAYER DES ÉCHANGES AVEC GAIN NUL (pour diversifier)
        x_ameliore, z_ameliore, etat_ameliore, trouve = chercher_echange_neutre(x_courant, z_courant, C, A, etat)
        if trouve
            x_courant = x_ameliore
            z_courant = z_ameliore
            etat = etat_ameliore
            amelioration = true
            println("Itération ", iterations, ": Échange neutre effectué: z = ", z_courant)
        end
    end
    
    println("=== RECHERCHE LOCALE TERMINÉE ===")
    println("Valeur finale: z = ", z_courant)
    println("Nombre d'itérations: ", iterations)
    println("Amélioration totale: ", z_courant - dot(C, x))
    
    return x_courant, z_courant
end

function tester_recherche_locale()
    println("--------------TEST ÉTAPE 2 - RECHERCHE LOCALE AMÉLIORÉE----------")
    
    fname = "Data/pb_500rnd0100.dat"
    global C, A = loadSPP(fname)  # Rend C global pour la fonction réparer_solution
    
    println("Instance: ", fname)
    println("Dimensions: ", size(A, 1), " contraintes × ", size(A, 2), " variables")
    println()
    
    x_init = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    z_init = dot(C, x_init)
    println("Solution initiale: z = ", z_init)
    
    # Vérifier et réparer la solution si nécessaire
    if !est_faisable(x_init, A)
        println("Solution initiale non faisable - réparation...")
        x_reparé = reparer_solution(x_init, A)
        z_reparé = dot(C, x_reparé)
        println("Solution réparée: z = ", z_reparé)
        println("Coût supplémentaire: ", z_reparé - z_init)
        x_init = x_reparé
        z_init = z_reparé
    else
        println("✓ Solution initiale est faisable")
    end
    println()
    
    x_ameliore, z_ameliore = recherche_locale(x_init, C, A)
    
    println()
    println("Recherche locale terminée avec succès!")
    println("Amélioration totale: ", z_ameliore - z_init)
end

@time begin
    tester_recherche_locale()
end