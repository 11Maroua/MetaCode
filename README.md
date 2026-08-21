# MetaCode — Métaheuristiques pour le Set Packing Problem

![Julia](https://img.shields.io/badge/Julia-1.x-9558B2?logo=julia&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-projet%20académique-informational)

Implémentation et comparaison de plusieurs métaheuristiques appliquées au **Set Packing Problem (SPP)**.

> Heuristique gloutonne → recherche locale → GRASP / Reactive GRASP → Algorithme Génétique → Colonies de fourmis (ACO), comparés sur 10 instances allant de 9 à 2000 variables, avec la solution exacte (GLPK) comme référence quand elle est atteignable.

---

## Sommaire

- [Le problème](#le-problème)
- [Méthodes implémentées](#méthodes-implémentées)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Auteurs](#auteurs)

---

## Le problème

Le **Set Packing Problem** est un problème d'optimisation combinatoire NP-difficile : sélectionner un sous-ensemble d'objets de valeur maximale, sous contrainte qu'aucun couple d'objets sélectionnés ne soit en conflit.

**Formulation :**

$$\max \sum_{j=1}^{n} c_j x_j \quad \text{s.c.} \quad \sum_{j=1}^{n} a_{ij} x_j \leq 1, \ \forall i \in \{1,\dots,m\}, \quad x \in \{0,1\}^n$$

Les instances sont au format OR-Library (`dat/`), de la petite instance didactique (`m=7, n=9`) jusqu'à `pb_2000rnd0100.dat` (`m=10000, n=2000`).

## Méthodes implémentées

| # | Méthode | Fichier | Idée clé |
|---|---|---|---|
| EI1 | **Heuristique gloutonne** | `src/Glouton.jl` | Construction itérative par ratio d'utilité $u_j = c_j / \sum_i a_{ij}$ |
| EI1 | **Recherche locale** | `src/Exploration.jl` | Voisinages 1-1 et 2-1 (échange retrait/ajout), descente simple (first-improvement) et profonde (best-improvement) |
| EI2 | **GRASP** | `src/Grasp.jl` | Construction gloutonne-randomisée via *Restricted Candidate List* (RCL), multi-start + recherche locale |
| EI2 | **Reactive GRASP** | `src/Grasp.jl` | Apprentissage en ligne du paramètre α de la RCL, par probabilités mises à jour selon la qualité moyenne obtenue |
| EI3 | **Algorithme Génétique** | `src/AG.jl` | Population + sélection par tournoi, croisement 2 points, mutation, réparation des solutions infaisables, hybridation mémétique (recherche locale périodique) |
| EI3 | **ACO (colonies de fourmis)** | `src/ACO.jl` | Construction guidée par phéromones + heuristique locale, alternance exploration/exploitation, perturbation anti-stagnation |
| — | **Référence exacte** | `src/setSPP.jl`, `src/main.jl` | Modèle `JuMP` résolu par `GLPK` (solveur MILP exact, utilisé comme borne de comparaison sur les instances de taille raisonnable) |

Toutes les métaheuristiques réutilisent le même noyau de faisabilité (`peut_ajouter`, dans `Glouton.jl`) pour vérifier qu'une variable peut être ajoutée sans créer de conflit.

### RCL du GRASP

$$\text{RCL} = \{ j : u_j \geq u_{\min} + \alpha (u_{\max} - u_{\min}) \}$$

### Mise à jour du Reactive GRASP

$$p_k = \frac{q_k}{\sum_i q_i}, \qquad q_k = \frac{\bar z_k - z_{\text{worst}}}{z_{\text{best}} - z_{\text{worst}}}$$

Chaque valeur d'α reçoit une probabilité de sélection proportionnelle à la qualité moyenne des solutions qu'elle a produites.

## Structure du dépôt

```
MetaCode/
├── src/
│   ├── loadSPP.jl        # lecture des instances 
│   ├── setSPP.jl         # modèle JuMP exact du SPP
│   ├── getfname.jl       # utilitaire de listing de fichiers
│   ├── main.jl           # démo minimale : chargement + résolution exacte
│   ├── Glouton.jl        # EI1 : construction gloutonne + peut_ajouter
│   ├── Exploration.jl    # EI1 : recherche locale (1-1, 2-1)
│   ├── Grasp.jl          # EI2 : GRASP + Reactive GRASP
│   ├── AG.jl             # EI3 : algorithme génétique
│   ├── ACO.jl            # EI3 : colonies de fourmis
│   └── experiments.jl    # scripts d'expérimentation complets (EI1 + EI2), génère res/
├── dat/                  # 10 instances SPP 
├── res/                  # tableaux LaTeX et graphiques générés par experiments.jl
├── doc/
│   └── rapport.tex       # rapport complet du projet
├── Project.toml          # dépendances Julia
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<ton-pseudo>/MetaCode.git
cd MetaCode
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Utilisation

```julia
include("src/experiments.jl")

# EI1 : glouton + recherche locale sur les 10 instances (+ comparaison GLPK)
experimentationSPP()

# EI2 : GRASP vs Reactive GRASP (5 runs par instance)
experimentationSPP_EI2(n_runs=5)
```

Pour lancer un algorithme isolément :

```julia
include("src/loadSPP.jl")
include("src/Glouton.jl")
include("src/Exploration.jl")
include("src/Grasp.jl")

C, A = loadSPP("dat/pb_500rnd0100.dat")

x0 = construction_gloutonne(C, A)
x1 = descente_profonde(x0, C, A)

x_best, hist_construct, hist_improved, hist_best = grasp_complet(C, A, alpha=0.7, n_iter=100)
```

```julia
include("src/ACO.jl")
conf = precompute_conflicts(A)
best_sol, best_val = ACO_SPP(C, A, conf; maxIter=200, maxAnt=15)
```

## Résultats

### EI1 — Glouton + recherche locale (extrait)

| Instance | m | n | Z* (GLPK) | Z glouton | Gap | Z + recherche locale | Gap | Speedup vs GLPK |
|---|---|---|---|---|---|---|---|---|
| `pb_100rnd0100.dat` | 500 | 100 | 372 | 342 | 8.06% | 348 | 6.45% | **×56.7** |
| `pb_500rnd0300.dat` | 2500 | 500 | — | 674 | — | 674 | — | — |
| `pb_2000rnd0100.dat` | 10000 | 2000 | — (intraitable) | 37 | — | 37 | — | — |

Sur la seule instance résolue à l'optimum exact, la recherche locale ramène le gap de 8% à 6.5% pour un temps de calcul environ **56 fois inférieur** à GLPK. Sur les grandes instances, GLPK devient intraitable — seules les heuristiques passent à l'échelle.

### EI2 — GRASP vs Reactive GRASP (5 runs, extrait)

| Instance | GRASP (Z moyen) | Reactive GRASP (Z moyen) |
|---|---|---|
| `pb_100rnd0100.dat` | 367.6 | 367.2 |
| `pb_200rnd0100.dat` | 399.8 | 394.6 |
| `pb_500rnd0100.dat` | 294.8 | 293.4 |
| `pb_1000rnd0100.dat` | 60.6 | **57.0** |

Les tableaux et graphiques complets (`res/tableau_EI1.tex`, `res/tableau_EI2.tex`, `res/graphiques_EI1.pdf`, `res/graphiques_EI2.pdf`) sont générés automatiquement par `src/experiments.jl` et intégrés dans `doc/rapport.tex`.

## Auteurs

Projet réalisé par **NAIT SLIMANI Maroua**, **MOLLI Lila** et **ABID Ikram** dans le cadre du cours de Métaheuristiques.

`loadSPP.jl`, `setSPP.jl`, `getfname.jl` et la structure initiale de `main.jl` sont fournis par l'enseignant (Xavier Gandibleux) comme socle de départ du projet voir `LICENSE`.

---

📄 Rapport complet : [`doc/rapport.tex`](doc/rapport.tex)
