# cppi-markov-pricer
Implémentation Python de la méthode de matrices de transition pour le pricing de CPPI et la quantification du gap risk, d'après Paulot &amp; Lacroze (2009, 2010).

Limites du modele:
    -Ca reste un modele discret,donc la precision est moindre(cela peut etre vu dans le test martingale de la matrice)
    -les limites(l'infini) ne sont pas bien modélises 