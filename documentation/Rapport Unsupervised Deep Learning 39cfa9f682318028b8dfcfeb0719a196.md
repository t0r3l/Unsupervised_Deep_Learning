# Rapport Unsupervised Deep Learning

# Algo nature

| Algo | Nature de espace latent | CodeBook | CodeByte | Niveau de compression |
| --- | --- | --- | --- | --- |
| K-means (k=300) | **Entier discret** : 1 indice de cluster ∈ {0…k−1} | les **k centroïdes** (k × 784 floats) | ⌈log₂ 300⌉ = **9 bits** (~2 octets) | 6272 / 9 = **697:1** |
| Kohenen (k=300) | **Entier discret** : indice/coordonnées du BMU sur la grille | les **k feature vectors** (k × 784 floats) | ⌈log₂ 300⌉ = **9 bits** — *identique* | *identique* ≈ **697:1** — *identique* |

# Kmeas

## Mnist

Test 1 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%201.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%202.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%203.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%204.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%205.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%206.png)

Test 2 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%207.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%208.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%209.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2010.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2011.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2012.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2013.png)

Test 3 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2014.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2015.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2016.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2017.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2018.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2019.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2020.png)

Test 3 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2021.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2022.png)

### Conclusion

Sur le dataset mnist nos tests nous permettes de conclure que pour 10 classes, plus on augmente notre nombre de centroid plus les images générer ainsi que les compressions / décompressions sont plus claire et moins floue. On peut déduire qu’avoir plusieurs centroid permet moins de diversité de classe par cluster, comme ce qu’on observe dans nos bar chart de l’espace latent, et ça permet d’avoir de meilleurs représentant en fonction des différentes manières d’écrire des chiffres dans le dataset mnist. Par conséquent, ça améliore les résultats de compression / décompression

## Quickdraw 3 classe

Test 1 :  Premier test sur quickdraw avec k3 sur tout notre dataset de train  

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2023.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2024.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2025.png)

Test 2 : on re test mais avoir une autre seed 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2026.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2027.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2028.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2029.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2030.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2031.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2032.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2033.png)

Test 3: Maintenant avec k10 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2034.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2035.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2036.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2037.png)

Test 4 : maintenant avec k50 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2038.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2039.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2040.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2041.png)

Test 5 : Maintenant k = 100

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2042.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2043.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2044.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2045.png)

Test 6 : k  = 200

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2046.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2047.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2048.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2049.png)

Test 7 : k = 500

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2050.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2051.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2052.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2053.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2054.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2055.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2056.png)

Test 8 : k = 1000 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2057.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2058.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2059.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2060.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2061.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2062.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2063.png)

### Conclusion

On peut conclure sur kmeans avec l’algorithme de Lloyd que à partir de k = 100 le flou est assez bas pour reconnaitre l’image et à partir de K = 300 on réduit assez le floue pour être presque net. Si on met un k = 1000 ou k = 500 les images sont de bonne qualité avec une répartition par cluster de presque 1 classe par cluster avec peu de diversité dans les clusters.  

Si on met les k = 3 on se retrouve avec des images floue très peu reconnaissable, en fonction de l’initialisation toutes mes classes ne sont pas représenter parmi les 3 centroids qui sont généré.  

Même constat que pour mnist même avec moins de classe, augmenter le nombre de centroid permet moins de diversité dans les classes et donc d’avoir de meilleurs représentant pour chaque classe. 

Néanmoins on voit que sur notre dataset, certaine classe sont mieux représenté que d’autre. Les chats sont difficilement représenter, les dessins de chats diffères beaucoup en fonction de qui les as dessiné, on a quand même un forme global qui se dessine sur les centroids. Les pommes sont très bien représenté, on reconnait bien leurs forme ronde avec une tige. Les voitures c’est très variables, la forme qui est conservé est celle de la voiture longue avec les deux roues vue de profils. 

Sur les taches de compression / decompression, kmeans reste imparfait et floue du à l’utilisation de la distance euclidienne. 

## Quickdraw 10 classe

Test 1 avec k = 10 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2064.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2065.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2066.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2067.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2068.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2069.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2070.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2071.png)

Test : k = 100  

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2072.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2073.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2074.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2075.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2076.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2077.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2078.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2079.png)

Test 3 : k = 500

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2080.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2081.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2082.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2083.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2084.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2085.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2086.png)

Test 4 : k = 1000

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2087.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2088.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2089.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2090.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2091.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2092.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2093.png)

Test 5 : k = 5000

Les graphiques des barres plot plantes

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2094.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2095.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2096.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2097.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2098.png)

Les courbes de l’espace latent ne marche pas. 

### Conclusion

Pour approfondir nos tests, on a ajouter plusieurs classe disponible dans le dataset quickdraw, pour se retrouver avec 10 classes (cat, apple, car, fish, house, tree, clock, star, umbrella, airplane). 

Dans cette versions on a testé k = 10, k = 100, k = 1000 et k = 5000, on peut en conclure qu’avec un k = au nombre de classe, on obtiens pas de bon résultat, il y a trop de diversité par classe, avec 10 classes en fonction de l’initialisation on peut se retrouver avec des classes non représenté.

 Ensuite, pour k = 100, on observe une amélioration mais toujours très floue, on commence à reconnaitre les formes globales de chacune des classes du dataset, on remarque que l’ajout de classe rend la compression / décompression moins claire avec un k = 100 que pour 3 classes. On a encore trop de diversité de classe par cluster mais on commence à voir l’apparition d’un représentant plus fort d’une seule classe dans chaque cluster. 

Pour k = 1000, on a beaucoup moins de diversité par cluster. On a une nette amélioration sur les taches de compression / décompression sur mon dataset. Faire un x10 sur K à permis une amélioration sur la représentation de chaque classe de notre dataset. 

Pour k = 5000, les images sont beaucoup mieux représentés, on voit que chaque dessins à une images décompresser qui ressemble plus à ça forme et pas seulement une forme global. ça reste floue mais on a une bien meilleures représentation. Par contre, je ne peux que déduire qu’on a moins de classe de cluster, l’affichage des bar charts, ne fonctionne pas pour k = 5000. 

# Kohonen

## Mnist

Test 1 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%2099.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20100.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20101.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20102.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20103.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20104.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20105.png)

Test 2 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20106.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20107.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20108.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20109.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20110.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20111.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20112.png)

### Conclusion

Pour Kohonen, les résultats sont similaires à ceux de K-means, à la différence de la disposition des centroids, comme on peut visualiser dans la carte de kohonen, on voit ques les images proches se retrouve côte à côte et ça produit comme un “dégradé” d’images. 

On remarque aussi, que augmenter la taille de la grille produit des résultats similaire à augmenter le K sur K-means, plus la grille est grande moins on retrouve de diversité dans chaque cluster. 

Pour une grille 10x10, les images restes floue mais reconnaissable, on voit bien que le rapprochement entre les centroids proches c’est bien fais mais on voit que certaine classe comme le 4 et le 7 sont mal représenté. 

Pour une grille de 20x20 on commence à avoir de meilleurs résultats, moins de diversité dans chaque cluster, une meilleurs représentation globale de chaque classe. On voit que les classes mal représenté en 10x10 sont ici bien représenté. Les 4 et 7 on est représenter reconnaissable. On garde un aspect globale floue. 

## Quickdraw 3 classe

Test 1 :  Gamma = 1, K = 10x10 et 8 époques

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20113.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20114.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20115.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20116.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20117.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20118.png)

Test 2 : Gamma = 0,5, K = 10x10 et 8 époques

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20119.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20120.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20121.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20122.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20123.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20124.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20125.png)

Test 3 :  Gamma = 0,5, K = 10x10 et 100 époques

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20126.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20127.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20128.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20129.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20130.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20131.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20132.png)

Test 3 :  Gamma = 0,5, K = 10x10 et 100 époques et sur tout les exemples du dataset 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20133.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20134.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20135.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20136.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20137.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20138.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20139.png)

Test 4 :  Gamma = 1, K = 20x20 et 8 époques et sur tout les exemples du dataset 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20140.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20141.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20142.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20143.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20144.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20145.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20146.png)

Test 5 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20147.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20148.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20149.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20150.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20151.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20152.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20153.png)

Test 6 : 

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20154.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20155.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20156.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20157.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20158.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20159.png)

![image.png](Rapport%20Unsupervised%20Deep%20Learning/image%20160.png)

### Conclusion

Pour le quickdraw dataset, avec kohonen même constat que sur mnist, plus on augmente notre taille de grille plus les résultats de compression / décompression et génération sont moins floue et plus reconnaissable. Le nombre d’époque et le gamma influe sur la qualité des rapprochement. Un gamma assez petit avec beaucoup d’époque permet de bien rapprocher entre eux. Plus gamma et petit, moins il affecte c’est voisin quand on update la position des centroids. Plus un gamma est grand plus le nombre de voisin affecter est grand. Donc avec un grand nombre d’époque et une petit gamma,on laisse le temps à l’algo de correctement faire les rapprochement.

Si on test avec une petite grille de 1x1 ou 3x3 on se retrouve avec des resultats peut reconnaissable avec un vague début de forme. Comme pour kmeans, avec peut de centroid on a trop de diversité par cluster pour avoir de bonnes représentation. En 3x3, on voit par contre que kohonen fonctionne bien car il commence déjà à faire le rapprochement entre les éléments qui se ressemble. 

Avec une grille de 10x10, on commence par avoir de meilleurs résultat, les éléments se regroupe bien dans la carte de kohonen et les images sont plus reconnaissable avec un de réprésentant pour les différents cas du dataset. On a toujours des images floues et des formes pas claire. On voit bien dans les histogrammes, moins de diversité par classe que pour les petites grilles. 

Pour une grille de 20x20, on commence à bien reconnaitre les résultats, pour obtenir les résultats on a passé le gamma à 0.5 on remarque qu’un gamma plus petit permet d’avoir de meilleurs résultat. Une grille de 20x20 permet d’avoir de meilleurs résultats.