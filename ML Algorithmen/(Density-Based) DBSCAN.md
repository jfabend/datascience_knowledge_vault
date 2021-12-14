"draws" a radius epsilon around all data points

no neighbor in this radius => outlier
neighbors => no outlier

![[Pasted image 20210830201455.png|300]]

In this diagram, `minPts = 4`. Point A and the other red points are  `core points `, because the area surrounding these points in an ε radius contain `at least 4 points` (including the point itself). Because they are all reachable from one another, they form a single cluster. Points B and C are not core points, but are reachable from A (via other core points) and thus belong to the cluster as well. Point N is a noise point that is neither a core point nor directly-reachable.