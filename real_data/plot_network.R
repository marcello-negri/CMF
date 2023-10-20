rm(list = ls())

library(igraph)
library(org.Hs.eg.db)

### dataset from Sheffer et al.: "Association of survival and disease progression with chromosomal instability: a genomic exploration of colorectal cancer."
get.data <- function(pathway = NULL, use.log2 = TRUE, class = NULL, included.in.analysis = TRUE) {
  load('./data/data.soft.collapsed.info.RData')
  load('./data/map.msigdb.kegg.entrez.RData')

  if (is.null(pathway)) {
    cat('get.data(): pathway cannot be NULL\n')
  } else {
    pathways = sapply(map, function(x) {x[1]})
    pathway.index = which(pathways == pathway)

    genes.on.pathway = map[[pathway.index]][3:length(map[[pathway.index]])]
    w.pathway = intersect(genes.on.pathway, colnames(data))

    if (is.null(class)) {
      if (included.in.analysis) {
        w.patient = which(data[, 'included_in_analysis'] == 'Yes')
      } else {
        w.patient = 1:nrow(data)
      }
    } else {
      if (included.in.analysis) {
        w.patient = which(data[, 'class'] == class & data[, 'included_in_analysis'] == 'Yes')
      } else {
        w.patient = which(data[, 'class'] == class)
      }
    }

    if (use.log2) {
      gene.expr = log2(data[w.patient, w.pathway])
    } else {
      gene.expr = data[w.patient, w.pathway]
    }
    info = data[w.patient, 1:20]

    return(list(gene.expr = gene.expr, info = info, pathway = pathway, use.log2 = use.log2, class = class, included.in.analysis = included.in.analysis))
  }
}

produce.graph = function(MB) {

  p = nrow(MB)
  q = ncol(MB)

  W = matrix(0, nrow = q, ncol = q)

  w = which(abs(MB) > 1e-5, arr.ind = TRUE)
  W[w] = MB[w]
  W[lower.tri(W)] = W[upper.tri(W)]

  g <- graph.adjacency(abs(W), weighted = TRUE, mode = 'upper', diag = FALSE)
  signMat <- sign(W)
  e.names <- get.edgelist(g)
  edge.signs <- rep(0, nrow(e.names))
  edge.weights <- rep(0, nrow(e.names))

  if (nrow(e.names) > 0) {
    for (j in 1:nrow(e.names)) {
      edge.signs[j] <- signMat[e.names[j, 1], e.names[j, 2]]
      edge.weights[j] <- abs(W[e.names[j, 1], e.names[j, 2]])
    }
    edge.weights <- edge.weights / max(edge.weights) * 30
  }

  V(g)$label = Y.label# append(rownames(MB), Y.label)

  V(g)$color = c(rep('red', p), rep('white', q-p))
  V(g)$size = 5
  V(g)$label.family = 'sans'
  V(g)$label.cex = 0.7
  E(g)$width = edge.weights

  edge.colors = rep(NA, length(edge.signs))
  edge.colors[which(edge.signs == 1)] = 'black'
  edge.colors[which(edge.signs == -1)] = 'grey'
  edge.colors[which(edge.signs == 0)] = 'green' ### green color to identify zero-width edges, should not appear if everything is correct
  E(g)$color = edge.colors

  return(g)
}

threshold.matrix = function(MB, thresh.low, p, q) {
  qua <- c(thresh.low, 0.5, 1 - thresh.low)
  CI <- apply(MB, 2, quantile, qua)
  CI.sign <- apply(CI, 2, function(x) {
    x[1] * x[3]
  })
  MB.vec <- rep(0, p*q)
  w <- which(CI.sign > 0)
  MB.vec[w] <- CI[2, w]
  MB.mat <- matrix(MB.vec, nrow = p)

  return(MB.mat)
}

pathway = 'KEGG_PATHWAYS_IN_CANCER'

p = 6
q = 312
NSCAN = 5000
burnin = floor(0.3 * NSCAN)

lambda = 0.4
thresh.low = 0.2

data = get.data(pathway = pathway, use.log2 = TRUE, included.in.analysis = TRUE, class = 'Primary Tumor')
Y.label = colnames(data$gene.expr) ### gene names in entrez notation

X.label = c('AGE', 'SEX', 'T', 'N', 'M', 'GS')#, 'GS') ### names of clinical variables

### look up the entrez names and find their symbol notation
x <- org.Hs.egSYMBOL
mapped_genes <- mappedkeys(x)
xx <- as.list(x[mapped_genes])
a = xx[Y.label]
a[sapply(a, is.null)] <- NA
Y.label.symbol = unlist(a)
Y.label.symbol[which(is.na(Y.label.symbol))] = 'LOC652346'
Y.label.symbol = as.vector(Y.label.symbol) ### gene names in symbol notation
# Y.label = Y.label.symbol
# print(Y.label.symbol)
# print(thresh.low)
cat(sprintf('thresh.low = %.2f, lambda = %.2f\n', thresh.low, lambda))

# load("./cond_flow_data_optimal_0.050_1.000.RData")
load("./cond_flow_data_optlamb_16.248_optalph_0.167_1.000_p0.50.RData")
CMB.array = as.matrix(sapply(CMB.array, as.numeric))
CMB.mat = threshold.matrix(CMB.array, thresh.low, p, p+q)
CMB.mat[lower.tri(CMB.mat)] = 0
rownames(CMB.mat) = X.label ### names of clinical variables
Y.label.symbol <- append(X.label,Y.label.symbol)
colnames(CMB.mat) = Y.label.symbol ### names of genes in symbol notation
Y.label = Y.label.symbol
g = produce.graph(CMB.mat)

lay = layout.fruchterman.reingold(g, niter = 1000)#, area = vcount(g)^1.55, repulserad = vcount(g)^1.9)

### to produce the subgraph, find all nodes with at least 1 edge
e = get.edges(g, 1:length(E(g)))
v = sort(unique(c(1:3, e[, 1], e[, 2]))) ### in any case, include the 7 clinical variables (1:7)

g.sub = induced.subgraph(g, v)
lay.sub = layout.fruchterman.reingold(g.sub, niter = 1000)#, area = vcount(g)^1.55, repulserad = vcount(g)^1.9)
# lay.sub = layout.circle(g.sub)

V(g)$label = ''
V(g)$size = 3
V(g)$label.family = 'sans'
V(g)$label.cex = 0.7

## node sizes for 7 clinical variables and remaining 312 variables, here: 11 for both
V(g.sub)$size = c(rep(11, 7), rep(11, length(V(g.sub)) - 7))

### adjust the label font size based on its number of characters
label.cex = rep(NA)
n.char = nchar(V(g.sub)$label)
label.cex[which(n.char == 1)] = 2.7
label.cex[which(n.char == 2)] = 1.9
label.cex[which(n.char == 3)] = 1.7
label.cex[which(n.char == 4)] = 1.3
label.cex[which(n.char == 5)] = 1.1
label.cex[which(n.char == 6)] = 1.0
label.cex[which(n.char == 7)] = 0.9
label.cex[which(n.char == 8)] = 0.8
label.cex[which(n.char == 9)] = 0.7
label.cex[which(n.char == 10)] = 0.6
V(g.sub)$label.cex = label.cex

### same label font color for all nodes
V(g.sub)$label.color = c(rep('black', 7), rep('black', length(V(g.sub)) - 7))

w = 15
pdf('./diagram_1.pdf', width = w, height = w)
par(oma = c(0, 0, 0, 0), mai = c(0, 0, 0, 0))
plot(g, layout = lay, main = '')
dev.off()

pdf('./diagram_2.pdf', width = w, height = w)
par(oma = c(0, 0, 0, 0), mai = c(0, 0, 0, 0))
plot(g.sub, layout = lay.sub, main = '')
dev.off()

print("FINISHED")
