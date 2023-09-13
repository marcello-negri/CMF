library("BayesianGLasso")

BlockMCMC <- function (X, n_iter, burnin){
    BayesGLasso <- blockGLasso(X=X, iterations=n_iter, burnIn=burnin, lambdaPriora=1, lambdaPriorb=.1)
    return (BayesGLasso)
}

inverseGaussian <- function(mu, lambda) {
  v <- rnorm(1)  ## sample from a normal distribution with a mean of 0 and 1 standard deviation
  y <- v * v
  x <- mu + (mu * mu * y)/(2 * lambda) - (mu/(2 * lambda)) * sqrt(4 * mu * lambda * y + mu * mu * y * y)
  test <- runif(1)  ## sample from a uniform distribution between 0 and 1
  if (test <= (mu)/(mu + x)) {
    s <- x
  } else {
    s <- (mu * mu)/x
  }
  s
}

Bayes.glasso.MB <- function(S, n, lambda, NSCAN = 1000, p.1, p.2) {
  ## the Bayesian graphical lasso
  p <- nrow(S)
  T.inv <- matrix(0, p, p)
  W <- solve(S + diag(lambda, p))
  MB.arr <- matrix(0, NSCAN, p.1 * p.2)
  W.avg <- W
  for (nscan in 1:NSCAN) {
    for (i in 1:p) {
      ## partition
      seq.1 <- NULL
      if (i - 1 >= 1) {
        seq.1 <- 1:(i - 1)
      }
      seq.2 <- NULL
      if (i + 1 <= p) {
        seq.2 <- (i + 1):p
      }
      s <- c(seq.1, seq.2, i)
      S.part <- S[s, s]
      W.part <- W[s, s]
      T.inv.part <- T.inv[s, s]
      ## print(nscan)
      W.xx <- W.part[1:(p - 1), 1:(p - 1)]
      W.xy <- W.part[1:(p - 1), p]
      ## W.yy <- W.part[p,p,collapse=FALSE]
      S.xx <- S.part[1:(p - 1), 1:(p - 1)]
      S.xy <- matrix(S.part[1:(p - 1), p], ncol = 1)
      S.yy <- as.double(S.part[p, p])
      ## T.inv.xx <- T.inv.part[1:(p-1),1:(p-1)]
      T.inv.xy <- T.inv.part[1:(p - 1), p]
      ## T.inv.yy <- T.inv.part[p,p,collapse=FALSE]
      ## update W.xy
      C.inv <- solve(W.xx) * (S.yy + lambda) + diag(T.inv.xy)
      R.C <- chol(C.inv)  ## t(R.C) %*% R.C = L * L^t = C.inv
      z <- forwardsolve(t(R.C), S.xy)
      m <- backsolve(R.C, z)
      mu <- -t(m)
      r.n <- matrix(rnorm(p - 1), ncol = 1)
      z <- backsolve(R.C, r.n)
      W.xy <- matrix(t(z) + mu, ncol = 1)
      ## W.yx <- t(W.xy)
      ## update W.yy gam <- rgamma(1,shape=n/2+1,scale=(S.yy+lambda)/2) gam <- rWishart(1, n, solve(matrix(S.yy +lambda)))[,,1]
      gam <- (S.yy + lambda)^(-1) * rchisq(1, n)
      W.yy <- gam + as.double(t(W.xy) %*% solve(W.xx, W.xy))
      ## update W
      W[i, i] <- W.yy
      if (i - 1 >= 1) {
        W[i, 1:(i - 1)] <- W.xy[1:(i - 1), 1]
        W[1:(i - 1), i] <- W[i, 1:(i - 1)]
      }
      if (p >= i + 1) {
        W[i, (i + 1):p] <- W.xy[i:(p - 1), 1]
        W[(i + 1):p, i] <- W[i, (i + 1):p]
      }
    }
    ## update T.inv
    for (l in 1:(p - 1)) {
      for (j in (l + 1):p) {
        T.inv[l, j] <- inverseGaussian(lambda/abs(W[l, j]), lambda^2)
        T.inv[j, l] <- T.inv[l, j]
      }
    }
    W.avg <- W.avg + W
    ## MB.arr[nscan, ] <- as.vector(W[(1:p.1), (p.1 + 1):(p.1 + p.2)])
    MB.arr[nscan, ] <- as.vector(W)
  }
  list(MB.arr = MB.arr, W.avg = W.avg/NSCAN)
}