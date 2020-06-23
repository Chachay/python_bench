const now = require('performance-now')

const genSymMatrix = (N) => {
  const M = Array(N).fill().map(_ => Array(N).fill().map(_ => Math.random()))
  const result = Array(N).fill().map(_ => Array(N).fill())
  return result.map((row, i) => {
    return row.map((val, j) => {
      return M[i].reduce((sum, elm, k) => sum + (elm * M[j][k]), 0)
    })
  })
}

// Cholesky decomposition - Rosetta Code https://rosettacode.org/wiki/Cholesky_decomposition
const cholesky = (array) => {
  const zeros = [...Array(array.length)].map(_ => Array(array.length).fill(0))
  const L = zeros.map((row, r, xL) => row.map((v, c) => {
    const sum = row.reduce((s, _, i) => i < c ? s + xL[r][i] * xL[c][i] : s, 0)
    return xL[r][c] = c < r + 1 ? r === c ? Math.sqrt(array[r][r] - sum) : (array[r][c] - sum) / xL[c][c] : v
  }))
  return L
}

const TimeIt = (iterations, testFunction) => {
  const results = []
  let total = 0
  for (let i = 0; i < iterations; i++) {
    const start = now()
    testFunction()
    const end = now()
    const duration = end - start
    results.push(duration)
    total += duration
  }
  return {
    results: results,
    loops: iterations,
    mean: total / results.length
  }
}

const A = genSymMatrix(256)
const bench = () => {
  cholesky(A)
}
const result = TimeIt(10, bench)
console.log(`${result.mean.toFixed(2)} msec`)
