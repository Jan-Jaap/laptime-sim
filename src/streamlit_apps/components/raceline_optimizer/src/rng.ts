// Seedable PRNG for the browser-side optimizer (mulberry32).
// The client does not need to reproduce numpy's PCG64 sequences — parity
// fixtures pass explicit sampling inputs instead — it only needs to be
// deterministic for a given seed (used by the worker and tests).

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
