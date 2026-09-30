import rfdc from 'rfdc';
const _rfdc = rfdc();

/**
 * Deep-clones an object using Really Fast Deep Clone.
 * This is the fastest deep clone library on Chrome, but not the fastest on FF. Still, it's much faster than lodash
 * and structuredClone, so it's the best all-around choice.
 *
 * Simple Benchmark: https://www.measurethat.net/Benchmarks/Show/30358/0/lodash-clonedeep-vs-jsonparsejsonstringify-vs-recursive
 * Repo: https://github.com/davidmarkclements/rfdc
 *
 * @param obj The object to deep-clone
 * @returns The cloned object
 */
export const deepClone = <T>(obj: T): T => _rfdc(obj);
