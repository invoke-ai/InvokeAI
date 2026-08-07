/**
 * Run an async mapper over a list with a bounded number of tasks in flight.
 *
 * Every caller here is fanning out HTTP requests — importing gallery images
 * onto the canvas, bundling a project's assets into an archive, restoring them
 * on the way back. `Promise.all` over the whole list is the obvious spelling
 * and the wrong one: a project with three hundred images opens three hundred
 * sockets, the browser queues most of them anyway, and the backend sees a
 * thundering herd it did nothing to deserve.
 *
 * Results keep the input order regardless of completion order, so callers can
 * zip them back against the source list by index.
 *
 * A rejecting mapper rejects the whole call, exactly as `Promise.all` would;
 * callers that want partial success catch inside the mapper and return a
 * result union instead. The first rejection also *stops* the remaining work:
 * `Promise.all` only stops waiting, so without this a failed export would go on
 * to issue one request per asset it had not reached yet — hundreds of them, for
 * an operation that has already failed. Results for items never started are
 * left as holes, which callers see as `undefined` and never reach, because the
 * rejection is what they get instead.
 */
export const mapWithConcurrency = async <T, R>(
  items: readonly T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<R>,
  options: { signal?: AbortSignal } = {}
): Promise<R[]> => {
  const results: R[] = [];
  let nextIndex = 0;
  let stopped = false;

  const worker = async (): Promise<void> => {
    while (nextIndex < items.length) {
      if (stopped || options.signal?.aborted) {
        return;
      }

      const index = nextIndex;

      nextIndex += 1;
      try {
        results[index] = await mapper(items[index]!, index);
      } catch (error) {
        stopped = true;
        throw error;
      }
    }
  };

  // At least one worker: a concurrency of zero would otherwise spawn none, resolve immediately and
  // hand back an empty array as though the list had been processed.
  const workerCount = Math.max(1, Math.min(concurrency, items.length));

  await Promise.all(Array.from({ length: items.length === 0 ? 0 : workerCount }, worker));

  return results;
};
