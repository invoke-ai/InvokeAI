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
 * result union instead.
 */
export const mapWithConcurrency = async <T, R>(
  items: readonly T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<R>
): Promise<R[]> => {
  const results: R[] = [];
  let nextIndex = 0;

  const worker = async (): Promise<void> => {
    while (nextIndex < items.length) {
      const index = nextIndex;

      nextIndex += 1;
      results[index] = await mapper(items[index]!, index);
    }
  };

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));

  return results;
};
