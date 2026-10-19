/**
 * Bound concurrent mapping and preserve input order. First rejection stops scheduling new work but cannot cancel
 * running tasks; catch inside the mapper for partial results.
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

  // Always start at least one worker; zero would falsely resolve without processing input.
  const workerCount = Math.max(1, Math.min(concurrency, items.length));

  await Promise.all(Array.from({ length: items.length === 0 ? 0 : workerCount }, worker));

  return results;
};
