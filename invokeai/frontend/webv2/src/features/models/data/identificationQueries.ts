import type { ModelIdentification, ModelIdentificationOutcome } from '@features/models/core/types';

import { chunkKeys } from '@features/models/core/identification';
import { assertAccountScopeCurrent, captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
// `streamedQuery` is still exported as experimental: a TanStack Query minor update may rename or change it, so the
// unknown-models scan below is the first place to check when that dependency is bumped.
import { experimental_streamedQuery as streamedQuery, queryOptions } from '@tanstack/react-query';

import { bulkGetModelIdentification, getModelIdentification } from './api';

const accountKey = (owner: AccountScope) => ({ accountId: owner.accountId, epoch: owner.epoch });

/** Keys carry the account so a sign-out never serves one account's preview to the next. */
export const identificationKeys = {
  forAccount: (owner: AccountScope) => ['models', 'identification', accountKey(owner)] as const,
  /**
   * `fingerprint` is the identity the preview was taken against. Re-identifying or editing the record changes it, so
   * a preview of the old record is never shown for the new one, and the old entry just ages out.
   */
  model: (owner: AccountScope, modelKey: string, fingerprint: string) =>
    [...identificationKeys.forAccount(owner), 'model', modelKey, fingerprint] as const,
  unknownScan: (owner: AccountScope, modelKeys: readonly string[]) =>
    [...identificationKeys.forAccount(owner), 'unknown-scan', modelKeys] as const,
};

const fenced = <T>(owner: AccountScope, request: (signal: AbortSignal) => Promise<T>, signal: AbortSignal) =>
  request(AbortSignal.any([signal, owner.signal])).then((result) => {
    assertAccountScopeCurrent(owner);
    return result;
  });

/** The preview of one record. It reads the file, so it is not refetched on focus or reconnect. */
export const modelIdentificationQueryOptions = (modelKey: string, fingerprint: string, owner = captureAccountScope()) =>
  queryOptions<ModelIdentification>({
    gcTime: 60_000,
    queryFn: ({ signal }) => fenced(owner, (fencedSignal) => getModelIdentification(modelKey, fencedSignal), signal),
    queryKey: identificationKeys.model(owner, modelKey, fingerprint),
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    retry: false,
    staleTime: Infinity,
  });

/**
 * Previews the keys batch by batch, yielding each batch's outcomes as it arrives. A batch already requested finishes;
 * `stop` only prevents further ones. Aborting `signal` (the query is dropped) abandons the request in flight.
 */
export async function* streamModelIdentifications(
  keys: readonly string[],
  signal: AbortSignal,
  stop?: AbortSignal
): AsyncGenerator<ModelIdentificationOutcome> {
  for (const batch of chunkKeys(keys)) {
    if (stop?.aborted) {
      return;
    }

    yield* await bulkGetModelIdentification(batch, signal);
  }
}

/**
 * What one opening of the scan has already done. A retry continues from it instead of reading the checked files
 * again: `checked` holds the keys whose outcome the query data already has, and a retry after Stop replaces `stop`.
 */
export interface UnknownScanProgress {
  checked: Set<string>;
  stop: AbortController;
}

export const createUnknownScanProgress = (): UnknownScanProgress => ({
  checked: new Set(),
  stop: new AbortController(),
});

/** Lifts a Stop, so the next run of the scan goes on with the unchecked keys. */
export const resumeUnknownScan = (progress: UnknownScanProgress) => {
  progress.stop = new AbortController();
};

/**
 * The scan behind "Check unknown models": dropped as soon as the dialog closes. Its data grows with each batch, so
 * the dialog lists results while the rest are still being checked; a refetch appends the keys not yet checked.
 */
export const unknownModelScanQueryOptions = (
  modelKeys: readonly string[],
  progress: UnknownScanProgress,
  owner = captureAccountScope()
) =>
  queryOptions({
    gcTime: 0,
    queryFn: streamedQuery<ModelIdentificationOutcome>({
      refetchMode: 'append',
      streamFn: ({ signal }) => {
        const fencedSignal = AbortSignal.any([signal, owner.signal]);

        return (async function* () {
          const pending = modelKeys.filter((key) => !progress.checked.has(key));

          for await (const outcome of streamModelIdentifications(pending, fencedSignal, progress.stop.signal)) {
            assertAccountScopeCurrent(owner);
            yield outcome;
            progress.checked.add(outcome.key);
          }
        })();
      },
    }),
    queryKey: identificationKeys.unknownScan(owner, modelKeys),
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    retry: false,
    staleTime: 0,
  });
