import type { IntermediatesOperation } from '@features/intermediates/core/types';
import type { QueryClient } from '@tanstack/react-query';

import { isOperationSettled } from '@features/intermediates/core/types';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import { listIntermediatesOperations } from './api';
import { intermediatesKeys } from './keys';
import { attachIntermediatesRealtime } from './realtime';

/**
 * The operation the manager is following. Kept outside the component so closing Settings while a cleanup runs and
 * reopening it later shows the same operation and its result. The server, not this tab, remembers running work: a
 * manager that opens with nothing followed asks it, so a reload picks a running cleanup back up.
 */
export const activeOperationStore = createExternalStore<{ operationId: string | null }>({ operationId: null });

registerAccountOwnedResource({
  clear: () => activeOperationStore.setSnapshot({ operationId: null }),
  name: 'intermediates-active-operation',
});

export const followIntermediatesOperation = (operationId: string | null): void => {
  activeOperationStore.setSnapshot({ operationId });
};

/** Follows an operation the server just returned, seeding its query so the panel needs no extra lookup. */
export const adoptIntermediatesOperation = (
  queryClient: QueryClient,
  owner: AccountScope,
  operation: IntermediatesOperation
): void => {
  queryClient.setQueryData(intermediatesKeys.operation(owner, operation.operationId), operation);
  followIntermediatesOperation(operation.operationId);
};

/** A fresh listing after a timed-out start must answer quickly or not at all; the dialog stays usable either way. */
const FRESH_LISTING_TIMEOUT_MS = 10_000;

// One listing per account at a time, so a remount (or StrictMode's replayed mount) shares a request; a sign-out and
// back in gets a fresh one.
const listings = new Map<string, Promise<IntermediatesOperation[]>>();

const listOnce = (owner: AccountScope): Promise<IntermediatesOperation[]> => {
  const key = `${owner.accountId ?? 'local'}\u0000${owner.epoch}`;
  let listing = listings.get(key);
  if (!listing) {
    listing = listIntermediatesOperations(owner.signal).finally(() => listings.delete(key));
    listings.set(key, listing);
  }
  return listing;
};

/**
 * Asks the server for the account's operations and follows the newest unsettled one, unless the followed operation
 * or the account changed meanwhile: a Confirm that lands while the listing is in flight is never overwritten by an
 * older run. Returns what it adopted, or null when the server listed nothing running; rejects when it could not ask.
 */
export const reconcileIntermediatesOperations = async (
  queryClient: QueryClient,
  owner: AccountScope = captureAccountScope(),
  options: {
    /**
     * Ask again rather than join a listing already in flight: a listing begun before a Confirm cannot contain the
     * run it started, and one that hangs must not hold the confirmation open. Bounded by its own timeout. The
     * server's `created_at` of the confirmed preview: only a run created after it can be the one that Confirm
     * started, since the server also lists earlier finished runs.
     */
    startedAfter?: string;
  } = {}
): Promise<IntermediatesOperation | null> => {
  const followedAtStart = activeOperationStore.getSnapshot().operationId;
  const { startedAfter } = options;
  const operations =
    startedAfter !== undefined
      ? await listIntermediatesOperations(
          AbortSignal.any([owner.signal, AbortSignal.timeout(FRESH_LISTING_TIMEOUT_MS)])
        )
      : await listOnce(owner);
  // After a lost start the run counts even when it already finished; a catch-up on open only picks up work still
  // running, so finished runs from before are not re-shown. Both timestamps are the server's, so they compare.
  const candidate =
    startedAfter !== undefined
      ? (operations.find(
          (operation) =>
            operation.operationId !== followedAtStart && Date.parse(operation.createdAt) >= Date.parse(startedAfter)
        ) ?? null)
      : (operations.find((operation) => !isOperationSettled(operation)) ?? null);
  if (
    candidate === null ||
    !isAccountScopeCurrent(owner) ||
    activeOperationStore.getSnapshot().operationId !== followedAtStart
  ) {
    return null;
  }
  adoptIntermediatesOperation(queryClient, owner, candidate);
  return candidate;
};

/** Registers an open manager: realtime updates for its lifetime, and a catch-up when nothing is followed yet. */
export const attachIntermediatesManager = (queryClient: QueryClient): (() => void) => {
  const detachRealtime = attachIntermediatesRealtime(queryClient);
  if (activeOperationStore.getSnapshot().operationId === null) {
    // A failed catch-up is not an error the section shows; the next open or a Confirm asks again.
    void reconcileIntermediatesOperations(queryClient).catch(() => undefined);
  }
  return detachRealtime;
};
