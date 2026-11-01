import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createListenerChannel } from '@platform/state/externalStoreCore';

/**
 * The closures behind `{ kind: 'undo' }` notification actions. Notifications are plain workbench state, so they name
 * an undo by id and the closure stays here, for this tab and account only. The record owns the entry and a toast
 * only borrows it: an entry lives while a listed notification names it (`retainNotificationUndos`) or a toast holds
 * it, until it runs, is withdrawn, or the account changes. Entries neither referenced nor held (say, recorded while the
 * editor shell was not mounted) are capped, oldest dropped first. With the list capped at 100, a tab keeps at most 100
 * listed and 20 idle closures, plus what open toasts hold (Clear's toast can hold up to 100 cleared notices'). Most
 * capture ids or small records; Canvas's restore undo keeps a document (metadata only, pixels stay engine-owned), one
 * per project through `supersedes`.
 */
const MAX_IDLE_UNDOS = 20;

interface PendingUndo {
  /** Toasts showing this Undo now. */
  holds: number;
  /** Whether a listed notification names it, as of the last `retainNotificationUndos`. */
  isReferenced: boolean;
  onUndo: () => void;
  supersedes: string | undefined;
  withdrawal: AbortController;
}

const pendingUndos = new Map<string, PendingUndo>();
// Removals only: a new entry is always recorded or shown by its own caller, so nothing re-reads on registration.
const removals = createListenerChannel();
let nextUndoId = 0;

const isIdle = (entry: PendingUndo): boolean => entry.holds === 0 && !entry.isReferenced;

const dropOldestIdleUndo = (): void => {
  const idleIds = [...pendingUndos].flatMap(([undoId, entry]) => (isIdle(entry) ? [undoId] : []));

  if (idleIds.length > MAX_IDLE_UNDOS) {
    pendingUndos.delete(idleIds[0]!);
    removals.notify();
  }
};

/**
 * `supersedes` names what the undo reverts, such as one project's Generate settings. A newer undo with the same key
 * withdraws the older ones (their toasts close): undoing an older step after a newer one would leave a mix of both.
 */
export const registerNotificationUndo = (onUndo: () => void, supersedes?: string): string => {
  const undoId = `undo-${++nextUndoId}`;

  if (supersedes !== undefined) {
    withdrawNotificationUndos(supersedes);
  }

  pendingUndos.set(undoId, {
    holds: 0,
    isReferenced: false,
    onUndo,
    supersedes,
    withdrawal: new AbortController(),
  });
  dropOldestIdleUndo();

  return undoId;
};

const withdraw = (undoId: string, entry: PendingUndo): void => {
  pendingUndos.delete(undoId);
  entry.withdrawal.abort();
};

/**
 * Withdraws every undo registered under `supersedes` (their toasts close, their buttons go), for an owner that knows
 * the change they revert is no longer the latest: undone another way, or built on since.
 */
export const withdrawNotificationUndos = (supersedes: string): void => {
  let changed = false;

  for (const [undoId, entry] of pendingUndos) {
    if (entry.supersedes === supersedes) {
      withdraw(undoId, entry);
      changed = true;
    }
  }

  if (changed) {
    removals.notify();
  }
};

/** Withdraws one undo for good, closing a toast that offers it: its target is gone (its project closed). */
export const withdrawNotificationUndo = (undoId: string): void => {
  const entry = pendingUndos.get(undoId);

  if (entry) {
    withdraw(undoId, entry);
    removals.notify();
  }
};

export const hasNotificationUndo = (undoId: string): boolean => pendingUndos.has(undoId);

/** Calls `listener` whenever entries go, so a surface listing Undo buttons can drop the ones that went. */
export const subscribeNotificationUndos = removals.subscribe;

/** Aborts when the undo runs elsewhere or is withdrawn; a toast offering it closes then. */
export const getNotificationUndoWithdrawal = (undoId: string): AbortSignal | undefined =>
  pendingUndos.get(undoId)?.withdrawal.signal;

/** Keeps the entry while a toast shows its Undo; each hold ends with one `releaseNotificationUndo`. */
export const holdNotificationUndo = (undoId: string): void => {
  const entry = pendingUndos.get(undoId);

  if (entry) {
    entry.holds += 1;
  }
};

/** Ends one toast's hold. The entry goes with the last hold unless a listed notification still names it. */
export const releaseNotificationUndo = (undoId: string): void => {
  const entry = pendingUndos.get(undoId);

  if (!entry) {
    return;
  }

  entry.holds = Math.max(0, entry.holds - 1);

  if (isIdle(entry)) {
    pendingUndos.delete(undoId);
    removals.notify();
  }
};

/**
 * Marks the entries the listed notifications name, after the list changed. The rest lose their reference; those no
 * toast holds go: their record was dismissed, cleared or pushed past the list's limit.
 */
export const retainNotificationUndos = (referencedIds: ReadonlySet<string>): void => {
  let changed = false;

  for (const [undoId, entry] of pendingUndos) {
    entry.isReferenced = referencedIds.has(undoId);

    if (isIdle(entry)) {
      pendingUndos.delete(undoId);
      changed = true;
    }
  }

  if (changed) {
    removals.notify();
  }
};

/** Runs the undo at most once, however many surfaces offered it; a toast still offering it closes. */
export const runNotificationUndo = (undoId: string): void => {
  const entry = pendingUndos.get(undoId);

  if (!entry) {
    return;
  }

  withdraw(undoId, entry);
  removals.notify();
  entry.onUndo();
};

registerAccountOwnedResource({
  clear: () => {
    pendingUndos.clear();
    removals.notify();
  },
  name: 'notification-undo',
});
