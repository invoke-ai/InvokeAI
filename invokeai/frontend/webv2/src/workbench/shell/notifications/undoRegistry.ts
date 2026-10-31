import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';

/**
 * The closures behind `{ kind: 'undo' }` notification actions. Notifications are plain workbench state, so they name
 * an undo by id and the closure stays here, for this tab and account only. An entry lives until it runs or the toast
 * showing it closes. A shown entry is held, so its Undo button always works; only entries no toast has shown (say,
 * recorded while the editor shell was not mounted) count against the bound, oldest dropped first.
 */
const MAX_IDLE_UNDOS = 20;
const pendingUndos = new Map<
  string,
  { isHeld: boolean; onUndo: () => void; supersedes: string | undefined; withdrawal: AbortController }
>();
let nextUndoId = 0;

const dropOldestIdleUndo = (): void => {
  const idleIds = [...pendingUndos].flatMap(([undoId, entry]) => (entry.isHeld ? [] : [undoId]));

  if (idleIds.length > MAX_IDLE_UNDOS) {
    pendingUndos.delete(idleIds[0]!);
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

  pendingUndos.set(undoId, { isHeld: false, onUndo, supersedes, withdrawal: new AbortController() });
  dropOldestIdleUndo();

  return undoId;
};

/**
 * Withdraws every undo registered under `supersedes` (their toasts close, their buttons go), for an owner that knows
 * the change they revert is no longer the latest: undone another way, or built on since.
 */
export const withdrawNotificationUndos = (supersedes: string): void => {
  for (const [undoId, entry] of pendingUndos) {
    if (entry.supersedes === supersedes) {
      pendingUndos.delete(undoId);
      entry.withdrawal.abort();
    }
  }
};

export const hasNotificationUndo = (undoId: string): boolean => pendingUndos.has(undoId);

/** Aborts when a newer undo supersedes this one; a toast offering it closes then. */
export const getNotificationUndoWithdrawal = (undoId: string): AbortSignal | undefined =>
  pendingUndos.get(undoId)?.withdrawal.signal;

/** Keeps the entry while a toast shows its Undo; `releaseNotificationUndo` ends that. */
export const holdNotificationUndo = (undoId: string): void => {
  const entry = pendingUndos.get(undoId);

  if (entry) {
    entry.isHeld = true;
  }
};

/** Runs the undo at most once, however many surfaces offered it. */
export const runNotificationUndo = (undoId: string): void => {
  const entry = pendingUndos.get(undoId);

  pendingUndos.delete(undoId);
  entry?.onUndo();
};

export const releaseNotificationUndo = (undoId: string): void => {
  pendingUndos.delete(undoId);
};

registerAccountOwnedResource({
  clear: () => pendingUndos.clear(),
  name: 'notification-undo',
});
