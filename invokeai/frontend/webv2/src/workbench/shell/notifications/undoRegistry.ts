import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';

/**
 * The closures behind `{ kind: 'undo' }` notification actions. Notifications are plain workbench state, so they name
 * an undo by id and the closure stays here, for this tab and account only. An entry lives until it runs or the toast
 * showing it closes. A shown entry is held, so its Undo button always works; only entries no toast has shown (say,
 * recorded while the editor shell was not mounted) count against the bound, oldest dropped first.
 */
const MAX_IDLE_UNDOS = 20;
const pendingUndos = new Map<string, { isHeld: boolean; onUndo: () => void }>();
let nextUndoId = 0;

const dropOldestIdleUndo = (): void => {
  const idleIds = [...pendingUndos].flatMap(([undoId, entry]) => (entry.isHeld ? [] : [undoId]));

  if (idleIds.length > MAX_IDLE_UNDOS) {
    pendingUndos.delete(idleIds[0]!);
  }
};

export const registerNotificationUndo = (onUndo: () => void): string => {
  const undoId = `undo-${++nextUndoId}`;

  pendingUndos.set(undoId, { isHeld: false, onUndo });
  dropOldestIdleUndo();

  return undoId;
};

export const hasNotificationUndo = (undoId: string): boolean => pendingUndos.has(undoId);

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
