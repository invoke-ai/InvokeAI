import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  getNotificationUndoWithdrawal,
  hasNotificationUndo,
  holdNotificationUndo,
  registerNotificationUndo,
  releaseNotificationUndo,
  retainNotificationUndos,
  runNotificationUndo,
} from './undoRegistry';

beforeEach(() => {
  accountLifecycle.invalidate();
});

describe('notification undo registry', () => {
  it('drops the oldest undo nothing lists or shows once more than twenty wait', () => {
    const ids = Array.from({ length: 21 }, () => registerNotificationUndo(vi.fn()));

    expect(hasNotificationUndo(ids[0]!)).toBe(false);
    expect(ids.slice(1).every(hasNotificationUndo)).toBe(true);
  });

  it('never drops an undo a toast is showing, so its button keeps working', () => {
    const onUndo = vi.fn();
    const shown = registerNotificationUndo(onUndo);
    holdNotificationUndo(shown);

    for (let index = 0; index < 25; index += 1) {
      registerNotificationUndo(vi.fn());
    }
    runNotificationUndo(shown);

    expect(onUndo).toHaveBeenCalledOnce();
  });

  it('runs an undo at most once, and closes a toast still offering it', () => {
    const onUndo = vi.fn();
    const undoId = registerNotificationUndo(onUndo);
    holdNotificationUndo(undoId);
    const withdrawal = getNotificationUndoWithdrawal(undoId)!;

    runNotificationUndo(undoId);
    runNotificationUndo(undoId);

    expect(onUndo).toHaveBeenCalledOnce();
    expect(withdrawal.aborted).toBe(true);
  });

  it('frees an undo nothing lists once its toast releases it', () => {
    const onUndo = vi.fn();
    const undoId = registerNotificationUndo(onUndo);
    holdNotificationUndo(undoId);

    releaseNotificationUndo(undoId);
    runNotificationUndo(undoId);

    expect(onUndo).not.toHaveBeenCalled();
  });

  it('keeps an undo its notice still lists after the toast closed, and frees it once the notice goes', () => {
    const onUndo = vi.fn();
    const listed = registerNotificationUndo(onUndo);
    const dismissed = registerNotificationUndo(vi.fn());
    holdNotificationUndo(listed);
    retainNotificationUndos(new Set([listed, dismissed]));

    releaseNotificationUndo(listed);
    retainNotificationUndos(new Set([listed]));

    expect(hasNotificationUndo(listed)).toBe(true);
    expect(hasNotificationUndo(dismissed)).toBe(false);
    runNotificationUndo(listed);
    expect(onUndo).toHaveBeenCalledOnce();

    const cleared = registerNotificationUndo(vi.fn());
    retainNotificationUndos(new Set([cleared]));
    retainNotificationUndos(new Set());
    expect(hasNotificationUndo(cleared)).toBe(false);
  });

  it('keeps an undo whose notice left the list while a toast still holds it', () => {
    const undoId = registerNotificationUndo(vi.fn());
    holdNotificationUndo(undoId);

    retainNotificationUndos(new Set());
    expect(hasNotificationUndo(undoId)).toBe(true);

    releaseNotificationUndo(undoId);
    expect(hasNotificationUndo(undoId)).toBe(false);
  });

  it('caps only undos no notice lists, so a listed Undo outlasts many unlisted ones', () => {
    const listed: string[] = [];
    // Each recorded the way the editor does it: registered, then listed.
    for (let index = 0; index < 30; index += 1) {
      listed.push(registerNotificationUndo(vi.fn()));
      retainNotificationUndos(new Set(listed));
    }

    const unlisted = Array.from({ length: 21 }, () => registerNotificationUndo(vi.fn()));

    expect(listed.every(hasNotificationUndo)).toBe(true);
    expect(hasNotificationUndo(unlisted[0]!)).toBe(false);
    expect(unlisted.slice(1).every(hasNotificationUndo)).toBe(true);
  });

  it('forgets every undo when the account changes', () => {
    const onUndo = vi.fn();
    const undoId = registerNotificationUndo(onUndo);
    holdNotificationUndo(undoId);

    accountLifecycle.invalidate();
    runNotificationUndo(undoId);

    expect(hasNotificationUndo(undoId)).toBe(false);
    expect(onUndo).not.toHaveBeenCalled();
  });

  it('withdraws an older undo of the same thing when a newer one arrives, closing its toast', () => {
    const older = vi.fn();
    const olderId = registerNotificationUndo(older, 'generate-settings:project-1');
    holdNotificationUndo(olderId);
    const withdrawal = getNotificationUndoWithdrawal(olderId)!;
    const otherProject = registerNotificationUndo(vi.fn(), 'generate-settings:project-2');

    const newerId = registerNotificationUndo(vi.fn(), 'generate-settings:project-1');

    expect(withdrawal.aborted).toBe(true);
    expect(hasNotificationUndo(olderId)).toBe(false);
    runNotificationUndo(olderId);
    expect(older).not.toHaveBeenCalled();
    expect(hasNotificationUndo(newerId)).toBe(true);
    expect(hasNotificationUndo(otherProject)).toBe(true);
  });
});
