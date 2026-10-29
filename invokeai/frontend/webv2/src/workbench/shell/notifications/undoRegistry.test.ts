import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  hasNotificationUndo,
  holdNotificationUndo,
  registerNotificationUndo,
  releaseNotificationUndo,
  runNotificationUndo,
} from './undoRegistry';

beforeEach(() => {
  accountLifecycle.invalidate();
});

describe('notification undo registry', () => {
  it('drops the oldest undo no toast has shown once more than twenty wait', () => {
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

  it('runs an undo at most once and not after its toast released it', () => {
    const onUndo = vi.fn();
    const pressed = registerNotificationUndo(onUndo);
    const closed = registerNotificationUndo(onUndo);

    runNotificationUndo(pressed);
    runNotificationUndo(pressed);
    releaseNotificationUndo(closed);
    runNotificationUndo(closed);

    expect(onUndo).toHaveBeenCalledOnce();
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
});
