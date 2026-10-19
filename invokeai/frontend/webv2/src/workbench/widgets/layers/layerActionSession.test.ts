import { describe, expect, it } from 'vitest';

import { createLayerActionSession } from './layerActionSession';

describe('createLayerActionSession', () => {
  it('allows only one request at a time and permits a new request after current finish', () => {
    const session = createLayerActionSession();
    const first = session.begin();

    expect(first).not.toBeNull();
    expect(session.begin()).toBeNull();
    expect(session.isCurrent(first!.token)).toBe(true);
    session.finish(first!.token);
    expect(session.isCurrent(first!.token)).toBe(false);
    expect(session.begin()).not.toBeNull();
  });

  it('cancel aborts and invalidates the active request while keeping the session reusable', () => {
    const session = createLayerActionSession();
    const first = session.begin()!;

    session.cancel();

    expect(first.signal.aborted).toBe(true);
    expect(session.isCurrent(first.token)).toBe(false);
    const second = session.begin();
    expect(second).not.toBeNull();
    expect(second!.token).not.toBe(first.token);
  });

  it('ignores stale finish so an old completion cannot clear the current request', () => {
    const session = createLayerActionSession();
    const first = session.begin()!;
    session.cancel();
    const second = session.begin()!;

    session.finish(first.token);

    expect(session.isCurrent(second.token)).toBe(true);
    expect(session.begin()).toBeNull();
  });

  it('rejects a late completion after close/cancel invalidates its token', () => {
    const session = createLayerActionSession();
    const request = session.begin()!;
    const wasCurrent = () => session.isCurrent(request.token);

    session.cancel();

    expect(wasCurrent()).toBe(false);
  });
});
