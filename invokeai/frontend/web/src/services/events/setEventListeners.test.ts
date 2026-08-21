import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const read = (file: string) => readFileSync(fileURLToPath(new URL(file, import.meta.url)), 'utf8');

// Wiring checks: the behavior behind cancelScheduledRetries is covered by real tests in
// onInvocationComplete.test.ts. What cannot be seen from there is whether anything calls it.
describe('socket session teardown', () => {
  it('hands back a disposer for the work its listeners may have scheduled', () => {
    const source = read('./setEventListeners.tsx');
    expect(source).toContain('onInvocationComplete.cancelScheduledRetries();');
  });

  it('disposes those listeners before the socket goes away', () => {
    // A pending refetch holds an event from the session being torn down but dispatches into
    // whatever store is current when it fires — the next user's, after a logout or account switch.
    const source = read('./useSocketIO.ts');
    expect(source).toContain('const disposeEventListeners = setEventListeners(');
    const cleanup = source.slice(source.indexOf('return () => {'), source.lastIndexOf('};'));
    expect(cleanup).toContain('disposeEventListeners();');
  });
});
