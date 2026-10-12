import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useExitRetainedValue } from './useExitRetainedValue';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const capture = vi.fn<(result: ReturnType<typeof useExitRetainedValue<{ name: string }>>) => void>();

const Probe = ({ subject }: { subject: { name: string } | null }) => {
  capture(useExitRetainedValue(subject));
  return null;
};

const render = (subject: { name: string } | null) => act(() => root?.render(<Probe subject={subject} />));
const retained = () => capture.mock.lastCall![0];

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  capture.mockClear();
});

describe('useExitRetainedValue', () => {
  it('keeps the last open subject through the exit and drops it on release, never while open', async () => {
    host = document.createElement('div');
    root = createRoot(host);
    const first = { name: 'first' };
    const renamed = { name: 'renamed' };

    await render(first);
    const opened = retained().generation;
    await act(() => retained().release());
    expect(retained()).toMatchObject({ isOpen: true, value: first });

    // A subject rebuilt while open updates in place rather than remounting.
    await render(renamed);
    expect(retained()).toMatchObject({ generation: opened, isOpen: true, value: renamed });

    await render(null);
    expect(retained()).toMatchObject({ generation: opened, isOpen: false, value: renamed });

    await act(() => retained().release());
    expect(retained().value).toBeNull();
  });

  it('gives a reopen during the exit a fresh generation and ignores the stale release', async () => {
    host = document.createElement('div');
    root = createRoot(host);
    const subject = { name: 'subject' };

    await render(subject);
    const opened = retained().generation;
    await render(null);
    const releaseFirstExit = retained().release;

    await render(subject);
    expect(retained()).toMatchObject({ isOpen: true, value: subject });
    expect(retained().generation).toBe(opened + 1);

    await act(() => releaseFirstExit());
    expect(retained()).toMatchObject({ isOpen: true, value: subject });
  });
});
