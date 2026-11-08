import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { useModifierHeld } from './useModifierHeld';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ShiftProbe = () => <output>{useModifierHeld('Shift') ? 'held' : 'free'}</output>;

const renderProbe = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root?.render(<ShiftProbe />));

  return () => host?.querySelector('output')?.textContent;
};

afterEach(async () => {
  await userEvent.keyboard('{/Shift}');
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('useModifierHeld', () => {
  it('follows the key down and up', async () => {
    const read = await renderProbe();

    await act(() => userEvent.keyboard('{Shift>}'));
    expect(read()).toBe('held');
    await act(() => userEvent.keyboard('{/Shift}'));
    expect(read()).toBe('free');
  });

  // A release the page never hears (another window took the keys, a listener ate it) is corrected by the next
  // pointer event, which carries the live modifier state.
  it('resyncs from a pointer event when the release was missed', async () => {
    const read = await renderProbe();

    await act(() => userEvent.keyboard('{Shift>}'));
    expect(read()).toBe('held');
    await act(() => {
      document.body.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, shiftKey: false }));
    });

    expect(read()).toBe('free');
  });

  it('hears the release even when a listener below stops its propagation', async () => {
    const read = await renderProbe();
    const input = document.createElement('input');
    host?.append(input);
    input.addEventListener('keyup', (event) => event.stopPropagation());
    input.focus();

    await act(() => userEvent.keyboard('{Shift>}'));
    expect(read()).toBe('held');
    await act(() => userEvent.keyboard('{/Shift}'));

    expect(read()).toBe('free');
  });

  it('lets go when the window loses focus', async () => {
    const read = await renderProbe();

    await act(() => userEvent.keyboard('{Shift>}'));
    await act(() => {
      window.dispatchEvent(new Event('blur'));
    });

    expect(read()).toBe('free');
  });
});
