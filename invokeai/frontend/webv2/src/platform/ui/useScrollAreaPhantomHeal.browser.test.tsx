import { act, useReducer, useRef } from 'react';
import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

import { useScrollAreaPhantomHeal } from './useScrollAreaPhantomHeal';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.restoreAllMocks();
});

/** Zag's markup after a stale measure: the scrollbar claims vertical overflow the viewport does not have. */
const PhantomArea = ({ onHealScroll }: { onHealScroll: () => void }) => {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [, bump] = useReducer((count: number) => count + 1, 0);

  useScrollAreaPhantomHeal(viewportRef);

  return (
    <div>
      <div
        ref={viewportRef}
        style={{ height: 80, overflow: 'auto' }}
        onScroll={() => {
          onHealScroll();
          // Virtualizers with useFlushSync commit scroll snapshots this way.
          flushSync(bump);
        }}
      >
        short
      </div>
      <div data-overflow-y="" data-part="scrollbar" />
    </div>
  );
};

const mount = (onHealScroll: () => void) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  act(() => root?.render(<PhantomArea onHealScroll={onHealScroll} />));
};

const nextFrame = () =>
  new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
  });

it('heals a phantom scrollbar before the next paint without flushing inside React effects', async () => {
  const consoleError = vi.spyOn(console, 'error');
  const onHealScroll = vi.fn();

  mount(onHealScroll);
  expect(onHealScroll).not.toHaveBeenCalled();

  await nextFrame();

  expect(onHealScroll).toHaveBeenCalled();
  expect(consoleError.mock.calls.flat().filter((arg) => String(arg).includes('flushSync'))).toEqual([]);
});

it('does not heal after unmounting before the deferred check runs', async () => {
  const onHealScroll = vi.fn();

  mount(onHealScroll);
  act(() => root?.unmount());
  root = null;

  await new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, 700);
  });

  expect(onHealScroll).not.toHaveBeenCalled();
});
