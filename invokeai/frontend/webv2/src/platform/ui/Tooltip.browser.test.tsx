import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { CameraIcon } from 'lucide-react';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it } from 'vitest';

import { Toolbar, ToolbarButton } from './Toolbar';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** A one-shot action that is busy, and so disabled, while it runs. */
const BusyAction = () => {
  const [isBusy, setIsBusy] = useState(false);
  const start = useCallback(() => setIsBusy(true), []);
  return (
    <Toolbar left="200px" position="absolute" top="200px">
      <ToolbarButton disabled={isBusy} icon={CameraIcon} label="Export" loading={isBusy} onClick={start} />
    </Toolbar>
  );
};

/** Runs `run` while animation frames are held back, as when it all happens within one frame. */
const withinOneFrame = async (run: () => Promise<void>): Promise<void> => {
  const { cancelAnimationFrame, requestAnimationFrame } = window;
  let nextId = 1_000_000;
  window.requestAnimationFrame = () => (nextId += 1);
  window.cancelAnimationFrame = () => undefined;
  try {
    await run();
  } finally {
    window.requestAnimationFrame = requestAnimationFrame;
    window.cancelAnimationFrame = cancelAnimationFrame;
  }
};

const exportTooltip = () =>
  [...document.querySelectorAll<HTMLElement>('[data-scope="tooltip"][data-part="content"]')].find(
    (candidate) => candidate.textContent === 'Export'
  );

it('keeps a tooltip that closes before it was placed out of sight while it fades', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <BusyAction />
      </ChakraProvider>
    )
  );
  const button = host.querySelector<HTMLButtonElement>('button[aria-label="Export"]')!;

  // Keyboard focus opens the tooltip, which is placed a frame later; activating the button first closes it.
  await withinOneFrame(async () => {
    // Async act lets the tooltip render open; frames stay held, so it is not placed yet.
    await act(() => {
      document.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Tab' }));
      button.focus();
      return Promise.resolve();
    });
    expect(exportTooltip()?.dataset.state).toBe('open');
    await act(() => {
      button.click();
      return Promise.resolve();
    });
  });

  expect(button.disabled).toBe(true);
  // Still mounted for its exit animation; it must not fade out at the page's top-left corner.
  expect(exportTooltip()?.dataset.state).toBe('closed');
  expect(exportTooltip()!.getBoundingClientRect().bottom).toBeLessThanOrEqual(0);
});
