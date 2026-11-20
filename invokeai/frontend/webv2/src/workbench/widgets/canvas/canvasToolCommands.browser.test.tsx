import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { showCanvasRefusalNotice } from './canvasToolCommands';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => {
    toaster.remove();
    root?.unmount();
  });
  host?.remove();
  host = null;
  root = null;
});

const openTitles = (): (string | null | undefined)[] =>
  [...document.querySelectorAll<HTMLElement>('[data-part="root"][data-scope="toast"]')]
    .filter((toast) => toast.getAttribute('data-state') === 'open')
    .map((toast) => toast.querySelector('[data-part="title"]')?.textContent);

describe('showCanvasRefusalNotice', () => {
  it('keeps one toast per key however often the key is pressed', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <AppToaster />
        </ChakraProvider>
      )
    );

    await act(() => {
      for (let press = 0; press < 4; press += 1) {
        showCanvasRefusalNotice('merge-down:a:no-layer-below', 'Can’t merge “A” down', 'There is no layer below it.');
      }
      showCanvasRefusalNotice(
        'transform:ineligible:b',
        'Can’t transform “B”',
        'Transform works on layers with content.'
      );
    });

    await expect.poll(openTitles).toEqual(['Can’t transform “B”', 'Can’t merge “A” down']);
  });
});
