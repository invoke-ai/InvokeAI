import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { ImageIndexProgressPanel } from './ImageIndexProgress';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 40);
    });
  });

const render = async (
  counts: { embedded: number; failed: number; pending: number; total: number },
  rate: number | null
) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <ImageIndexProgressPanel counts={counts} rate={rate} />
      </ChakraProvider>
    );
  });
};

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ImageIndexProgressPanel', () => {
  it('exposes the share done to assistive tech, not just as a painted width', async () => {
    await render({ embedded: 25, failed: 0, pending: 75, total: 100 }, 5);

    const bar = host!.querySelector('[role="progressbar"]')!;

    expect(bar.getAttribute('aria-valuenow')).toBe('25');
    expect(bar.getAttribute('aria-valuemax')).toBe('100');
    expect(bar.getAttribute('aria-label')).toBe('Image indexing progress');
  });

  it('shows the counts, the percentage and the time remaining', async () => {
    await render({ embedded: 25, failed: 0, pending: 75, total: 100 }, 5);

    const text = host!.textContent ?? '';

    expect(text).toContain('25 of 100 images');
    expect(text).toContain('25%');
    expect(text).toContain('About 15s remaining');
  });

  it('says it is measuring rather than showing a time it does not have', async () => {
    await render({ embedded: 0, failed: 0, pending: 100, total: 100 }, null);

    expect(host!.textContent).toContain('Estimating time remaining');
  });

  it('accounts for images that were given up on', async () => {
    await render({ embedded: 25, failed: 4, pending: 71, total: 100 }, 5);

    expect(host!.textContent).toContain('4 skipped after repeated failures');
  });
});
