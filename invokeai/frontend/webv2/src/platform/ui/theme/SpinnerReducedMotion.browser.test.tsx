import { ChakraProvider, Skeleton, Spinner } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

/**
 * Reduce-motion kills every animation token, which froze spinners into a
 * static arc that reads as a broken icon. A loading spinner is essential
 * status (WCAG 2.3.3 exempts essential motion), so it slows instead.
 */

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  delete document.documentElement.dataset.reduceMotion;
  host = null;
  root = null;
});

const render = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <Spinner data-testid="spinner" size="sm" />
        <Skeleton data-testid="skeleton" h="4" w="20" />
      </ChakraProvider>
    )
  );

  return {
    skeleton: getComputedStyle(host.querySelector('[data-testid="skeleton"]')!),
    spinner: getComputedStyle(host.querySelector('[data-testid="spinner"]')!),
  };
};

describe('spinner under reduce-motion', () => {
  it('spins normally with motion enabled', async () => {
    const { spinner } = await render();

    expect(spinner.animationName).not.toBe('none');
    expect(spinner.animationDuration).toBe('0.5s');
  });

  it('slows instead of freezing with motion reduced, while skeletons stay still', async () => {
    document.documentElement.dataset.reduceMotion = 'true';

    const { skeleton, spinner } = await render();

    expect(spinner.animationName).not.toBe('none');
    expect(spinner.animationDuration).toBe('2s');
    expect(spinner.animationIterationCount).toBe('infinite');
    expect(skeleton.animationName).toBe('none');
  });
});
