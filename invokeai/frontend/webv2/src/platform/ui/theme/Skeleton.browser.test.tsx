import { ChakraProvider, Skeleton } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it } from 'vitest';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('sweeps its highlight at an even pace', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <Skeleton data-testid="skeleton" h="4" w="20" />
      </ChakraProvider>
    )
  );

  const style = getComputedStyle(host.querySelector('[data-testid="skeleton"]')!);
  expect(style.animationName).toBe('bg-position');
  expect(style.animationTimingFunction).toBe('linear');
  expect(style.animationIterationCount).toBe('infinite');
});
