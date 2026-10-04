import { ChakraProvider, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { TYPE_SCALE } from './scale';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const rem = (value: string) => `${Number.parseFloat(value) * 16}px`;

const renderText = async (fontSize?: 'lg' | 'xl') => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Text data-testid="text" fontSize={fontSize}>
          Text
        </Text>
      </ChakraProvider>
    );
  });

  return getComputedStyle(host.querySelector('[data-testid="text"]')!);
};

describe('base text', () => {
  it('renders text that names no size at the md font size', async () => {
    expect((await renderText()).fontSize).toBe(rem(TYPE_SCALE.md.fontSize));
  });

  // Text that sets only a font size keeps leading in proportion to it rather than inheriting a fixed line height.
  it.each([undefined, 'lg', 'xl'] as const)('keeps proportional line height for %s text', async (fontSize) => {
    const style = await renderText(fontSize);

    expect(Number.parseFloat(style.lineHeight)).toBeCloseTo(Number.parseFloat(style.fontSize) * 1.5, 1);
  });
});
