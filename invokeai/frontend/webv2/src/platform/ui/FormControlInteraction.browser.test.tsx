import { ChakraProvider, createListCollection, Input, Stack } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { Select } from './Select';

const collection = createListCollection({
  items: [{ label: 'Euler', value: 'euler' }],
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('form control interaction styles', () => {
  it('keeps the select focus border while hovered', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Stack>
            <Input aria-label="Input baseline" />
            <Select aria-label="Sampler" collection={collection} />
          </Stack>
        </ChakraProvider>
      );
    });

    const trigger = host.querySelector<HTMLButtonElement>('[data-part="trigger"]')!;
    const input = host.querySelector<HTMLInputElement>('input[aria-label="Input baseline"]')!;
    const accentProbe = document.createElement('div');
    accentProbe.style.border = '1px solid var(--chakra-colors-accent-solid)';
    document.body.append(accentProbe);
    const accentBorderColor = getComputedStyle(accentProbe).borderTopColor;
    accentProbe.remove();

    await act(async () => {
      await userEvent.tab();
      await new Promise((resolve) => {
        globalThis.setTimeout(resolve, 200);
      });
    });
    expect(getComputedStyle(input).outlineStyle).toBe('none');
    expect(getComputedStyle(input).borderTopColor).toBe(accentBorderColor);
    await act(async () => {
      await userEvent.tab();
      await new Promise((resolve) => {
        globalThis.setTimeout(resolve, 200);
      });
    });
    expect(document.activeElement).toBe(trigger);
    expect(getComputedStyle(trigger).outlineStyle).toBe('none');
    expect(getComputedStyle(trigger).borderTopColor).toBe(accentBorderColor);

    await act(async () => {
      await userEvent.hover(trigger);
      trigger.click();
      await new Promise((resolve) => {
        globalThis.setTimeout(resolve, 200);
      });
    });
    expect(trigger.getAttribute('data-state')).toBe('open');
    const hoveredFocusBorderColor = getComputedStyle(trigger).borderTopColor;
    expect(hoveredFocusBorderColor).toBe(accentBorderColor);

    await act(async () => {
      await userEvent.unhover(trigger);
      await new Promise((resolve) => {
        globalThis.setTimeout(resolve, 200);
      });
    });
    expect(hoveredFocusBorderColor).toBe(getComputedStyle(trigger).borderTopColor);
  });
});
