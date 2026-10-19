import { ChakraProvider, createListCollection } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { Select } from './Select';

const RATIO_ITEMS = [
  { label: '16:9', value: '16:9' },
  { label: '1:1', value: '1:1' },
];
const RATIO_VALUE = ['16:9'];

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('Select', () => {
  let host: HTMLDivElement;
  let root: Root;
  const mount = (items: { label: string; value: string }[]) =>
    act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select aria-label="Scheduler" collection={createListCollection({ items })} />
        </ChakraProvider>
      );
    });
  const trigger = () => host.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]')!;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('shows the selected label by default and lets a caller name the value instead', async () => {
    // An omitted trigger-text override must fall back to the selection, not an empty display.
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select
            aria-label="Aspect ratio"
            collection={createListCollection({ items: RATIO_ITEMS })}
            value={RATIO_VALUE}
          />
        </ChakraProvider>
      );
    });
    expect(trigger().textContent).toContain('16:9');

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select
            aria-label="Aspect ratio"
            collection={createListCollection({ items: RATIO_ITEMS })}
            value={RATIO_VALUE}
            valueText="From the first frame"
          />
        </ChakraProvider>
      );
    });
    expect(trigger().textContent).toContain('From the first frame');
    expect(trigger().textContent).not.toContain('16:9');
  });

  it('disables the trigger while the collection is empty, and opens once it has items', async () => {
    await mount([]);
    expect(trigger().disabled).toBe(true);
    await act(() => trigger().click());
    expect(document.querySelector('[role="listbox"]')).toBeNull();

    await mount([{ label: 'Euler', value: 'euler' }]);
    expect(trigger().disabled).toBe(false);
    await act(() => trigger().click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(1);
  });
});
