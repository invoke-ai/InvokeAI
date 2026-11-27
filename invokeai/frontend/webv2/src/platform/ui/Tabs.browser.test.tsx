import { ChakraProvider, Tabs } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('tab focus feedback', () => {
  it('keeps the keyboard focus outline through hover without disturbing the selected tab', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <Tabs.Root value="selected" variant="line">
            <Tabs.List>
              <Tabs.Trigger aria-label="line idle" value="idle">
                Idle
              </Tabs.Trigger>
              <Tabs.Trigger aria-label="line selected" value="selected">
                Selected
              </Tabs.Trigger>
            </Tabs.List>
          </Tabs.Root>
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 0);
      });
    });

    const idle = host.querySelector<HTMLButtonElement>('[aria-label="line idle"]')!;
    const selected = host.querySelector<HTMLButtonElement>('[aria-label="line selected"]')!;

    await act(async () => {
      await userEvent.tab();
      await userEvent.keyboard('{ArrowLeft}');
    });
    expect(document.activeElement).toBe(idle);
    const focusOutline = getComputedStyle(idle).outline;
    expect(focusOutline).not.toBe('none');

    await act(async () => {
      await userEvent.hover(idle);
    });
    expect(getComputedStyle(idle).outline).toBe(focusOutline);
    expect(selected.dataset.selected).toBe('');
  });
});
