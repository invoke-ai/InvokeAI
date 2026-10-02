import { ChakraProvider, Popover, Portal, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { PopoverContent } from './Popover';
import { Scrollable } from './Scrollable';

/** Cover Zag's inline-width overflow and phantom thumbs from zero-size popover mounts. */

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** No spaces and no hyphens: nothing the layout can break on, so it contributes its whole width to min-content. */
const UNBREAKABLE = 'AVeryLongUnbreakableWorkflowNameThatHasNowhereToWrap';

/** Zag's measure timing varies under load; poll instead of guessing a delay. */
const waitFor = async (predicate: () => boolean, timeoutMs = 3000): Promise<void> => {
  const startedAt = Date.now();

  await act(async () => {
    while (!predicate() && Date.now() - startedAt < timeoutMs) {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 25);
      });
    }
  });
};

const renderScrollable = async (children: ReactNode, orientation: 'horizontal' | 'vertical'): Promise<HTMLElement> => {
  host = document.createElement('div');
  host.style.cssText = 'width:200px;height:80px;';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Scrollable label="subject" orientation={orientation}>
          {children}
        </Scrollable>
      </ChakraProvider>
    );
  });

  return host.querySelector<HTMLElement>('[role="region"][aria-label="subject"]')!;
};

describe('Scrollable', () => {
  it('does not let a vertical area scroll sideways, whatever its content asks for', async () => {
    const viewport = await renderScrollable(<div>{UNBREAKABLE}</div>, 'vertical');

    expect(viewport.clientWidth).toBeGreaterThan(0);
    expect(viewport.scrollWidth).toBeLessThanOrEqual(viewport.clientWidth + 1);
  });

  it('still lets a horizontal strip outgrow its viewport, scrollbar shown', async () => {
    const viewport = await renderScrollable(
      <div style={{ display: 'flex', whiteSpace: 'nowrap' }}>
        {UNBREAKABLE}
        {UNBREAKABLE}
      </div>,
      'horizontal'
    );

    expect(viewport.scrollWidth).toBeGreaterThan(viewport.clientWidth);

    const scrollbar = document.querySelector<HTMLElement>(
      '[data-scope="scroll-area"][data-part="scrollbar"][data-orientation="horizontal"]'
    )!;

    await waitFor(() => getComputedStyle(scrollbar).display !== 'none');
    expect(getComputedStyle(scrollbar).display).not.toBe('none');
  });

  it('hides the vertical scrollbar when only sideways spill exists', async () => {
    // Stock Chakra checks both axes together; horizontal overflow must not show a vertical thumb.
    const viewport = await renderScrollable(
      <div style={{ whiteSpace: 'nowrap' }}>
        {UNBREAKABLE}
        {UNBREAKABLE}
        {UNBREAKABLE}
      </div>,
      'vertical'
    );

    expect(viewport.scrollHeight).toBeLessThanOrEqual(viewport.clientHeight);

    const scrollbar = document.querySelector<HTMLElement>(
      '[data-scope="scroll-area"][data-part="scrollbar"][data-orientation="vertical"]'
    )!;

    await waitFor(() => getComputedStyle(scrollbar).display === 'none');

    expect(scrollbar).not.toBeNull();
    expect(scrollbar.hasAttribute('data-overflow-x')).toBe(true);
    expect(viewport.scrollWidth).toBeGreaterThan(viewport.clientWidth);
    expect(getComputedStyle(scrollbar).display).toBe('none');
  });

  it('hides the scrollbar for non-overflowing content inside a popover', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <Popover.Root lazyMount open unmountOnExit>
            <Portal>
              <Popover.Positioner>
                <PopoverContent w="22rem">
                  <Popover.Body p="2.5">
                    <Scrollable h="14rem" label="preview">
                      <Text fontSize="xs">one short row</Text>
                    </Scrollable>
                  </Popover.Body>
                </PopoverContent>
              </Popover.Positioner>
            </Portal>
          </Popover.Root>
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

    const scrollbar = document.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="scrollbar"]')!;

    expect(scrollbar).not.toBeNull();
    await waitFor(() => !scrollbar.hasAttribute('data-overflow-y'));
    expect(scrollbar.hasAttribute('data-overflow-y')).toBe(false);
    expect(getComputedStyle(scrollbar).display).toBe('none');
  });
});
