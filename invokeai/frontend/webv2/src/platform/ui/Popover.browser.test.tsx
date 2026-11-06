import { Box, ChakraProvider, Popover } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { PopoverContent } from './Popover';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const PLACEMENTS = {
  bottom: { placement: 'bottom' },
  top: { placement: 'top' },
} as const;

const renderPopover = async (placement: keyof typeof PLACEMENTS) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <Box left="50vw" position="fixed" top="50vh">
          <Popover.Root positioning={PLACEMENTS[placement]}>
            <Popover.Trigger>Open</Popover.Trigger>
            <Popover.Positioner>
              <PopoverContent w="12rem">
                <Popover.Body p="0">
                  <Box as="button" h="8" w="full">
                    Inside
                  </Box>
                </Popover.Body>
              </PopoverContent>
            </Popover.Positioner>
          </Popover.Root>
        </Box>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
  await act(() => userEvent.click(page.getByRole('button', { name: 'Open' })));
  const content = document.querySelector<HTMLElement>('[data-scope="popover"][data-part="content"]')!;
  await vi.waitFor(() => expect(content.dataset.placement).toBe(placement));
  await Promise.all(content.getAnimations().map((animation) => animation.finished));
  const arrow = content.querySelector<HTMLElement>('[data-part="arrow"]')!;
  return { arrow, content, tip: arrow.querySelector<HTMLElement>('[data-part="arrow-tip"]')! };
};

describe('PopoverContent arrow', () => {
  it.each(['bottom', 'top'] as const)(
    'centres on the content outer edge facing the trigger, as a menu arrow does (%s)',
    async (placement) => {
      const { content, tip } = await renderPopover(placement);
      const trigger = page.getByRole('button', { name: 'Open' }).element().getBoundingClientRect();
      const edge = content.getBoundingClientRect();
      const tipBox = tip.getBoundingClientRect();

      expect((tipBox.top + tipBox.bottom) / 2).toBeCloseTo(placement === 'bottom' ? edge.top : edge.bottom, 0);
      expect((tipBox.left + tipBox.right) / 2).toBeCloseTo((trigger.left + trigger.right) / 2, 0);
    }
  );

  it('lets the pointer through its transparent box to the content beneath', async () => {
    const { arrow, content } = await renderPopover('bottom');
    const box = arrow.getBoundingClientRect();
    const inside = page.getByRole('button', { name: 'Inside' }).element();
    const y = content.getBoundingClientRect().top + 4;
    expect(y).toBeLessThan(box.bottom);

    expect(document.elementFromPoint(box.left + box.width / 2, y)).toBe(inside);
  });

  it('stays open when its visible tip is clicked, as a click on the content would', async () => {
    const { content, tip } = await renderPopover('bottom');
    const tipBox = tip.getBoundingClientRect();
    // On the tip's outer half, above the content's edge: the part a user sees.
    const point = { x: (tipBox.left + tipBox.right) / 2, y: content.getBoundingClientRect().top - 3 };
    expect(document.elementFromPoint(point.x, point.y)).toBe(tip);

    const body = document.body.getBoundingClientRect();
    await act(() =>
      userEvent.click(page.elementLocator(document.body), {
        position: { x: point.x - body.left, y: point.y - body.top },
      })
    );
    await new Promise((resolve) => {
      globalThis.setTimeout(resolve, 100);
    });

    expect(content.dataset.state).toBe('open');
    expect(page.getByRole('button', { name: 'Open' }).element().getAttribute('aria-expanded')).toBe('true');
  });
});
