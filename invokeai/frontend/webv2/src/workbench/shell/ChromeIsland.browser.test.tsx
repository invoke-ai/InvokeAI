import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ComponentProps, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ChromeIsland } from './ChromeIsland';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: { common: { scrollToEnd: 'Scroll to end', scrollToStart: 'Scroll to start' } } } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  delete document.documentElement.dataset.reduceMotion;
});

/** An island holding `row`, in a parent `width` pixels wide. */
const mountIsland = async (row: ReactNode, { dir = 'ltr', width }: { dir?: 'ltr' | 'rtl'; width: number }) => {
  const tree = (nextRow: ReactNode) => (
    <I18nextProvider i18n={i18n}>
      <ChakraProvider value={system}>
        <Box dir={dir} display="flex" w={`${width}px`}>
          <ChromeIsland>{nextRow}</ChromeIsland>
        </Box>
      </ChakraProvider>
    </I18nextProvider>
  );

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root!.render(tree(row));
    await nextFrame();
  });

  return {
    rerender: async (nextRow: ReactNode) => {
      await act(async () => {
        root!.render(tree(nextRow));
        await nextFrame();
      });
    },
  };
};

const controls = (count: number, props: ComponentProps<typeof Box> = {}) =>
  Array.from({ length: count }, (_, index) => (
    <Box key={index} aria-label={`Control ${index + 1}`} as="button" boxSize="6" flexShrink={0} {...props} />
  ));

/** An island holding `count` 24px controls, in a parent `width` pixels wide. */
const renderIsland = async ({ count, dir, width }: { count: number; dir?: 'ltr' | 'rtl'; width: number }) => {
  const { rerender } = await mountIsland(controls(count), { dir, width });

  return { rerender: (nextCount: number) => rerender(controls(nextCount)) };
};

// Resize observations are delivered after layout, before the next paint.
const nextFrame = () =>
  new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });

const island = () => host!.querySelector<HTMLElement>('[data-center-chrome-island]')!;
const scroller = () => island().firstElementChild as HTMLElement;
const cue = (edge: 'end' | 'start') => host!.querySelector<HTMLElement>(`[data-overflow-cue="${edge}"]`);
const cueButton = (edge: 'end' | 'start') => cue(edge)?.querySelector('button') ?? null;
const shownCues = () =>
  [...host!.querySelectorAll('[data-overflow-cue]')].map((node) => node.getAttribute('data-overflow-cue'));
const control = (index: number) => host!.querySelector<HTMLElement>(`[aria-label="Control ${index}"]`)!;
const scrollRange = () => scroller().scrollWidth - scroller().clientWidth;

describe('ChromeIsland overflow', () => {
  it('scrolls a row wider than its parent and cues only the end that hides controls', async () => {
    await renderIsland({ count: 12, width: 160 });

    expect(island().getBoundingClientRect().width).toBeLessThanOrEqual(160);
    expect(scrollRange()).toBeGreaterThan(0);
    expect(shownCues()).toEqual(['end']);
    expect(cueButton('end')?.getAttribute('aria-label')).toBe('Scroll to end');
    // A pointer shortcut only: Tab moves through the controls themselves.
    expect(cueButton('end')?.tabIndex).toBe(-1);
  });

  it('shows no cue while every control fits', async () => {
    await renderIsland({ count: 3, width: 400 });

    expect(scrollRange()).toBe(0);
    expect(shownCues()).toEqual([]);
  });

  it('cues the end again when controls are added past a row scrolled to its end', async () => {
    const { rerender } = await renderIsland({ count: 12, width: 160 });
    await act(async () => {
      scroller().scrollLeft = scrollRange();
      await nextFrame();
    });
    expect(shownCues()).toEqual(['start']);

    // The island is already at its widest: only the row's content changes size.
    await rerender(14);

    expect(shownCues()).toEqual(['start', 'end']);
  });

  it('scrolls all the way to the end from its cue, which hands over to the start cue', async () => {
    await renderIsland({ count: 12, width: 160 });

    await userEvent.click(cueButton('end')!);

    await expect.poll(() => scroller().scrollLeft).toBe(scrollRange());
    await expect.poll(shownCues).toEqual(['start']);

    await userEvent.click(cueButton('start')!);

    await expect.poll(() => scroller().scrollLeft).toBe(0);
    await expect.poll(shownCues).toEqual(['end']);
  });

  it('jumps without animating when the app reduces motion', async () => {
    document.documentElement.dataset.reduceMotion = 'true';
    await renderIsland({ count: 12, width: 160 });

    act(() => cueButton('end')!.click());

    // Read in the same task as the click: a smooth scroll would still be on its way.
    expect(scroller().scrollLeft).toBe(scrollRange());
  });

  it('cues both ends while scrolled part way', async () => {
    await renderIsland({ count: 12, width: 160 });

    await act(async () => {
      scroller().scrollLeft = scrollRange() / 2;
      await nextFrame();
    });

    expect(shownCues()).toEqual(['start', 'end']);
  });

  it.each([
    ['past the island', 8, 'hidden'],
    ['inside the island but under the end cue', 6, 'covered'],
  ] as const)('brings a focused control %s into view, clear of the cues', async (_where, index, precondition) => {
    await renderIsland({ count: 12, width: 160 });
    const target = control(index);
    const before = target.getBoundingClientRect();
    if (precondition === 'hidden') {
      expect(before.left).toBeGreaterThan(island().getBoundingClientRect().right);
    } else {
      expect(before.right).toBeLessThanOrEqual(island().getBoundingClientRect().right);
      expect(before.right).toBeGreaterThan(cue('end')!.getBoundingClientRect().left);
    }

    await act(async () => {
      target.focus();
      await nextFrame();
    });

    const targetRect = target.getBoundingClientRect();
    const islandRect = island().getBoundingClientRect();
    expect(targetRect.left).toBeGreaterThanOrEqual(cue('start')?.getBoundingClientRect().right ?? islandRect.left);
    expect(targetRect.right).toBeLessThanOrEqual(cue('end')?.getBoundingClientRect().left ?? islandRect.right);
  });

  it('scrolls toward the inline end of a right-to-left row', async () => {
    await renderIsland({ count: 12, dir: 'rtl', width: 160 });
    expect(shownCues()).toEqual(['end']);
    // The hidden controls, and the cue pointing at them, sit on the left.
    expect(cue('end')!.getBoundingClientRect().left).toBe(island().getBoundingClientRect().left + 1);

    await userEvent.click(cueButton('end')!);

    await expect.poll(() => scroller().scrollLeft).toBe(-scrollRange());
    await expect.poll(shownCues).toEqual(['start']);
  });
  it('cues the end when nested text grows past a control that keeps its size', async () => {
    // A fixed-width counter whose text overflows it: no control resizes, only the row's scrollable width grows.
    const counter = (text: string) => (
      <>
        {controls(4)}
        <Box flexShrink={0} w="6" whiteSpace="nowrap">
          <span>{text}</span>
        </Box>
      </>
    );
    const { rerender } = await mountIsland(counter('9'), { width: 160 });
    expect(shownCues()).toEqual([]);

    await rerender(counter('9 of 12 controls in a much longer readout'));

    expect(shownCues()).toEqual(['end']);
  });

  it('scrolls the row from a wheel over the row or either cue', async () => {
    await renderIsland({ count: 12, width: 160 });

    await act(async () => {
      scroller().dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 40 }));
      await nextFrame();
    });

    expect(scroller().scrollLeft).toBe(40);
    expect(shownCues()).toEqual(['start', 'end']);

    // The cues take the pointer over the row's ends, yet a wheel or swipe over one still scrolls the row.
    await act(async () => {
      cue('end')!.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 40 }));
      await nextFrame();
    });
    expect(scroller().scrollLeft).toBe(80);
    await act(async () => {
      cueButton('start')!.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaX: -30 }));
      await nextFrame();
    });
    expect(scroller().scrollLeft).toBe(50);
  });

  it('takes the pointer over the whole cue, above controls that raise their own z-index', async () => {
    const pressed = vi.fn();
    const hovered = vi.fn();
    await mountIsland(controls(12, { onClick: pressed, onPointerEnter: hovered, position: 'relative', zIndex: 1 }), {
      width: 160,
    });
    const button = cueButton('end')!.getBoundingClientRect();
    const fade = cue('end')!.getBoundingClientRect();
    const fadeX = fade.left + 4;
    const centerY = fade.top + fade.height / 2;
    // A control does sit under the fade's open side, half hidden.
    expect(control(5).getBoundingClientRect().right).toBeGreaterThan(fadeX);

    expect(
      cueButton('end')!.contains(
        document.elementFromPoint(button.left + button.width / 2, button.top + button.height / 2)
      )
    ).toBe(true);
    expect(document.elementFromPoint(fadeX, centerY)).toBe(cue('end'));

    await userEvent.hover(cue('end')!, { position: { x: 4, y: fade.height / 2 } });
    await userEvent.click(cue('end')!, { position: { x: 4, y: fade.height / 2 } });

    expect(pressed).not.toHaveBeenCalled();
    expect(hovered).not.toHaveBeenCalled();
    expect(scroller().scrollLeft).toBe(0);
  });
});
