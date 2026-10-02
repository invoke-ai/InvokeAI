import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('./settings/store', () => ({
  useWorkbenchPreferenceSelector: (selector: (preferences: { showFocusRegionHighlight: boolean }) => unknown) =>
    selector({ showFocusRegionHighlight: true }),
}));

import { FocusRegionProvider, focusOpenedWidget, useFocusRegionProps } from './focusRegions';

const FocusableRegion = () => <Box data-testid="focus-region" h="20" {...useFocusRegionProps('center')} />;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('focus region highlight', () => {
  it('lays the highlight over the borders beside the region', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider>
            <FocusableRegion />
          </FocusRegionProvider>
        </ChakraProvider>
      );
    });

    const region = host.querySelector<HTMLElement>('[data-testid="focus-region"]');
    expect(region).not.toBeNull();

    await act(() => region?.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));

    expect(region?.getAttribute('data-highlighted')).toBe('true');
    const highlight = getComputedStyle(region!, '::after');
    expect([highlight.top, highlight.right, highlight.bottom, highlight.left]).toEqual(['0px', '-1px', '0px', '-1px']);
  });
});

/** A side panel whose button opens a widget in the center region, which mounts a little later as lazy widgets do. */
const OpeningHarness = ({ onOpen }: { onOpen: (open: () => void) => void }) => {
  const [isShown, setIsShown] = useState(false);
  const open = () => {
    window.setTimeout(() => setIsShown(true), 50);
    focusOpenedWidget('center', 'workflow');
  };

  return (
    <>
      <Box data-testid="left" {...useFocusRegionProps('left')}>
        <button type="button" onClick={() => onOpen(open)}>
          Open editor
        </button>
        <div data-hotkey-widget-type-id="gallery">Gallery</div>
      </Box>
      <Box data-testid="center" {...useFocusRegionProps('center')}>
        <button type="button">Center control</button>
        {isShown ? <div data-hotkey-widget-type-id="workflow">Workflow editor</div> : null}
      </Box>
    </>
  );
};

const frames = (count: number) =>
  act(async () => {
    for (let index = 0; index < count; index += 1) {
      await new Promise((resolve) => {
        requestAnimationFrame(resolve);
      });
    }
  });

describe('focusOpenedWidget', () => {
  const renderHarness = async (onOpen: (open: () => void) => void = (open) => open()) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider>
            <OpeningHarness onOpen={onOpen} />
          </FocusRegionProvider>
        </ChakraProvider>
      );
    });
    return {
      center: host.querySelector<HTMLElement>('[data-testid="center"]')!,
      opener: [...host.querySelectorAll('button')].find((button) => button.textContent === 'Open editor')!,
    };
  };

  it('moves focus and the highlight to the region once the opened widget shows', async () => {
    const { center, opener } = await renderHarness();
    opener.focus();

    await act(() => opener.click());
    await act(
      () =>
        new Promise((resolve) => {
          window.setTimeout(resolve, 80);
        })
    );
    await frames(4);

    expect(center.contains(document.activeElement)).toBe(true);
    expect(center.getAttribute('data-highlighted')).toBe('true');
  });

  it('keeps the move when a closing menu hands focus back to the control that opened it', async () => {
    const { center, opener } = await renderHarness();
    // A menu item's menu restores focus to its trigger as it closes, after the move has landed.
    center.addEventListener('focusin', () => window.setTimeout(() => opener.focus(), 0), { once: true });
    opener.focus();

    await act(() => opener.click());
    await act(
      () =>
        new Promise((resolve) => {
          window.setTimeout(resolve, 80);
        })
    );
    await frames(6);

    expect(center.contains(document.activeElement)).toBe(true);
  });

  it('follows the last widget a control opens when it opens several', async () => {
    const { center, opener } = await renderHarness((open) => {
      open();
      focusOpenedWidget('left', 'gallery');
    });
    opener.focus();

    await act(() => opener.click());
    await act(
      () =>
        new Promise((resolve) => {
          window.setTimeout(resolve, 80);
        })
    );
    await frames(4);

    expect(center.contains(document.activeElement)).toBe(false);
    expect(document.activeElement).toBe(opener);
  });

  it('leaves focus where it is when it is already inside that region', async () => {
    const { center, opener } = await renderHarness((open) => {
      centerControl().focus();
      open();
    });
    const centerControl = () =>
      [...center.querySelectorAll('button')].find((button) => button.textContent === 'Center control')!;

    await act(() => opener.click());
    await act(
      () =>
        new Promise((resolve) => {
          window.setTimeout(resolve, 80);
        })
    );
    await frames(4);

    expect(document.activeElement).toBe(centerControl());
  });
});
