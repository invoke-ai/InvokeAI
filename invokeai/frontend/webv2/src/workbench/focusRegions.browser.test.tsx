import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('./settings/store', () => ({
  useWorkbenchPreferenceSelector: (selector: (preferences: { showFocusRegionHighlight: boolean }) => unknown) =>
    selector({ showFocusRegionHighlight: true }),
}));

import {
  FocusRegionProvider,
  useFloatingWindowFocus,
  useFocusRegionProps,
  useWorkbenchFocus,
  type WorkbenchFocusController,
} from './focusRegions';
import { createTestFocusController } from './focusRegions.testing';

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
          <FocusRegionProvider controller={createTestFocusController()}>
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
type FocusRegion = WorkbenchFocusController['focusRegion'];

const OpeningHarness = ({ onOpen }: { onOpen: (open: () => void, focusRegion: FocusRegion) => void }) => {
  const [isShown, setIsShown] = useState(false);
  const { focusRegion } = useWorkbenchFocus();
  const open = () => {
    window.setTimeout(() => setIsShown(true), 50);
    focusRegion('center', 'workflow');
  };

  return (
    <>
      <Box data-testid="left" {...useFocusRegionProps('left')}>
        <button type="button" onClick={() => onOpen(open, focusRegion)}>
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

describe('focusRegion', () => {
  const renderHarness = async (onOpen: (open: () => void, focusRegion: FocusRegion) => void = (open) => open()) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={createTestFocusController()}>
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

  it('moves focus into a region as it stands when no widget is named', async () => {
    // The editor never opens here, so a move waiting on a widget would leave focus on the opener.
    const { center, opener } = await renderHarness((_open, focusRegion) => focusRegion('center'));
    opener.focus();

    await act(() => opener.click());
    await frames(3);

    expect(document.activeElement).toBe(center);
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
    const { center, opener } = await renderHarness((open, focusRegion) => {
      open();
      focusRegion('left', 'gallery');
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

/** A center view whose control opens another view in its place; the replaced view stays mounted but hidden. */
const ReplacingViewHarness = () => {
  const [isReplaced, setIsReplaced] = useState(false);
  const { focusRegion } = useWorkbenchFocus();
  const open = () => {
    setIsReplaced(true);
    focusRegion('center', 'preview');
  };

  return (
    <Box data-testid="center" {...useFocusRegionProps('center')}>
      <Box display={isReplaced ? 'none' : undefined}>
        <button type="button" onClick={open}>
          Open in Preview
        </button>
      </Box>
      {isReplaced ? <div data-hotkey-widget-type-id="preview">Preview</div> : null}
    </Box>
  );
};

describe('focusRegion from inside the view it replaces', () => {
  it('catches the focus the hidden opener drops instead of leaving it on the document', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={createTestFocusController()}>
            <ReplacingViewHarness />
          </FocusRegionProvider>
        </ChakraProvider>
      );
    });
    const center = host.querySelector<HTMLElement>('[data-testid="center"]')!;
    const opener = center.querySelector('button')!;
    opener.focus();

    await act(() => opener.click());
    await frames(6);

    expect(document.activeElement).toBe(center);
  });
});

/** A side region as the shell renders it: the panel shown before stays mounted but hidden beside the current one. */
const KeptPanelsHarness = () => {
  const { focusRegion } = useWorkbenchFocus();
  const hiddenPanel = useFocusRegionProps('right');
  const shownPanel = useFocusRegionProps('right');

  return (
    <>
      <button type="button" onClick={() => focusRegion('right', 'layers')}>
        Open layers
      </button>
      <Box data-testid="hidden-panel" display="none" {...hiddenPanel}>
        <div data-hotkey-widget-type-id="gallery">Gallery</div>
      </Box>
      <Box data-testid="shown-panel" {...shownPanel}>
        <div data-hotkey-widget-type-id="layers">Layers</div>
      </Box>
    </>
  );
};

describe('focusRegion with kept-alive panels', () => {
  it('moves focus into the panel on screen, not a hidden one kept mounted in the same region', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={createTestFocusController()}>
            <KeptPanelsHarness />
          </FocusRegionProvider>
        </ChakraProvider>
      );
    });
    const opener = host.querySelector('button')!;
    opener.focus();

    await act(() => opener.click());
    await frames(3);

    expect(document.activeElement).toBe(host.querySelector('[data-testid="shown-panel"]'));
  });
});

/** A docked region beside a floating window that shows a little later, as a lazily loaded window does. */
const WindowHarness = ({ isWindowShown, windowProjectId }: { isWindowShown: boolean; windowProjectId: string }) => (
  <>
    <Box data-testid="region" {...useFocusRegionProps('left')}>
      <button type="button">Region control</button>
    </Box>
    {isWindowShown ? <WindowStub projectId={windowProjectId} /> : null}
  </>
);

const WindowStub = ({ projectId }: { projectId: string }) => {
  const { activate, isHighlighted } = useFloatingWindowFocus('map', projectId);
  const handleFocus = useCallback(() => activate(), [activate]);
  const handlePointerDown = useCallback(() => activate({ byPointer: true }), [activate]);

  return (
    <Box
      data-floating-window="map"
      data-highlighted={isHighlighted}
      data-testid="window"
      onFocusCapture={handleFocus}
      onPointerDownCapture={handlePointerDown}
    >
      <button type="button">Window control</button>
    </Box>
  );
};

describe('workbench focus across regions and floating windows', () => {
  let projectId = 'project-1';
  let isWindowFloating = true;
  let controller: WorkbenchFocusController;

  const renderWindowHarness = async (isWindowShown = true) => {
    projectId = 'project-1';
    isWindowFloating = true;
    controller = createTestFocusController({ getProjectId: () => projectId, isFloating: () => isWindowFloating });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await rerender(isWindowShown);
  };
  const rerender = (isWindowShown: boolean, windowProjectId = 'project-1') =>
    act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={controller}>
            <WindowHarness isWindowShown={isWindowShown} windowProjectId={windowProjectId} />
          </FocusRegionProvider>
        </ChakraProvider>
      );
    });
  const region = () => host!.querySelector<HTMLElement>('[data-testid="region"]')!;
  const floatingWindow = () => host!.querySelector<HTMLElement>('[data-testid="window"]');
  const highlighted = () => ({
    region: region().getAttribute('data-highlighted'),
    window: floatingWindow()?.getAttribute('data-highlighted'),
  });

  it('gives the outline to whichever of a region or a window was last pressed or focused, never to both', async () => {
    await renderWindowHarness();

    await act(() => region().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    expect(highlighted()).toEqual({ region: 'true', window: 'false' });

    await act(() => floatingWindow()!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    expect(highlighted()).toEqual({ region: 'false', window: 'true' });
    expect(controller.getTarget()).toEqual({ instanceId: 'map', kind: 'floating' });

    // Keyboard focus moves it just as a press does.
    await act(() => region().querySelector('button')!.focus());
    expect(highlighted()).toEqual({ region: 'true', window: 'false' });
    await act(() => floatingWindow()!.querySelector('button')!.focus());
    expect(highlighted()).toEqual({ region: 'false', window: 'true' });
  });

  it('moves keyboard focus into a window once it shows, which activates it', async () => {
    await renderWindowHarness(false);
    const opener = region().querySelector('button')!;
    opener.focus();

    controller.focusFloating('map');
    await frames(2);
    await rerender(true);
    await frames(3);

    expect(floatingWindow()!.contains(document.activeElement)).toBe(true);
    expect(highlighted()).toEqual({ region: 'false', window: 'true' });
  });

  it('refuses activation from a window of a project that has left the screen', async () => {
    await renderWindowHarness();
    await act(() => region().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    projectId = 'project-2';
    // The stale window still renders for the project it was mounted under.
    await rerender(true, 'project-1');
    await act(() => region().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));

    await act(() => floatingWindow()!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));

    expect(controller.getTarget()).toEqual({ kind: 'region', region: 'left' });
    expect(controller.activate({ instanceId: 'map', kind: 'floating' }, { projectId: 'project-1' })).toBe('refused');
  });

  it('forgets the target and abandons a pending focus move when the project changes', async () => {
    await renderWindowHarness(false);
    await act(() => region().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    const opener = region().querySelector('button')!;
    opener.focus();
    controller.focusFloating('map');

    // The project switches before the window ever shows. Returning to it must not resurrect the old focus.
    projectId = 'project-2';
    expect(controller.getTarget()).toBeNull();
    await act(() => controller.clear());
    projectId = 'project-1';
    await rerender(true);
    await frames(4);

    expect(controller.getTarget()).toBeNull();
    expect(highlighted()).toEqual({ region: 'false', window: 'false' });
    expect(document.activeElement).toBe(opener);
  });

  it('stops treating a window as the target once it no longer floats', async () => {
    await renderWindowHarness();
    await act(() => floatingWindow()!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    expect(controller.getTarget()).toEqual({ instanceId: 'map', kind: 'floating' });

    // Docked or removed, with nothing else focused since.
    isWindowFloating = false;

    expect(controller.getTarget()).toBeNull();
  });

  it('abandons a pending focus move when the user presses somewhere else first', async () => {
    await renderWindowHarness(false);
    const opener = region().querySelector('button')!;
    opener.focus();
    controller.focusFloating('map');

    // Pressing content that takes no focus of its own: focus falls to the body, where a move would reclaim it.
    await act(() => region().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    opener.blur();
    await rerender(true);
    await frames(4);

    expect(floatingWindow()!.contains(document.activeElement)).toBe(false);
    expect(controller.getTarget()).toEqual({ kind: 'region', region: 'left' });
  });

  it('keeps a pending focus move when focus, not a press, passes through another region', async () => {
    await renderWindowHarness(false);
    const opener = region().querySelector('button')!;
    opener.focus();

    controller.focusFloating('map');
    // A closing menu handing focus back to its trigger arrives as focus, and must not cancel the move.
    await act(() => {
      opener.blur();
      opener.focus();
    });
    await rerender(true);
    await frames(4);

    expect(floatingWindow()!.contains(document.activeElement)).toBe(true);
  });
});
