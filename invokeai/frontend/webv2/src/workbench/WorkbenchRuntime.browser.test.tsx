/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { WorkbenchFocusController } from '@workbench/focusRegions';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act, useCallback } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * The workbench's focus provider against a real store: focus is forgotten, falling back to the center, and any focus
 * move still waiting is abandoned when the project on screen changes, the account changes, the window closes, or the
 * workbench unmounts. Only the context plumbing is replaced.
 */

const runtimeMocks = vi.hoisted(() => ({ store: null as unknown as WorkbenchInternalStore }));

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchInternalStore: () => runtimeMocks.store,
  // No canvas engine is live here, so a frame resize the runtime asks for stays a plain write.
  useWorkbenchLiveCanvasEngines: () => ({ commitFrame: () => false }),
  useWorkbenchQueries: () => runtimeMocks.store.queries,
  useWorkbenchSubscription: () => runtimeMocks.store.subscribe,
}));

import {
  useFloatingWindowFocus,
  useFocusRegionProps,
  useWorkbenchFocus,
  useWorkbenchFocusTarget,
} from './focusRegions';
import { WorkbenchFocusProvider } from './WorkbenchRuntime';
import { createWorkbenchStore } from './workbenchStore';

/** What the controller names once it has forgotten what was focused. */
const CENTER = { kind: 'region', region: 'center' };

type FocusApi = Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion' | 'getTarget'>;

let api: FocusApi;

/** Hands the provider's focus API to the test when it mounts. */
const Probe = () => {
  const moves = useWorkbenchFocus();
  const getTarget = useWorkbenchFocusTarget();
  const publish = useCallback(
    (node: HTMLSpanElement | null) => {
      if (node) {
        api = { ...moves, getTarget };
      }
    },
    [getTarget, moves]
  );

  return <span ref={publish} />;
};

const WindowStub = () => {
  const { activate } = useFloatingWindowFocus('image-map', runtimeMocks.store.getSnapshot().activeProject.id);
  const handlePointerDown = useCallback(() => activate({ byPointer: true }), [activate]);

  return (
    <div data-floating-window="image-map" data-testid="window" tabIndex={-1} onPointerDownCapture={handlePointerDown}>
      Image Map
    </div>
  );
};

const Harness = ({ showWindow }: { showWindow: boolean }) => (
  <>
    <Probe />
    <div data-testid="center" {...useFocusRegionProps('center')} />
    <div data-testid="region" {...useFocusRegionProps('right')}>
      <button type="button">Region control</button>
    </div>
    {showWindow ? <WindowStub /> : null}
  </>
);

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = (showWindow: boolean) =>
  act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <WorkbenchFocusProvider>
          <Harness showWindow={showWindow} />
        </WorkbenchFocusProvider>
      </ChakraProvider>
    );
  });
const frames = (count: number) =>
  act(async () => {
    for (let index = 0; index < count; index += 1) {
      await new Promise((resolve) => {
        requestAnimationFrame(resolve);
      });
    }
  });
const press = (testId: string) =>
  act(() =>
    host!.querySelector(`[data-testid="${testId}"]`)!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
  );
const opener = () => host!.querySelector<HTMLButtonElement>('button')!;

beforeEach(() => {
  runtimeMocks.store = createWorkbenchStore();
  runtimeMocks.store.commands.widgets.float('image-map');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WorkbenchFocusProvider', () => {
  it('forgets focus and abandons a pending move when the project changes, even after returning to it', async () => {
    const { commands, getSnapshot } = runtimeMocks.store;
    const firstProjectId = getSnapshot().activeProject.id;
    await render(false);
    await press('region');
    opener().focus();
    api.focusFloating('image-map');
    expect(api.getTarget()).toEqual({ kind: 'region', region: 'right' });

    await act(() => commands.projects.create());
    expect(getSnapshot().activeProject.id).not.toBe(firstProjectId);
    expect(api.getTarget()).toEqual(CENTER);

    // Back on the first project, where the window the move was waiting for now shows.
    await act(() => commands.projects.switchTo(firstProjectId));
    await render(true);
    await frames(4);

    expect(api.getTarget()).toEqual(CENTER);
    expect(document.activeElement).toBe(opener());
    expect(host!.querySelector('[data-testid="region"]')?.getAttribute('data-highlighted')).toBe('false');
  });

  it('forgets focus and abandons a pending move when the account changes', async () => {
    await render(false);
    await press('region');
    opener().focus();
    api.focusFloating('image-map');

    await act(() => {
      accountLifecycle.invalidate();
    });
    await render(true);
    await frames(4);

    expect(api.getTarget()).toEqual(CENTER);
    expect(document.activeElement).toBe(opener());
  });

  it('stops naming a window once it is closed or docked', async () => {
    await render(true);
    await press('window');
    expect(api.getTarget()).toEqual({ instanceId: 'image-map', kind: 'floating' });

    const center = host!.querySelector('[data-testid="center"]')!;
    expect(center.getAttribute('data-highlighted')).toBe('false');

    await act(() => runtimeMocks.store.commands.widgets.closeFloating('image-map'));

    expect(api.getTarget()).toEqual(CENTER);
    // Nothing else re-renders the center here: its outline returns only because forgetting the window notifies.
    expect(center.getAttribute('data-highlighted')).toBe('true');
  });

  it('does not hand focus back to a window that floats again without being focused', async () => {
    const { commands } = runtimeMocks.store;
    await render(true);
    await press('window');

    // The focused window leaves — as a preset applied in the same project can make it — and a later change floats
    // the same instance again.
    await act(() => commands.widgets.dockFloating('image-map'));
    await act(() => commands.widgets.float('image-map'));

    expect(runtimeMocks.store.getSnapshot().activeProject.floatingWidgets?.['image-map']).toBeDefined();
    expect(api.getTarget()).toEqual(CENTER);
  });

  it('abandons a pending move when the workbench unmounts', async () => {
    await render(false);
    const move = api.focusFloating;
    move('image-map');

    await act(() => root?.render(null));
    // A window with the awaited id shows up in whatever mounts next.
    const impostor = document.createElement('div');
    impostor.dataset.floatingWindow = 'image-map';
    document.body.append(impostor);
    await frames(4);

    expect(document.activeElement).not.toBe(impostor);
    impostor.remove();
  });
});
