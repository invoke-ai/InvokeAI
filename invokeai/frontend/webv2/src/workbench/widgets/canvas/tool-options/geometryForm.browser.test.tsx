import type { CanvasInteractionState, Rect } from '@workbench/canvas-engine/api';
import type { ToolFormProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { bboxForm, transformForm } from './geometryForm';

const frame = vi.hoisted(() => ({
  bbox: { height: 1024, width: 1024, x: 0, y: 0 },
  patchSettings: vi.fn(),
  sourceId: 'canvas',
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({ canvas: { document: { bbox: frame.bbox } }, invocation: { sourceId: frame.sourceId } }),
  useOptionalWorkbenchCommands: () => null,
  useWorkbenchCommands: () => ({ generation: { patchSettings: frame.patchSettings } }),
}));
// Saved ratios are account preferences, synced through the backend's client state; this keeps them in memory.
vi.mock('@workbench/projects/api', async (importOriginal) => {
  const clientState = new Map<string, string>();
  return {
    ...(await importOriginal<object>()),
    getClientStateValue: (key: string) => Promise.resolve(clientState.get(key) ?? null),
    setClientStateValue: (key: string, value: string) => Promise.resolve(void clientState.set(key, value)),
  };
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

/** The engine's interaction store as the footer reads it: a value per key and a listener set per key. */
const createEngine = () => {
  const state: Partial<CanvasInteractionState> = {
    hasFloatingSelection: false,
    pendingTransformLayerId: null,
    transformSession: null,
  };
  const listeners = new Map<string, Set<() => void>>();
  const engine = {
    interaction: {
      get: (key: keyof CanvasInteractionState) => state[key],
      subscribe: (key: string, listener: () => void) => {
        const keyListeners = listeners.get(key) ?? new Set();
        keyListeners.add(listener);
        listeners.set(key, keyListeners);
        return () => keyListeners.delete(listener);
      },
    },
    layers: { applyTransform: vi.fn(), cancelTransform: vi.fn() },
  };
  const setPending = (layerId: string | null) =>
    act(() => {
      state.pendingTransformLayerId = layerId;
      listeners.get('pendingTransformLayerId')?.forEach((listener) => listener());
    });
  return { engine: engine as unknown as ToolFormProps['engine'], setPending };
};

/** Renders the footer `width` px wide. */
const renderFooter = async (engine: ToolFormProps['engine'], width = 320) => {
  const Footer = transformForm.footer!;
  host = document.createElement('div');
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Footer engine={engine} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

describe('Transform footer', () => {
  it('says while a transform waits for its stroke to save, in a polite live region, and clears after', async () => {
    const { engine, setPending } = createEngine();
    await renderFooter(engine);
    const status = page.getByRole('status');
    await expect.element(status).toBeInTheDocument();
    await expect.element(status).toHaveTextContent('');

    await setPending('layer-1');
    await expect.element(status).toHaveTextContent('Waiting for the stroke to save…');
    await expect.element(page.getByRole('button', { name: 'Apply' })).toBeDisabled();

    await setPending(null);
    await expect.element(status).toHaveTextContent('');
  });

  it('shows the whole status in a tooltip where a narrow pane truncates it', async () => {
    const { engine, setPending } = createEngine();
    await renderFooter(engine, 180);
    await setPending('layer-1');
    const status = page.getByRole('status');
    const line = status.element() as HTMLElement;
    expect(line.scrollWidth).toBeGreaterThan(line.clientWidth);

    await status.getByText('Waiting for the stroke to save…').hover();

    await expect.element(page.getByRole('tooltip')).toHaveTextContent('Waiting for the stroke to save…');
  });
});

/** The Frame tool's engine: its bbox options and grid as the form reads and writes them, and the frame commit. */
const createFrameEngine = (bboxOptions = { aspectLocked: false, aspectRatio: 1 }) => {
  const state: Partial<CanvasInteractionState> = { bboxGrid: 8, bboxOptions };
  const listeners = new Map<string, Set<() => void>>();
  const commits: Rect[] = [];
  const engine = {
    interaction: {
      get: (key: keyof CanvasInteractionState) => state[key],
      set: (key: keyof CanvasInteractionState, value: never) => {
        state[key] = value;
        listeners.get(key)?.forEach((listener) => listener());
      },
      subscribe: (key: string, listener: () => void) => {
        const keyListeners = listeners.get(key) ?? new Set();
        keyListeners.add(listener);
        listeners.set(key, keyListeners);
        return () => keyListeners.delete(listener);
      },
    },
    layers: { commitStructural: (_label: string, change: { bbox: Rect }) => commits.push(change.bbox) },
  };
  return { commits, engine: engine as unknown as ToolFormProps['engine'], state };
};

describe('Frame aspect ratio', () => {
  const FrameSize = bboxForm.groups.find((group) => group.id === 'frame-size')!.body;

  const renderFrame = async (bboxOptions?: { aspectLocked: boolean; aspectRatio: number }) => {
    accountLifecycle.activate('frame-aspect-ratio');
    await patchWorkbenchPreferences({
      aspectRatioPresets: [{ height: 1, id: 'cinema', label: 'Cinema', width: 2.39 }],
    });
    const frameEngine = createFrameEngine(bboxOptions);
    host = document.createElement('div');
    host.style.width = '320px';
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root!.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <FrameSize engine={frameEngine.engine} />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
    await userEvent.click(page.getByRole('button', { name: /^Aspect ratio/ }));
    return frameEngine;
  };

  afterEach(() => {
    frame.bbox = { height: 1024, width: 1024, x: 0, y: 0 };
    frame.patchSettings.mockReset();
    accountLifecycle.invalidate();
  });

  // 1024×1024 holds 1,048,576 pixels. At 2.39:1 that is 1583.1×662.4, snapped to the 8px grid as 1584×664 (what
  // Generate sets), centred on the old frame's centre (512, 512).
  const CINEMA_FRAME = { height: 664, width: 1584, x: -280, y: 180 };

  it('keeps the pixel count when a saved ratio is picked, locking it on the frame and on Generate', async () => {
    const { commits, state } = await renderFrame();

    // The size column shows the size the pick sets.
    await userEvent.click(page.getByRole('button', { name: 'Cinema, 2.39:1, 1584×664' }));

    expect(commits).toEqual([CINEMA_FRAME]);
    expect(state.bboxOptions).toEqual({ aspectLocked: true, aspectRatio: 2.39 });
    expect(frame.patchSettings).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: true,
      aspectRatioValue: 2.39,
    });
    await expect
      .element(page.getByRole('button', { name: /^Aspect ratio/ }))
      .toHaveAccessibleName('Aspect ratio Cinema 2.39:1');
  });

  it('resizes the same way when the canvas does not drive Generate, leaving Generate alone', async () => {
    frame.sourceId = 'generate';
    try {
      const { commits, state } = await renderFrame();

      await userEvent.click(page.getByRole('button', { name: 'Cinema, 2.39:1, 1584×664' }));

      expect(commits).toEqual([CINEMA_FRAME]);
      expect(state.bboxOptions).toEqual({ aspectLocked: true, aspectRatio: 2.39 });
      expect(frame.patchSettings).not.toHaveBeenCalled();
    } finally {
      frame.sourceId = 'canvas';
    }
  });

  it('keeps the pixel count for a built-in ratio too', async () => {
    const { commits, state } = await renderFrame();

    await userEvent.click(page.getByRole('radio', { exact: true, name: '16:9' }));

    // 1024² at 16:9 is 1365.3×768, snapped to 1368×768, centred on (512, 512).
    expect(commits).toEqual([{ height: 768, width: 1368, x: -172, y: 128 }]);
    expect(state.bboxOptions).toEqual({ aspectLocked: true, aspectRatio: 16 / 9 });
    expect(frame.patchSettings).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 16 / 9,
    });
  });

  it('keeps the ratio and pixel count of a frame larger than any one generation', async () => {
    frame.bbox = { height: 6144, width: 6144, x: 0, y: 0 };
    const { commits } = await renderFrame();

    await userEvent.click(page.getByRole('radio', { exact: true, name: '16:9' }));

    // 6144² at 16:9 is exactly 8192×4608: no 4096 cap on the frame, centred on (3072, 3072).
    expect(commits).toEqual([{ height: 4608, width: 8192, x: -1024, y: 768 }]);
  });

  it('keeps a tiny frame at least one grid step on each side', async () => {
    frame.bbox = { height: 8, width: 8, x: 0, y: 0 };
    const { commits } = await renderFrame();

    await userEvent.click(page.getByRole('radio', { exact: true, name: '9:21' }));

    // 8² at 9:21 is 5.2×12.2; scaled up so the narrow side is 8, then 18.7 snaps to 16. Centred on (4, 4).
    expect(commits).toEqual([{ height: 16, width: 8, x: 0, y: -4 }]);
  });

  it('unlocks with Free, leaving the frame as it is', async () => {
    const { commits, state } = await renderFrame({ aspectLocked: true, aspectRatio: 16 / 9 });

    await userEvent.click(page.getByRole('radio', { exact: true, name: 'Free' }));

    expect(commits).toEqual([]);
    expect(state.bboxOptions).toEqual({ aspectLocked: false, aspectRatio: 1 });
    expect(frame.patchSettings).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: 1,
    });
  });

  it('unlocks from the lock button, leaving the frame as it is', async () => {
    const { commits, state } = await renderFrame({ aspectLocked: true, aspectRatio: 16 / 9 });
    await userEvent.keyboard('{Escape}');

    await userEvent.click(page.getByRole('button', { name: 'Unlock aspect ratio' }));

    expect(commits).toEqual([]);
    expect(state.bboxOptions).toEqual({ aspectLocked: false, aspectRatio: 1 });
  });

  it('locks the ratio the frame has without resizing it', async () => {
    const { commits, state } = await renderFrame();
    await userEvent.keyboard('{Escape}');

    await userEvent.click(page.getByRole('button', { name: 'Lock aspect ratio' }));

    expect(commits).toEqual([]);
    expect(state.bboxOptions).toEqual({ aspectLocked: true, aspectRatio: 1 });
  });
});
