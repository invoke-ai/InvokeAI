/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { Rect } from '@workbench/canvas-engine/api';
import type { ComponentProps } from 'react';

import { Box, ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

interface HarnessProject {
  canvas: { document: { bbox: Rect } };
  values: Record<string, Record<string, unknown>>;
}

// The active project as a store, so a committed frame or setting re-renders its readers as the Workbench would.
const harness = vi.hoisted(() => ({
  project: null as unknown as {
    getSnapshot: () => HarnessProject;
    setSnapshot: (next: HarnessProject) => void;
    subscribe: (listener: () => void) => () => void;
  },
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: <Selected,>(
    selector: (project: HarnessProject) => Selected,
    isEqual?: (a: Selected, b: Selected) => boolean
  ) => useExternalStoreSelector(harness.project.subscribe, harness.project.getSnapshot, selector, isEqual),
}));
vi.mock('@workbench/widgetState', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProjectWidgetValues: (project: HarnessProject, widgetId: string) => project.values[widgetId] ?? {},
}));

import { CanvasFrameSizeReadout } from './CanvasFrameSizeReadout';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SD1_MODEL = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' } as const;
const WAN_TI2V_MODEL = { base: 'wan', key: 'wan-ti2v', name: 'Wan 2.2 TI2V-5B', type: 'main', variant: 'ti2v_5b' };

type ReadoutEngine = ComponentProps<typeof CanvasFrameSizeReadout>['engine'];

/** The engine's interaction store as the readout reads it: only the Frame tool's drag preview. */
const createFakeEngine = () => {
  let preview: Rect | null = null;
  const listeners = new Set<() => void>();
  const engine = {
    interaction: {
      get: (key: string) => (key === 'bboxPreview' ? preview : undefined),
      subscribe: (key: string, listener: () => void) => {
        if (key !== 'bboxPreview') {
          return () => undefined;
        }
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
  };
  return {
    engine: engine as unknown as ReadoutEngine,
    setPreview: (next: Rect | null) => {
      preview = next;
      listeners.forEach((listener) => listener());
    },
  };
};

const createProject = ({
  canvas = {},
  frame,
  model = SD1_MODEL,
}: {
  canvas?: Record<string, unknown>;
  frame: { height: number; width: number };
  model?: Record<string, unknown> | null;
}): HarnessProject => ({
  canvas: { document: { bbox: { ...frame, x: 0, y: 0 } } },
  values: { canvas, generate: { ...frame, model, pidMode: 'off' } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let beneathClicks = 0;

/** The readout over a canvas-sized stand-in for the surface, which counts the clicks that reach it. */
const mount = async (project: HarnessProject, { height = 480, width = 720 } = {}) => {
  harness.project = createExternalStoreCore(project);
  const fake = createFakeEngine();
  applyThemeToRoot('classic');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  beneathClicks = 0;
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Box bg="bg.inset" h={`${height}px`} overflow="hidden" position="relative" w={`${width}px`}>
            <button
              aria-label="Canvas surface"
              style={{ inset: 0, position: 'absolute' }}
              type="button"
              onClick={() => {
                beneathClicks += 1;
              }}
            />
            <CanvasFrameSizeReadout engine={fake.engine} />
          </Box>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
  return fake;
};

const readout = () => page.getByRole('group', { name: 'Frame and render size' });
/** Each term and its value, as assistive tech reads the list. */
const rows = () =>
  [...readout().element().querySelectorAll('dt')].map((term) => [
    term.textContent,
    term.nextElementSibling?.textContent,
  ]);

beforeEach(() => {
  setArchitectureCapabilities(architectureCapabilitiesFixture);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  resetArchitectureCapabilities();
});

describe('CanvasFrameSizeReadout', () => {
  it('shows only the frame size when the frame renders at its own size', async () => {
    // 1024×768 is on SD 1.5's 8px grid and already past its 512×512 optimum, so Optimal for model leaves it alone.
    await mount(createProject({ frame: { height: 768, width: 1024 } }));

    await expect.element(readout()).toBeVisible();
    expect(rows()).toEqual([['Frame', '1024×768']]);
  });

  it('shows the render size each Render at mode generates at', async () => {
    // Optimal for model: a 1:2 frame grows toward SD 1.5's 512×512 area on its 8px grid. 720 tall needs 360 wide
    // (259,200 px, short of 262,144); 728 tall needs 364, which snaps to 368 (267,904 px).
    await mount(createProject({ canvas: { scaleMethod: 'auto' }, frame: { height: 512, width: 256 } }));
    expect(rows()).toEqual([
      ['Frame', '256×512'],
      ['Render', '368×728'],
    ]);

    // Custom renders at the chosen size.
    await act(() => {
      harness.project.setSnapshot(
        createProject({
          canvas: { scaleMethod: 'manual', scaledHeight: 1152, scaledWidth: 1536 },
          frame: { height: 512, width: 256 },
        })
      );
    });
    expect(rows()).toEqual([
      ['Frame', '256×512'],
      ['Render', '1536×1152'],
    ]);

    // Frame size renders at the frame itself, so nothing differs.
    await act(() => {
      harness.project.setSnapshot(
        createProject({ canvas: { scaleMethod: 'none' }, frame: { height: 512, width: 256 } })
      );
    });
    expect(rows()).toEqual([['Frame', '256×512']]);
  });

  it('shows the size a frame off the model grid renders at', async () => {
    // Wan 2.2 TI2V-5B generates on a 32px grid: 720 rounds to 736 (22.5 × 32 rounds up to 23 × 32).
    await mount(
      createProject({ canvas: { scaleMethod: 'none' }, frame: { height: 720, width: 1280 }, model: WAN_TI2V_MODEL })
    );

    expect(rows()).toEqual([
      ['Frame', '1280×720'],
      ['Render', '1280×736'],
    ]);
  });

  it('follows a committed frame change and a Frame tool drag before it commits', async () => {
    const fake = await mount(createProject({ frame: { height: 768, width: 1024 } }));

    await act(() => {
      harness.project.setSnapshot(createProject({ frame: { height: 640, width: 896 } }));
    });
    expect(rows()).toEqual([['Frame', '896×640']]);

    // A free (Alt) drag reshapes the frame off whole pixels; the drag commits it rounded, and renders it on the grid.
    await act(() => fake.setPreview({ height: 480.6, width: 640.4, x: 10, y: 10 }));
    expect(rows()).toEqual([
      ['Frame', '640×481'],
      ['Render', '640×480'],
    ]);

    // The drag ended without a commit: the committed frame shows again.
    await act(() => fake.setPreview(null));
    expect(rows()).toEqual([['Frame', '896×640']]);
  });

  it('shows the frame alone until a model is selected', async () => {
    await mount(createProject({ canvas: { scaleMethod: 'auto' }, frame: { height: 128, width: 128 }, model: null }));

    expect(rows()).toEqual([['Frame', '128×128']]);
  });

  it('lets a click at its position through to the canvas beneath', async () => {
    await mount(createProject({ canvas: { scaleMethod: 'auto' }, frame: { height: 384, width: 384 } }));
    const box = readout().element().getBoundingClientRect();
    const surface = page.getByRole('button', { name: 'Canvas surface' });
    const surfaceBox = surface.element().getBoundingClientRect();

    // A real pointer click at the readout's centre; Playwright refuses it if the readout would intercept it.
    await userEvent.click(surface, {
      position: { x: box.left + box.width / 2 - surfaceBox.left, y: box.top + box.height / 2 - surfaceBox.top },
    });
    expect(beneathClicks).toBe(1);
  });

  it('sits in the top-right corner and steps aside when the canvas is too small', async () => {
    await mount(createProject({ frame: { height: 768, width: 1024 } }));
    const box = readout().element().getBoundingClientRect();
    const canvas = host!.firstElementChild!.getBoundingClientRect();
    // 8px (spacing 2) in from the top and right edges outside the center region's chrome.
    expect(Math.round(canvas.right - box.right)).toBe(8);
    expect(Math.round(box.top - canvas.top)).toBe(8);
    await act(() => root?.unmount());
    host?.remove();

    await mount(createProject({ frame: { height: 768, width: 1024 } }), { height: 480, width: 480 });
    // Hidden, so assistive tech skips it too.
    await expect.element(readout()).not.toBeInTheDocument();
    expect(getComputedStyle(host!.querySelector('[role="group"]')!).display).toBe('none');
  });
});
