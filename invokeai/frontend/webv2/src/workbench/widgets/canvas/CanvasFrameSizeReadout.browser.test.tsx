/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { Rect } from '@workbench/canvas-engine/api';
import type { ComponentProps } from 'react';

import { Box, ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/models/core/architectureCapabilities.testing';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

interface HarnessProject {
  canvas: { document: { bbox: Rect } };
  values: Record<string, Record<string, unknown>>;
}

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

type ReadoutEngine = ComponentProps<typeof CanvasFrameSizeReadout>['engine'];

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
  it('shows the render size each Render at mode generates at', async () => {
    // Optimal for model: a 1:2 frame grows toward SD 1.5's 512×512 area on its 8px grid. 720 tall needs 360 wide
    // (259,200 px, short of 262,144); 728 tall needs 364, which snaps to 368 (267,904 px).
    await mount(createProject({ canvas: { scaleMethod: 'auto' }, frame: { height: 512, width: 256 } }));
    expect(rows()).toEqual([
      ['Frame', '256×512'],
      ['Render', '368×728'],
    ]);

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

    await act(() => {
      harness.project.setSnapshot(
        createProject({ canvas: { scaleMethod: 'none' }, frame: { height: 512, width: 256 } })
      );
    });
    expect(rows()).toEqual([['Frame', '256×512']]);
  });

  it('follows a committed frame change and a Frame tool drag before it commits', async () => {
    const fake = await mount(createProject({ frame: { height: 768, width: 1024 } }));

    await act(() => {
      harness.project.setSnapshot(createProject({ frame: { height: 640, width: 896 } }));
    });
    expect(rows()).toEqual([['Frame', '896×640']]);

    await act(() => fake.setPreview({ height: 480.6, width: 640.4, x: 10, y: 10 }));
    expect(rows()).toEqual([
      ['Frame', '640×481'],
      ['Render', '640×480'],
    ]);

    await act(() => fake.setPreview(null));
    expect(rows()).toEqual([['Frame', '896×640']]);
  });
});
