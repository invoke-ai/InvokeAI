/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { CanvasRenderSize } from '@features/generation/react';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { GenerateDimensionFields } from '@features/generation/ui/GenerateDimensionFields';
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
  canvas: { document: { bbox: { height: number; width: number; x: number; y: number } } };
  values: Record<string, Record<string, unknown>>;
}

// The active project as a store, so persisted canvas values re-render their readers as the Workbench would.
const harness = vi.hoisted(() => ({
  invocationSourceId: 'canvas',
  project: null as unknown as {
    getSnapshot: () => HarnessProject;
    setSnapshot: (next: HarnessProject) => void;
    subscribe: (listener: () => void) => () => void;
  },
  slot: null as unknown,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: <Selected,>(
    selector: (project: HarnessProject) => Selected,
    isEqual?: (a: Selected, b: Selected) => boolean
  ) => useExternalStoreSelector(harness.project.subscribe, harness.project.getSnapshot, selector, isEqual),
  useWorkbenchCommands: () => ({
    widgets: {
      patchValues: (widgetId: string, patch: Record<string, unknown>) => {
        const project = harness.project.getSnapshot();
        harness.project.setSnapshot({
          ...project,
          values: { ...project.values, [widgetId]: { ...project.values[widgetId], ...patch } },
        });
      },
    },
  }),
}));
vi.mock('@workbench/widgetState', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProjectWidgetValues: (project: HarnessProject, widgetId: string) => project.values[widgetId] ?? {},
}));
vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useGenerationQueueInsights: (select: (insights: unknown) => unknown) =>
    select({ secondsPerRun: null, seedHistory: [] }),
  useGenerationUi: () => ({
    CanvasRenderSize: harness.slot,
    project: { invocationSourceId: harness.invocationSourceId },
    sectionPreferences: { sectionsOpen: { dimensions: true }, setSectionOpen: () => undefined },
  }),
}));

import { GenerateCanvasRenderSize } from './GenerateCanvasRenderSize';

harness.slot = GenerateCanvasRenderSize;

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SD1_MODEL = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' } as const;
const SDXL_MODEL = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' } as const;
const WAN_TI2V_MODEL = { base: 'wan', key: 'wan-ti2v', name: 'Wan 2.2 TI2V-5B', type: 'main', variant: 'ti2v_5b' };

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const createProject = ({
  canvas = {},
  frame,
  model,
}: {
  canvas?: Record<string, unknown>;
  frame: { height: number; width: number };
  model: Record<string, unknown>;
}): HarnessProject => ({
  canvas: { document: { bbox: { ...frame, x: 0, y: 0 } } },
  values: { canvas, generate: { ...frame, model, pidMode: 'off' } },
});

const mount = async (children: ReactNode) => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '350px';
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{children}</ChakraProvider>
      </I18nextProvider>
    );
  });
};

/** The Size section as the Generate form renders it, over a frame of `frame`. */
const renderSizeSection = async (project: HarnessProject) => {
  harness.project = createExternalStoreCore(project);
  const { height, width } = project.canvas.document.bbox;
  await mount(
    <GenerateDimensionFields
      draft={createExternalStoreCore({
        ...getDefaultGenerateSettings(SD1_MODEL),
        aspectRatioId: 'Free',
        aspectRatioIsLocked: false,
        height,
        width,
      })}
      projectId="project"
      selectedModel={SD1_MODEL}
      onCommit={vi.fn()}
    />
  );
};

const canvasValues = () => harness.project.getSnapshot().values.canvas;
/** The section header's size badge. */
const badgeText = () =>
  [...(host?.querySelectorAll('span') ?? [])].find(
    (element) => element.childElementCount === 0 && /^\d+x\d+/.test(element.textContent ?? '')
  )?.textContent;
const statusText = () =>
  [...(host?.querySelectorAll('p') ?? [])].find((element) => element.textContent?.includes('MP'))?.textContent;
const optimalSizeButton = () => page.getByRole('button', { name: 'Set optimal size' });
const renderAt = () => page.getByRole('combobox', { name: 'Render at' });

/** The dashed ghost in the size preview, in preview pixels. */
const ghostSize = () => {
  const ghost = [...(host?.querySelectorAll<HTMLElement>('div') ?? [])].find(
    (element) => getComputedStyle(element).borderStyle === 'dashed'
  );
  const box = ghost?.getBoundingClientRect();
  return box ? [Math.round(box.width), Math.round(box.height)] : null;
};

const chooseRenderAt = async (option: string) => {
  await userEvent.click(renderAt());
  await userEvent.click(page.getByRole('option', { name: option }));
};

beforeEach(() => {
  harness.invocationSourceId = 'canvas';
});

const unmount = async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
};

afterEach(async () => {
  await unmount();
  resetArchitectureCapabilities();
});

describe('GenerateCanvasRenderSize', () => {
  it('reports the size the graph will generate at, on the grid the model variant declares', async () => {
    harness.project = createExternalStoreCore(
      createProject({
        canvas: { scaleMethod: 'manual', scaledHeight: 720, scaledWidth: 1280 },
        frame: { height: 720, width: 1280 },
        model: WAN_TI2V_MODEL,
      })
    );
    let reported: CanvasRenderSize['size'] = null;
    await mount(
      <GenerateCanvasRenderSize frame={{ height: 720, width: 1280 }}>
        {({ size }) => {
          reported = size;
          return null;
        }}
      </GenerateCanvasRenderSize>
    );
    // Before the table the fallback grid is 8, and 720 sits on it.
    expect(reported).toEqual({ height: 720, width: 1280 });

    // TI2V-5B requires grid 32 rather than its base's 16; preserve variant and capability updates so shown and
    // compiled sizes agree.
    await act(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    const { height, width } = reported ?? { height: 0, width: 0 };
    expect(width).toBe(1280);
    expect(height % 32).toBe(0);
    expect(height).not.toBe(720);
  });

  /** Mount the slot alone over `project`, showing it `frame`, and return what it reports. */
  const reportFor = async (project: HarnessProject, frame: { height: number; width: number }) => {
    harness.project = createExternalStoreCore(project);
    const reported: { size: CanvasRenderSize['size'] } = { size: null };
    await mount(
      <GenerateCanvasRenderSize frame={frame}>
        {({ size }) => {
          reported.size = size;
          return null;
        }}
      </GenerateCanvasRenderSize>
    );
    return reported;
  };

  it('follows a frame the Size section has not committed yet', async () => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
    // Committed at 384x384, which Auto would grow to 512x512; a 600x600 drag is already past the optimum.
    const reported = await reportFor(createProject({ frame: { height: 384, width: 384 }, model: SD1_MODEL }), {
      height: 600,
      width: 600,
    });

    expect(reported.size).toEqual({ height: 600, width: 600 });
  });

  it('resolves a committed frame from the exact bbox, as the graph compiler does', async () => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
    // The bbox is off SDXL's grid; the Size section shows it snapped to 832x1216, an SDXL training size that Auto
    // would keep. The exact 830x1216 is not one, so Auto grows it toward 1024x1024's area.
    const project = createProject({ frame: { height: 1216, width: 832 }, model: SDXL_MODEL });
    project.canvas.document.bbox.width = 830;
    const reported = await reportFor(project, { height: 1216, width: 832 });

    expect(reported.size).toEqual({ height: 1240, width: 848 });
  });
});

describe('Size section in canvas mode', () => {
  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  it('describes the render size and follows a change of mode', async () => {
    // SD 1.5 is optimal at 512x512, so Auto grows this frame.
    await renderSizeSection(createProject({ frame: { height: 384, width: 384 }, model: SD1_MODEL }));

    await expect.element(renderAt()).toHaveTextContent('Optimal for model');
    // People who knew the setting by its legacy name find it in the select's description, tooltip open or not.
    await expect.element(renderAt()).toHaveAccessibleDescription(/^Scale before processing/);
    expect(badgeText()).toBe('384x384 → 512x512');
    expect(statusText()).toMatch(/^512×512 · 0\.26 MP/);
    // The stage fits the larger render size: 88px of preview spans 512px.
    expect(ghostSize()).toEqual([88, 88]);

    // The visible label is a pointer target for the select.
    await userEvent.click(page.elementLocator(host?.querySelector('label[aria-hidden]') as HTMLElement));
    await expect.element(renderAt()).toHaveFocus();
    await expect.element(renderAt()).toHaveAttribute('aria-expanded', 'false');

    await chooseRenderAt('Frame size');

    expect(canvasValues().scaleMethod).toBe('none');
    expect(badgeText()).toBe('384x384');
    expect(statusText()).toMatch(/^0\.15 MP/);
    expect(ghostSize()).toEqual([88, 88]);
  });

  it('snaps a custom render size to the model grid on commit and persists it', async () => {
    await renderSizeSection(createProject({ frame: { height: 384, width: 384 }, model: SD1_MODEL }));

    await chooseRenderAt('Custom');

    // Custom starts from the size the frame rendered at.
    const width = page.getByRole('spinbutton', { name: 'Render width' });
    const height = page.getByRole('spinbutton', { name: 'Render height' });
    await expect.element(width).toHaveValue('512');
    await expect.element(height).toHaveValue('512');
    expect(canvasValues()).toMatchObject({ scaleMethod: 'manual', scaledHeight: 512, scaledWidth: 512 });

    await userEvent.fill(width, '1023');
    // Typed digits land as typed until the edit is committed.
    expect(canvasValues().scaledWidth).toBe(1023);
    await userEvent.keyboard('{Enter}');

    expect(canvasValues().scaledWidth).toBe(1024);
    await expect.element(width).toHaveValue('1024');
    expect(badgeText()).toBe('384x384 → 1024x512');
    expect(statusText()).toMatch(/^1024×512 · 0\.52 MP/);
    // The preview spans the wider render size: 88px for 1024px, so the ghost is 44px tall.
    expect(ghostSize()).toEqual([88, 44]);
  });

  it('offers the optimal frame size only while the frame renders at its own size', async () => {
    // An undersized frame: Auto grows it and Custom sets its own, so resizing the frame adds nothing.
    await renderSizeSection(createProject({ frame: { height: 384, width: 384 }, model: SD1_MODEL }));
    await expect.element(optimalSizeButton()).not.toBeInTheDocument();
    await chooseRenderAt('Custom');
    await expect.element(optimalSizeButton()).not.toBeInTheDocument();
    await chooseRenderAt('Frame size');
    await expect.element(optimalSizeButton()).toBeInTheDocument();
    await unmount();

    // An oversized frame: Auto never shrinks it.
    await renderSizeSection(createProject({ frame: { height: 768, width: 768 }, model: SD1_MODEL }));
    await expect.element(optimalSizeButton()).toBeInTheDocument();
    await unmount();

    // A frame already at the recommended size has nothing to offer in any mode.
    await renderSizeSection(
      createProject({ canvas: { scaleMethod: 'none' }, frame: { height: 512, width: 512 }, model: SD1_MODEL })
    );
    await expect.element(optimalSizeButton()).not.toBeInTheDocument();
  });

  it('leaves the Size section as it is when the canvas is not the invocation source', async () => {
    harness.invocationSourceId = 'generate';
    await renderSizeSection(
      createProject({
        canvas: { scaleMethod: 'manual', scaledHeight: 1024, scaledWidth: 1536 },
        frame: { height: 384, width: 384 },
        model: SD1_MODEL,
      })
    );

    await expect.element(renderAt()).not.toBeInTheDocument();
    expect(badgeText()).toBe('384x384');
    expect(statusText()).toMatch(/^0\.15 MP/);
    await expect.element(optimalSizeButton()).toBeInTheDocument();
    // The ghost is the recommended 512x512, not the canvas's custom render size.
    expect(ghostSize()).toEqual([88, 88]);
  });
});
