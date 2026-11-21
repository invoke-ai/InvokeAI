import '@fontsource/inter/index.css';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type * as IdentityModule from '@features/identity';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import type * as PreviewHeaderStoreModule from './previewHeaderStore';

import { PreviewHeaderActions } from './PreviewHeaderActions';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { edit: 'Edit' },
        widgets: {
          canvas: {
            staging: {
              autoSwitch: 'Auto-switch',
              autoSwitchCurrent: 'Auto-switch: {{mode}}',
              autoSwitchLatest: 'Newest',
              autoSwitchOff: 'Off',
              autoSwitchProgress: 'In progress',
            },
          },
          gallery: {
            export: { exportMenu: 'Export…', menuGroup: 'Export', quickExport: 'Quick export ({{summary}})' },
            itemActions: {
              recall: {
                verbs: {
                  clipSkip: 'Use CLIP skip',
                  dimensions: 'Use size',
                  prompts: 'Use prompt',
                  remix: 'Remix image',
                  seed: 'Use seed',
                },
              },
              upscale: { label: 'Upscale' },
              useAsControl: 'Use as control',
            },
            itemMenu: {
              addToCanvasAs: 'Add to Canvas as…',
              allSettings: 'All settings',
              allSettingsHint: 'Includes the original seed',
              continueWorking: 'Continue working',
              editOnCanvasHint: 'Adds an image layer and opens Canvas',
              remixHint: 'Same settings, new seed',
              extendInVideo: 'Extend in Video',
              remixVideo: 'Remix video',
              reuseFromImage: 'Reuse from image',
              reuseFromVideo: 'Reuse from video',
              sendToVideo: 'Send to Video',
              useAsPromptTemplate: 'Use as prompt template',
              useAsReferenceImage: 'Use as reference image',
              useAsConditioningClip: 'Use as conditioning clip',
              useAsReferenceVideo: 'Use as reference video',
              useImageAs: 'Use image as',
              useVideoAs: 'Use video as',
            },
          },
          preview: {
            compareArmedTooltip: 'Pick another image to compare · click to cancel',
            details: 'Details',
            editOnCanvas: 'Edit on Canvas',
            editThisImage: 'Edit this image',
            imageActions: 'Image actions',
            selectForCompare: 'Select for Compare',
            starImage: 'Star image',
            starVideo: 'Star video',
            use: 'Use',
            useInGenerate: 'Use this image in Generate',
            useThisVideo: 'Use this video',
            zoom: 'Zoom',
            zoomFit: 'Fit to view',
            zoomLevel: 'Zoom {{percent}}%',
            zoomPercent: '{{percent}}%',
          },
        },
      },
    },
  },
});

const state = vi.hoisted(() => ({
  actionItem: null as null | Record<string, unknown>,
  cancelCompare: null as null | (() => void),
  density: 'full' as 'compact' | 'full' | 'minimal' | null,
  viewerAutoSwitch: 'progress' as 'latest' | 'off' | 'progress',
  zoom: null as null | {
    fitPercent: number | null;
    isZoomed: boolean;
    percent: number | null;
    reset: () => void;
    zoomTo: (actualZoom: number) => void;
  },
}));
const patchValues = vi.hoisted(() => vi.fn());
const openItemMenu = vi.hoisted(() => vi.fn());
const itemActions = vi.hoisted(() => ({
  canUseAsConditioningClip: { audio: true, video: true },
  canUseAsReferenceImage: false,
  canUseAsReferenceVideo: true,
  deriveImageRecallCapabilities: vi.fn(),
  recallImageData: vi.fn(() => Promise.resolve()),
  recallVideoData: vi.fn(() => Promise.resolve()),
  selectForCompare: vi.fn(),
  sendImageAsControl: vi.fn(() => Promise.resolve()),
  sendToCanvas: vi.fn(() => Promise.resolve()),
  setItemsStarred: vi.fn(),
}));
const updateProjectPreferences = vi.hoisted(() => vi.fn());
const showAll = vi.hoisted(() => vi.fn());
const imageMetadata = vi.hoisted(() => vi.fn());
const videoMetadata = vi.hoisted(() => vi.fn());
vi.mock('@features/queue/react', () => ({ useProgressImage: () => null }));
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageByName: () => Promise.resolve({ hasWorkflow: false }),
  getGalleryImageMetadata: imageMetadata,
  getGalleryVideoMetadata: videoMetadata,
}));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  useAuthSession: () => ({ accountEpoch: 1 }),
}));

vi.mock('./previewHeaderStore', async (importOriginal) => ({
  ...(await importOriginal<typeof PreviewHeaderStoreModule>()),
  usePreviewStageContext: () => ({
    density: state.density,
    stageElement: null,
    zoom: state.zoom && {
      fitPercent: state.zoom.fitPercent,
      isZoomed: state.zoom.isZoomed,
      percent: state.zoom.percent,
    },
  }),
  usePreviewHeaderContext: () => ({
    actionItem: state.actionItem,
    actions: state.actionItem ? itemActions : null,
    cancelCompare: state.cancelCompare,
    copyCurrentVideoFrame: null,
    isVideoFrameCopyAvailable: false,
    openItemMenu,
    position: null,
    zoom: state.zoom && { reset: state.zoom.reset, zoomTo: state.zoom.zoomTo },
  }),
}));

vi.mock('@workbench/useOpenWorkbenchWidget', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useOpenWorkbenchWidget: () => vi.fn(),
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
}));

vi.mock('@workbench/widgetState', () => ({
  getProjectWidgetInstance: () => undefined,
  getProjectWidgetValues: () => ({ metadataOpen: false }),
  getResolvedProjectWidgetValues: () => ({ metadataOpen: false }),
  applyWidgetDefaults: (values: Record<string, unknown>) => values,
  resolveWidgetValues: () => ({}),
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectSelector: (select: (project: unknown) => unknown) =>
    select({ settings: { viewerAutoSwitch: state.viewerAutoSwitch } }),
  useWorkbenchCommands: () => ({
    account: { updateProjectPreferences },
    generation: { setSource: vi.fn() },
    widgets: { patchValues },
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 40);
    });
  });

const STILL_IMAGE = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-09-21T00:00:00Z',
  fullUrl: '/images/still.png',
  height: 8,
  isIntermediate: false,
  kind: 'image',
  name: 'still.png',
  starred: false,
  thumbnailUrl: '/thumbnails/still.webp',
  width: 8,
};

const render = async (region: WidgetViewProps['region'] = 'center') => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <PreviewHeaderActions {...({ region, runtime: {} } as unknown as WidgetViewProps)} />
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    );
  });
  await document.fonts.ready;
};

const button = (label: string): HTMLButtonElement =>
  host!.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;

beforeEach(() => {
  state.actionItem = null;
  state.cancelCompare = null;
  state.density = 'full';
  itemActions.selectForCompare.mockClear();
  openItemMenu.mockClear();
  itemActions.canUseAsReferenceImage = false;
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  imageMetadata.mockReset().mockResolvedValue({ seed: 1 });
  videoMetadata.mockReset().mockResolvedValue({ generation_mode: 'wan_t2v', positive_prompt: 'a fox', seed: 7 });
  itemActions.recallVideoData.mockClear();
  itemActions.deriveImageRecallCapabilities.mockReset();
  itemActions.deriveImageRecallCapabilities.mockReturnValue({
    all: true,
    clipSkip: false,
    dimensions: true,
    prompts: true,
    remix: true,
    seed: true,
    workflow: false,
  });
  itemActions.recallImageData.mockClear();
  itemActions.sendToCanvas.mockClear();
  state.viewerAutoSwitch = 'progress';
  state.zoom = null;
  patchValues.mockClear();
  showAll.mockClear();
  updateProjectPreferences.mockClear();
});

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('preview header Auto-switch', () => {
  it('offers the staging bar modes and writes the chosen one, letting go of a pinned session', async () => {
    await render();

    await settle(() => button('Auto-switch: In progress').click());
    const option = (name: string) => page.getByRole('menuitemradio', { name });
    await expect.element(option('In progress')).toHaveAttribute('aria-checked', 'true');
    await expect.element(option('Off')).toHaveAttribute('aria-checked', 'false');
    await option('Newest').click();

    expect(updateProjectPreferences).toHaveBeenCalledExactlyOnceWith({ viewerAutoSwitch: 'latest' });
    expect(showAll).toHaveBeenCalledOnce();
  });
});

vi.mock('./livePreviewFollow', () => ({
  useLivePreviewFollow: () => ({ sessions: [], pinnedSessionId: null, pin: vi.fn(), showAll }),
}));

describe('preview zoom menu', () => {
  it('reads the fitted percent, offers every preset but the current level, and routes them to the loupe', async () => {
    const reset = vi.fn();
    const zoomTo = vi.fn();
    state.zoom = { fitPercent: 88, isZoomed: false, percent: 88, reset, zoomTo };
    await render();

    await settle(() => button('Zoom 88%').click());

    const item = (label: string) => page.getByRole('menuitem', { name: label });
    await expect.element(item('Fit to view')).toHaveAttribute('aria-disabled', 'true');
    await expect.element(item('100%')).not.toHaveAttribute('aria-disabled');
    await item('200%').click();
    expect(zoomTo).toHaveBeenCalledExactlyOnceWith(2);

    state.zoom = { fitPercent: 120, isZoomed: true, percent: 200, reset, zoomTo };
    await render();
    await settle(() => button('Zoom 200%').click());
    await expect.element(item('100%')).not.toHaveAttribute('aria-disabled');
    await expect.element(item('200%')).toHaveAttribute('aria-disabled', 'true');
    await item('Fit to view').click();
    expect(reset).toHaveBeenCalledOnce();
  });
});

describe('preview header order', () => {
  it('puts the item actions first, one divider, then zoom, auto-switch, and the Details toggle last', async () => {
    state.actionItem = STILL_IMAGE;
    state.zoom = { fitPercent: 88, isZoomed: false, percent: 88, reset: vi.fn(), zoomTo: vi.fn() };
    await render();

    const labels = [...host!.querySelectorAll<HTMLButtonElement>('button')].map((b) => b.getAttribute('aria-label'));
    expect(labels[0]).toBe('Edit');
    expect(labels.slice(-3)).toEqual(['Zoom 88%', 'Auto-switch: In progress', 'Details']);
    expect(host!.querySelector('[aria-label="Hide filmstrip"], [aria-label="Show filmstrip"]')).toBeNull();
    const bar = host!.firstElementChild!;
    const dividers = [...bar.children].filter((child) => child.tagName === 'DIV' && child.childElementCount === 0);
    expect(dividers).toHaveLength(1);
    expect(dividers[0]!.nextElementSibling?.getAttribute('aria-label')).toBe('Zoom 88%');
    expect(button('Details').getAttribute('aria-pressed')).toBe('false');

    await settle(() => button('Details').click());
    expect(patchValues).toHaveBeenCalledWith('preview', { metadataOpen: true });
  });
});

describe('preview header Edit and Use', () => {
  const menuRows = (menu: Element) =>
    [...menu.querySelectorAll<HTMLElement>('[role="menuitem"], [data-part="item-group-label"]')].map((row) =>
      row.getAttribute('role') === 'menuitem'
        ? `${(row.querySelector('p') ?? row).textContent?.trim()}${row.getAttribute('aria-disabled') === 'true' ? ' (disabled)' : ''}`
        : `# ${row.textContent?.trim()}`
    );

  it('opens every way to continue from Edit, Edit on Canvas first, each step saying what happens', async () => {
    state.actionItem = STILL_IMAGE;
    await render();

    const trigger = button('Edit');
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu');
    await settle(() => trigger.click());
    expect(itemActions.sendToCanvas).not.toHaveBeenCalled();
    const menu = page.getByRole('menu').element();

    expect(menuRows(menu)).toEqual([
      '# Continue working',
      'Edit on Canvas',
      'Upscale',
      'Send to Video',
      'Add to Canvas as…',
      'widgets.canvas.import.newProjectFromImage',
    ]);
    expect(menu.textContent).toContain('Adds an image layer and opens Canvas');
    const rowText = (name: string) => {
      const row = [...menu.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
        (candidate) => candidate.querySelector('p')?.textContent === name
      )!;
      return [...row.querySelectorAll('p')].map((text) => getComputedStyle(text).fontSize);
    };
    expect(rowText('Upscale')).toEqual(rowText('Edit on Canvas'));

    const editOnCanvas = page.getByRole('menuitem', { name: /^Edit on Canvas/ });
    await editOnCanvas.hover();
    await editOnCanvas.click();
    expect(itemActions.sendToCanvas).toHaveBeenCalledExactlyOnceWith(
      [expect.objectContaining({ imageName: 'still.png' })],
      'raster'
    );
  });

  it('loads what the image can lend only when Use opens, then lists both groups with the viewer keys', async () => {
    state.actionItem = STILL_IMAGE;
    await render();
    expect(imageMetadata).not.toHaveBeenCalled();

    await settle(() => button('Use this image in Generate').click());
    const menu = page.getByRole('menu').element();

    expect(imageMetadata).toHaveBeenCalledExactlyOnceWith('still.png', expect.any(AbortSignal));
    await vi.waitFor(() =>
      expect(menuRows(menu)).toEqual([
        '# Use image as',
        'Use as reference image (disabled)',
        'Use as control',
        'Use as prompt template',
        '# Reuse from image',
        'Remix image',
        'All settings',
        'Use prompt',
        'Use seed',
        'Use size',
        'Use CLIP skip (disabled)',
        '# Export',
        'Export…',
        'Quick export (PNG)',
      ])
    );
    const keycaps = [...menu.querySelectorAll<HTMLElement>('[role="menuitem"]')].map(
      (row) => [...row.querySelectorAll('kbd')].map((key) => key.textContent).join('+') || null
    );
    expect(keycaps.slice(3)).toEqual(['r', 'a', 'p', 's', 'd', null, 'ctrl+shift+e', null]);
    expect(menu.textContent).toContain('Same settings, new seed');

    const seed = page.getByRole('menuitem', { name: /^Use seed/ });
    await seed.hover();
    await seed.click();
    expect(itemActions.recallImageData).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ imageName: 'still.png' }),
      'seed'
    );
  });

  it('shows Select for compare pressed while it waits for a second image, and switches it off again', async () => {
    state.actionItem = STILL_IMAGE;
    await render();
    const compare = button('Select for Compare');

    expect(compare).toHaveAttribute('aria-pressed', 'false');
    await settle(() => compare.click());
    expect(itemActions.selectForCompare).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ imageName: 'still.png' })
    );

    const cancelCompare = vi.fn();
    state.cancelCompare = cancelCompare;
    await settle(() => root?.unmount());
    await render();
    const pressed = button('Select for Compare');

    expect(pressed).toHaveAttribute('aria-pressed', 'true');
    await settle(() => pressed.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse' })));
    await expect
      .element(page.getByRole('tooltip'))
      .toHaveTextContent('Pick another image to compare · click to cancel');
    await settle(() => pressed.click());
    expect(cancelCompare).toHaveBeenCalledOnce();
    expect(itemActions.selectForCompare).toHaveBeenCalledOnce();
  });

  it('holds an open menu on the image it opened on while a new result takes the selection', async () => {
    const rerender = () =>
      settle(() => {
        root?.render(
          <QueryClientProvider client={queryClient}>
            <I18nextProvider i18n={i18n}>
              <ChakraProvider value={system}>
                <PreviewHeaderActions {...({ region: 'center', runtime: {} } as unknown as WidgetViewProps)} />
              </ChakraProvider>
            </I18nextProvider>
          </QueryClientProvider>
        );
      });
    state.actionItem = STILL_IMAGE;
    await render();
    await settle(() => button('Use this image in Generate').click());
    await vi.waitFor(() =>
      expect(
        page
          .getByRole('menuitem', { name: /^Use seed/ })
          .element()
          .getAttribute('aria-disabled')
      ).not.toBe('true')
    );

    state.actionItem = { ...STILL_IMAGE, name: 'newer.png' };
    await rerender();

    expect(imageMetadata).toHaveBeenCalledExactlyOnceWith('still.png', expect.any(AbortSignal));
    const seed = page.getByRole('menuitem', { name: /^Use seed/ });
    await seed.hover();
    await seed.click();
    expect(itemActions.recallImageData).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ imageName: 'still.png' }),
      'seed'
    );
  });
});
