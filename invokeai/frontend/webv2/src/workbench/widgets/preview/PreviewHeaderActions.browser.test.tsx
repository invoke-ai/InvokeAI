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
              reuseFromImage: 'Reuse from image',
              sendToVideo: 'Send to Video',
              useAsPromptTemplate: 'Use as prompt template',
              useAsReferenceImage: 'Use as reference image',
              useImageAs: 'Use image as',
            },
          },
          preview: {
            details: 'Details',
            editOnCanvas: 'Edit on Canvas',
            editThisImage: 'Edit this image',
            imageActions: 'Image actions',
            selectForCompare: 'Select for Compare',
            starImage: 'Star image',
            use: 'Use',
            useInGenerate: 'Use this image in Generate',
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
  canUseAsReferenceImage: false,
  deriveImageRecallCapabilities: vi.fn(),
  recallImageData: vi.fn(() => Promise.resolve()),
  selectForCompare: vi.fn(),
  sendImageAsControl: vi.fn(() => Promise.resolve()),
  sendToCanvas: vi.fn(() => Promise.resolve()),
  setItemsStarred: vi.fn(),
}));
const updateProjectPreferences = vi.hoisted(() => vi.fn());
const showAll = vi.hoisted(() => vi.fn());
const imageMetadata = vi.hoisted(() => vi.fn());
vi.mock('@features/queue/react', () => ({ useProgressImage: () => null }));
// The transport, under both the gallery's shared metadata query and its public image port.
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageByName: () => Promise.resolve({ hasWorkflow: false }),
  getGalleryImageMetadata: imageMetadata,
}));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  useAuthSession: () => ({ accountEpoch: 1 }),
}));

// Partial: the real zoom limits decide which presets the menu offers.
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
// The Upscale submenu asks for the model library when it mounts; nothing here opens it.
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
}));

vi.mock('@workbench/widgetState', () => ({
  getProjectWidgetInstance: () => undefined,
  getProjectWidgetValues: () => ({ metadataOpen: false }),
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

/** The header sizes itself by the density the stage publishes; the region is passed only to show it is ignored. */
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
};

const button = (label: string): HTMLButtonElement =>
  host!.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;

beforeEach(() => {
  state.actionItem = null;
  state.density = 'full';
  openItemMenu.mockClear();
  itemActions.canUseAsReferenceImage = false;
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  imageMetadata.mockReset().mockResolvedValue({ seed: 1 });
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

  it('names the mode, Off as soon as a selection turns it off', async () => {
    await render();
    expect(button('Auto-switch: In progress')).not.toBeNull();

    // A Gallery selection writes Off; the trigger reads it from the project on the next render.
    state.viewerAutoSwitch = 'off';
    await settle(() => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <PreviewHeaderActions {...({ region: 'center', runtime: {} } as unknown as WidgetViewProps)} />
          </ChakraProvider>
        </I18nextProvider>
      );
    });

    expect(button('Auto-switch: Off')).not.toBeNull();
  });

  it('is the icon alone at every density, naming the mode in its tooltip, with each mode its own icon', async () => {
    // Lucide names each glyph in a class; Chakra's styling classes differ between trigger and menu.
    const glyph = (svg: Element | undefined) => [...(svg?.classList ?? [])].find((name) => name.startsWith('lucide-'));
    const icons = new Map<string, string | undefined>();

    await render();
    const trigger = button('Auto-switch: In progress');
    expect(trigger.textContent).toBe('');
    expect(trigger.querySelectorAll('svg')).toHaveLength(1);
    await settle(() => trigger.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse' })));
    await expect.element(page.getByRole('tooltip')).toHaveTextContent('Auto-switch: In progress');

    for (const [mode, label] of [
      ['off', 'Off'],
      ['progress', 'In progress'],
      ['latest', 'Newest'],
    ] as const) {
      state.viewerAutoSwitch = mode;
      await render();
      icons.set(mode, glyph(host!.querySelector('[aria-label^="Auto-switch:"] svg')!));

      // The menu shows each mode with the icon the trigger takes for it, so the mapping can be learned.
      await settle(() => button(`Auto-switch: ${label}`).click());
      const itemIcon = [...page.getByRole('menuitemradio', { name: label }).element().querySelectorAll('svg')].find(
        (svg) => !svg.closest('[data-part="item-indicator"]')
      );
      expect(glyph(itemIcon)).toBe(icons.get(mode));
      await settle(() => root?.unmount());
      host?.remove();
    }

    // Off reads by its shape, not by dimming the icon the other modes share.
    expect(new Set(icons.values()).size).toBe(3);
  });

  it('is square and sized like the icon button beside it', async () => {
    const box = (label: string) => button(label).getBoundingClientRect();
    state.actionItem = STILL_IMAGE;

    await render();
    expect(box('Auto-switch: In progress').height).toBe(box('Details').height);
    expect(box('Auto-switch: In progress').width).toBe(box('Details').width);
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
    // Fit is where the image already is; 100% and 200% are above the 88% fit, 400% too.
    await expect.element(item('Fit to view')).toHaveAttribute('aria-disabled', 'true');
    await expect.element(item('100%')).not.toHaveAttribute('aria-disabled');
    await item('200%').click();
    expect(zoomTo).toHaveBeenCalledExactlyOnceWith(2);

    state.zoom = { fitPercent: 120, isZoomed: true, percent: 200, reset, zoomTo };
    await render();
    await settle(() => button('Zoom 200%').click());
    // 100% is below a 120% fit yet still offered, so an enlarged image can be seen at its own size; the current
    // level is not re-offered.
    await expect.element(item('100%')).not.toHaveAttribute('aria-disabled');
    await expect.element(item('200%')).toHaveAttribute('aria-disabled', 'true');
    await item('Fit to view').click();
    expect(reset).toHaveBeenCalledOnce();
  });

  it('offers the presets between 100% and a fit that enlarges the image, since zoom goes down to 100%', async () => {
    const zoomTo = vi.fn();
    const item = (label: string) => page.getByRole('menuitem', { name: label });

    state.zoom = { fitPercent: 250, isZoomed: false, percent: 250, reset: vi.fn(), zoomTo };
    await render();
    await settle(() => button('Zoom 250%').click());
    await expect.element(item('100%')).not.toHaveAttribute('aria-disabled');
    await expect.element(item('400%')).not.toHaveAttribute('aria-disabled');
    await item('200%').click();
    expect(zoomTo).toHaveBeenCalledExactlyOnceWith(2);
  });

  it('offers 100% at a fit just past it, though both read as 100%, and not at an exact 100% fit', async () => {
    const zoomTo = vi.fn();
    const item = (label: string) => page.getByRole('menuitem', { name: label });

    state.zoom = { fitPercent: 100.2, isZoomed: false, percent: 100, reset: vi.fn(), zoomTo };
    await render();
    await settle(() => button('Zoom 100%').click());
    await item('100%').click();
    expect(zoomTo).toHaveBeenCalledExactlyOnceWith(1);

    state.zoom = { fitPercent: 100, isZoomed: false, percent: 100, reset: vi.fn(), zoomTo };
    await render();
    await settle(() => button('Zoom 100%').click());
    await expect.element(item('100%')).toHaveAttribute('aria-disabled', 'true');
  });

  it('names the fit level but shows the percent only while zoomed in full density', async () => {
    state.zoom = { fitPercent: 88, isZoomed: false, percent: 88, reset: vi.fn(), zoomTo: vi.fn() };
    await render();
    const fitted = button('Zoom 88%');
    expect(fitted.textContent).not.toContain('%');
    expect(fitted.querySelector('.lucide-zoom-in')).not.toBeNull();
    await settle(() => fitted.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse' })));
    await expect.element(page.getByRole('tooltip')).toHaveTextContent('Zoom 88%');

    state.zoom = { fitPercent: 88, isZoomed: true, percent: 200, reset: vi.fn(), zoomTo: vi.fn() };
    await render();
    const zoomed = button('Zoom 200%');
    expect(zoomed.textContent).toContain('200%');
    expect(zoomed.querySelector('.lucide-zoom-in')).not.toBeNull();
  });

  it('follows the density the stage publishes and is compact until it has', async () => {
    for (const density of ['compact', 'minimal', null] as const) {
      state.density = density;
      state.zoom = { fitPercent: 88, isZoomed: false, percent: 88, reset: vi.fn(), zoomTo: vi.fn() };
      await render('center');
      expect(host!.querySelector('button[aria-label="Zoom 88%"]')).toBeNull();

      state.zoom = { fitPercent: 88, isZoomed: true, percent: 200, reset: vi.fn(), zoomTo: vi.fn() };
      await render('center');
      const trigger = button('Zoom 200%');
      // Compact is the readout and chevron only: the magnifier would leave the widget title no room.
      expect(trigger.textContent).toContain('200%');
      expect(trigger.querySelector('.lucide-zoom-in')).toBeNull();
      expect(trigger.getBoundingClientRect().width).toBeLessThan(70);
      await settle(() => root?.unmount());
      host?.remove();
    }
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

  it('shows only the viewer controls, without a divider, when nothing is selected', async () => {
    state.zoom = null;
    await render();

    const bar = host!.firstElementChild!;
    expect([...bar.children].map((child) => child.getAttribute('aria-label'))).toEqual(['Auto-switch: In progress']);
  });
});

describe('preview header Edit and Use', () => {
  const VIDEO = { ...STILL_IMAGE, durationSeconds: 4, fullUrl: '/videos/clip.mp4', kind: 'video', name: 'clip.mp4' };
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
    // A menu trigger only: nothing goes to Canvas until an item is chosen.
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
    // A hinted submenu row (Upscale) sets its label and hint as the hinted item rows do.
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
        // The Generate panel's model has no room for another reference.
        'Use as reference image (disabled)',
        'Use as control',
        'Use as prompt template',
        '# Reuse from image',
        'Remix image',
        'All settings',
        'Use prompt',
        'Use seed',
        'Use size',
        // The image records no CLIP skip.
        'Use CLIP skip (disabled)',
      ])
    );
    const keycaps = [...menu.querySelectorAll<HTMLElement>('[role="menuitem"]')].map(
      (row) => [...row.querySelectorAll('kbd')].map((key) => key.textContent).join('+') || null
    );
    expect(keycaps.slice(3)).toEqual(['r', 'a', 'p', 's', 'd', null]);
    expect(menu.textContent).toContain('Same settings, new seed');

    // The keycap is part of the item's name.
    const seed = page.getByRole('menuitem', { name: /^Use seed/ });
    await seed.hover();
    await seed.click();
    expect(itemActions.recallImageData).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ imageName: 'still.png' }),
      'seed'
    );
  });

  it('keeps Edit and Use down to compact, icon-only there, and leaves minimal to star and the full ⋮', async () => {
    const strip = () =>
      [...host!.querySelectorAll<HTMLButtonElement>('button')]
        .map((entry) => entry.getAttribute('aria-label'))
        .filter((label) => label !== null && !label.startsWith('Auto-switch') && label !== 'Details');
    state.actionItem = STILL_IMAGE;

    await render();
    expect(strip()).toEqual([
      'Edit',
      'Use this image in Generate',
      'Select for Compare',
      'Star image',
      'Image actions',
    ]);
    expect(button('Edit').textContent).toContain('Edit');
    await settle(() => button('Image actions').click());
    expect(openItemMenu).toHaveBeenLastCalledWith(expect.any(Number), expect.any(Number), 'housekeeping');

    // Compact: the two menus as icon and chevron, named in full; compare waits in ⋮, which stays housekeeping.
    state.density = 'compact';
    await render();
    expect(strip()).toEqual(['Edit this image', 'Use this image in Generate', 'Star image', 'Image actions']);
    for (const label of ['Edit this image', 'Use this image in Generate']) {
      expect(button(label).textContent).toBe('');
      expect(button(label).getAttribute('aria-haspopup')).toBe('menu');
      expect(button(label).getBoundingClientRect().width).toBeLessThanOrEqual(32);
    }
    await settle(() => button('Image actions').click());
    expect(openItemMenu).toHaveBeenLastCalledWith(expect.any(Number), expect.any(Number), 'housekeeping');

    state.density = 'minimal';
    await render();
    expect(strip()).toEqual(['Star image', 'Image actions']);
    await settle(() => button('Image actions').click());
    expect(openItemMenu).toHaveBeenLastCalledWith(expect.any(Number), expect.any(Number), 'full');
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

    // Auto-switch on Newest brings a finished result in under the open menu.
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

  it('keeps a video to star and the full ⋮ menu', async () => {
    state.actionItem = VIDEO;
    await render();

    expect(button('Edit')).toBeNull();
    expect(button('Use this image in Generate')).toBeNull();
    await settle(() => button('Image actions').click());
    expect(openItemMenu).toHaveBeenLastCalledWith(expect.any(Number), expect.any(Number), 'full');
  });
});
