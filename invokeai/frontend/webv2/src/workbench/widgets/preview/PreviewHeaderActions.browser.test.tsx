import type * as IdentityModule from '@features/identity';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
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
          preview: {
            hideFilmstrip: 'Hide filmstrip',
            showFilmstrip: 'Show filmstrip',
            details: 'Details',
            editOnCanvas: 'Edit on Canvas',
            imageActions: 'Image actions',
            selectForCompare: 'Select for Compare',
            starImage: 'Star image',
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
  filmstripVisible: true,
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
const updateProjectPreferences = vi.hoisted(() => vi.fn());
const showAll = vi.hoisted(() => vi.fn());
vi.mock('@features/queue/react', () => ({ useProgressImage: () => null }));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  useAuthSession: () => ({ accountEpoch: 1 }),
}));

// Partial: the real zoom limits decide which presets the menu offers.
vi.mock('./previewHeaderStore', async (importOriginal) => ({
  ...(await importOriginal<typeof PreviewHeaderStoreModule>()),
  usePreviewStageContext: () => ({
    stageElement: null,
    zoom: state.zoom && {
      fitPercent: state.zoom.fitPercent,
      isZoomed: state.zoom.isZoomed,
      percent: state.zoom.percent,
    },
  }),
  usePreviewHeaderContext: () => ({
    actionItem: state.actionItem,
    actions: state.actionItem ? { selectForCompare: vi.fn(), sendToCanvas: vi.fn(), setItemsStarred: vi.fn() } : null,
    copyCurrentVideoFrame: null,
    isVideoFrameCopyAvailable: false,
    openItemMenu: vi.fn(),
    position: null,
    zoom: state.zoom && { reset: state.zoom.reset, zoomTo: state.zoom.zoomTo },
  }),
}));

vi.mock('@workbench/widgetState', () => ({
  getProjectWidgetInstance: () => undefined,
  getProjectWidgetValues: () => ({ filmstripVisible: state.filmstripVisible, metadataOpen: false }),
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectSelector: (select: (project: unknown) => unknown) =>
    select({ settings: { viewerAutoSwitch: state.viewerAutoSwitch } }),
  useWorkbenchCommands: () => ({
    account: { updateProjectPreferences },
    widgets: { patchValues },
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 40);
    });
  });

const render = async (region: WidgetViewProps['region'] = 'center') => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <PreviewHeaderActions {...({ region, runtime: {} } as unknown as WidgetViewProps)} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const button = (label: string): HTMLButtonElement =>
  host!.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;

beforeEach(() => {
  state.actionItem = null;
  state.filmstripVisible = true;
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

describe('preview header filmstrip toggle', () => {
  it('states its position and toggles the setting it owns', async () => {
    await render();

    expect(button('Hide filmstrip').getAttribute('aria-pressed')).toBe('true');
    await settle(() => button('Hide filmstrip').click());
    expect(patchValues).toHaveBeenCalledWith('preview', { filmstripVisible: false });

    state.filmstripVisible = false;
    await render();
    expect(button('Show filmstrip').getAttribute('aria-pressed')).toBe('false');
  });
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

  it('shows the mode on the trigger, Off as soon as a selection turns it off', async () => {
    await render();
    expect(button('Auto-switch: In progress').textContent).toContain('In progress');

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

    expect(button('Auto-switch: Off').textContent).toContain('Off');
  });

  it('is the icon alone in a compact region, naming the mode in its tooltip and dimming for Off', async () => {
    await render('right');
    const trigger = button('Auto-switch: In progress');
    const icon = () =>
      button(`Auto-switch: ${state.viewerAutoSwitch === 'off' ? 'Off' : 'In progress'}`).querySelector('svg')!;

    expect(trigger.textContent).toBe('');
    const onOpacity = getComputedStyle(icon()).opacity;
    await settle(() => trigger.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse' })));
    await expect.element(page.getByRole('tooltip')).toHaveTextContent('Auto-switch: In progress');

    // A selection turns it Off: the trigger says so and its icon dims.
    state.viewerAutoSwitch = 'off';
    await render('right');
    expect(Number(getComputedStyle(icon()).opacity)).toBeLessThan(Number(onOpacity));
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

    const trigger = button('Zoom 88%');
    expect(trigger.textContent).toContain('88%');
    await settle(() => trigger.click());

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

  it('shows the zoom control in compact regions only while zoomed', async () => {
    state.zoom = { fitPercent: 88, isZoomed: false, percent: 88, reset: vi.fn(), zoomTo: vi.fn() };
    await render('right');
    expect(host!.querySelector('button[aria-label="Zoom 88%"]')).toBeNull();

    state.zoom = { fitPercent: 88, isZoomed: true, percent: 200, reset: vi.fn(), zoomTo: vi.fn() };
    await render('right');
    expect(host!.querySelector('button[aria-label="Zoom 200%"]')).not.toBeNull();
  });
});

describe('preview header details toggle', () => {
  it('sits last, after the filmstrip toggle, and reads as a toggle that opens the popover', async () => {
    state.actionItem = {
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
    await render();

    const labels = [...host!.querySelectorAll<HTMLButtonElement>('button')].map((b) => b.getAttribute('aria-label'));
    expect(labels.slice(-3)).toEqual(['Auto-switch: In progress', 'Hide filmstrip', 'Details']);
    expect(labels[0]).toBe('Edit on Canvas');
    expect(button('Details').getAttribute('aria-pressed')).toBe('false');

    await settle(() => button('Details').click());
    expect(patchValues).toHaveBeenCalledWith('preview', { metadataOpen: true });
  });
});
