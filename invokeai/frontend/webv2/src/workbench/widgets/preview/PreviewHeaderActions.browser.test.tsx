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
          preview: {
            hideFilmstrip: 'Hide filmstrip',
            hideInProgressDiffusion: 'Hide in-progress diffusion',
            showFilmstrip: 'Show filmstrip',
            showInProgressDiffusion: 'Show in-progress diffusion',
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
  showProgressImagesInViewer: true,
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
vi.mock('@features/queue/react', () => ({ useProgressImage: () => null }));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<typeof IdentityModule>()),
  useAuthSession: () => ({ accountEpoch: 1 }),
}));

vi.mock('./previewHeaderStore', () => ({
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
    select({ settings: { showProgressImagesInViewer: state.showProgressImagesInViewer } }),
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
  state.showProgressImagesInViewer = true;
  state.zoom = null;
  patchValues.mockClear();
  updateProjectPreferences.mockClear();
});

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('preview header toggles', () => {
  it('states both on/off positions the same way', async () => {
    await render();
    const bothOn = [button('Hide filmstrip'), button('Hide in-progress diffusion')];

    for (const control of bothOn) {
      expect(control.getAttribute('aria-pressed'), control.ariaLabel ?? '').toBe('true');
    }

    state.filmstripVisible = false;
    state.showProgressImagesInViewer = false;
    await render();
    const bothOff = [button('Show filmstrip'), button('Show in-progress diffusion')];

    for (const control of bothOff) {
      expect(control.getAttribute('aria-pressed'), control.ariaLabel ?? '').toBe('false');
    }
  });

  it('still toggles the setting each one owns', async () => {
    await render();

    await settle(() => button('Hide filmstrip').click());
    expect(patchValues).toHaveBeenCalledWith('preview', { filmstripVisible: false });

    await settle(() => button('Hide in-progress diffusion').click());
    expect(updateProjectPreferences).toHaveBeenCalledWith({ showProgressImagesInViewer: false });
  });
});

vi.mock('./livePreviewFollow', () => ({
  useLivePreviewFollow: () => ({ sessions: [], pinnedSessionId: null, pin: vi.fn(), showAll: vi.fn() }),
}));

describe('preview zoom menu', () => {
  it('reads the fitted percent, offers only presets above fit, and routes fit/presets to the loupe', async () => {
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
    // A fit above 100% leaves the 100% preset nowhere to go; the current level is not re-offered.
    await expect.element(item('100%')).toHaveAttribute('aria-disabled', 'true');
    await expect.element(item('200%')).toHaveAttribute('aria-disabled', 'true');
    await item('Fit to view').click();
    expect(reset).toHaveBeenCalledOnce();
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
    expect(labels.slice(-3)).toEqual(['Hide in-progress diffusion', 'Hide filmstrip', 'Details']);
    expect(labels[0]).toBe('Edit on Canvas');
    expect(button('Details').getAttribute('aria-pressed')).toBe('false');

    await settle(() => button('Details').click());
    expect(patchValues).toHaveBeenCalledWith('preview', { metadataOpen: true });
  });
});
