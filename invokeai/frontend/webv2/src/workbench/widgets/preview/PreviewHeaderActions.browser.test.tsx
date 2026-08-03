import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
          },
        },
      },
    },
  },
});

const state = vi.hoisted(() => ({ filmstripVisible: true, showProgressImagesInViewer: true }));
const patchValues = vi.hoisted(() => vi.fn());
const updateProjectPreferences = vi.hoisted(() => vi.fn());

vi.mock('@features/queue/react', () => ({ useProgressImage: () => null }));

vi.mock('./previewHeaderStore', () => ({
  usePreviewHeaderContext: () => ({
    actionItem: null,
    actions: null,
    copyCurrentVideoFrame: null,
    isVideoFrameCopyAvailable: false,
    openItemMenu: vi.fn(),
    openVideoDetails: null,
  }),
}));

vi.mock('@workbench/widgetState', () => ({
  getProjectWidgetValues: () => ({ filmstripVisible: state.filmstripVisible }),
}));

vi.mock('@workbench/WorkbenchContext', () => ({
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

const render = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <PreviewHeaderActions {...({ region: 'center' } as unknown as WidgetViewProps)} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const button = (label: string): HTMLButtonElement =>
  host!.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;

beforeEach(() => {
  state.filmstripVisible = true;
  state.showProgressImagesInViewer = true;
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

    const [filmstripOn, progressOn] = bothOn.map((control) => getComputedStyle(control).backgroundColor);
    expect(filmstripOn).toBe(progressOn);

    state.filmstripVisible = false;
    state.showProgressImagesInViewer = false;
    await render();
    const bothOff = [button('Show filmstrip'), button('Show in-progress diffusion')];

    for (const control of bothOff) {
      expect(control.getAttribute('aria-pressed'), control.ariaLabel ?? '').toBe('false');
    }

    const [filmstripOff, progressOff] = bothOff.map((control) => getComputedStyle(control).backgroundColor);
    expect(filmstripOff).toBe(progressOff);
    expect(filmstripOff).not.toBe(filmstripOn);
  });

  it('still toggles the setting each one owns', async () => {
    await render();

    await settle(() => button('Hide filmstrip').click());
    expect(patchValues).toHaveBeenCalledWith('preview', { filmstripVisible: false });

    await settle(() => button('Hide in-progress diffusion').click());
    expect(updateProjectPreferences).toHaveBeenCalledWith({ showProgressImagesInViewer: false });
  });
});
