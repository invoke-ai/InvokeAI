import type { GenerateModelConfig, GenerateReferenceImage } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { type GenerationUiAdapter, GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ReferenceImageRow } from './ReferenceImageRow';

/** Crop intermediates have separate identities and stay out of the gallery; retain the original for reveal. */
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onFindInGallery = vi.fn();
const noop = vi.fn();
const ADAPTER = { models: { ModelSelect: () => null } } as unknown as GenerationUiAdapter;
const SDXL: GenerateModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };

const CROPPED_IMAGE = {
  crop: {
    box: { height: 128, width: 128, x: 16, y: 16 },
    image: { height: 128, image_name: 'crop-intermediate.png', width: 128 },
    ratio: 1,
  },
  original: { image: { height: 512, image_name: 'original.png', width: 512 } },
};

const CROPPED_REFERENCE: GenerateReferenceImage = {
  config: { image: CROPPED_IMAGE, type: 'external_reference_image' },
  id: 'cropped',
  isEnabled: true,
};

const CROPPED_IP_ADAPTER: GenerateReferenceImage = {
  config: {
    beginEndStepPct: [0, 1],
    clipVisionModel: 'ViT-H',
    image: CROPPED_IMAGE,
    method: 'full',
    model: { base: 'sdxl', key: 'ip', name: 'IP Adapter SDXL', type: 'ip_adapter' },
    type: 'ip_adapter',
    weight: 1,
  },
  id: 'cropped-ip',
  isEnabled: true,
};

const renderRow = async (referenceImage: GenerateReferenceImage, selectedModel?: GenerateModelConfig) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <ReferenceImageRow
              count={1}
              index={0}
              referenceImage={referenceImage}
              selectedModel={selectedModel}
              onFindInGallery={onFindInGallery}
              onMove={noop}
              onPatch={noop}
              onRemove={noop}
              onRename={noop}
              onUseSize={noop}
            />
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const menuItem = (value: string) => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);

const openMenu = async () => {
  await act(() =>
    userEvent.click(host!.querySelector<HTMLElement>('button[aria-haspopup="menu"]:not([data-list-primary])')!)
  );
  await expect.poll(() => menuItem('find-in-gallery')).not.toBeNull();
};

beforeEach(() => {
  host = document.createElement('div');
  host.style.width = '360px';
  document.body.append(host);
  root = createRoot(host);
  onFindInGallery.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
});

describe('reference image find in gallery', () => {
  it('reveals the original image a crop was taken from, not the crop upload, from the row menu', async () => {
    await renderRow(CROPPED_REFERENCE);

    // The tile shows the crop, marked as cropped, which is what makes this worth pinning.
    expect(host!.querySelector('img')?.src).toContain('crop-intermediate.png');
    expect(host!.querySelector('[data-cropped]')).not.toBeNull();
    expect(host!.querySelector('[data-list-primary]')?.textContent).toContain('128 × 128 · cropped');

    await openMenu();
    await act(() => userEvent.click(menuItem('find-in-gallery')!));

    expect(onFindInGallery).toHaveBeenCalledWith('original.png');
  });

  it('stays available while the reference is off, in the menu and in the open settings, unlike crop and size', async () => {
    await renderRow({ ...CROPPED_IP_ADAPTER, isEnabled: false }, SDXL);

    await openMenu();
    // Locate stays enabled for disabled references because it does not edit generation.
    expect(menuItem('find-in-gallery')?.hasAttribute('data-disabled')).toBe(false);
    expect(menuItem('crop')?.hasAttribute('data-disabled')).toBe(true);
    expect(menuItem('use-size')?.hasAttribute('data-disabled')).toBe(true);
    await act(() => userEvent.keyboard('{Escape}'));

    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Reference image 1 settings"]')!.click());
    const find = host!.querySelector<HTMLButtonElement>('button[aria-label="Find original.png in Gallery"]');

    expect(find?.disabled).toBe(false);
    expect(host!.querySelector<HTMLButtonElement>('button[aria-label="Crop Reference image 1"]')?.disabled).toBe(true);
    await act(() => find?.click());
    expect(onFindInGallery).toHaveBeenCalledWith('original.png');
  });
});
