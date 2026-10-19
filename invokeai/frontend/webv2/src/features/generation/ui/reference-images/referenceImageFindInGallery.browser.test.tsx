import type { GenerateReferenceImage } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReferenceImageCard } from './ReferenceImageCard';

/** Crop intermediates have separate identities and stay out of the gallery; retain the original for reveal. */
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          gallery: { findNamedInGallery: 'Find {{name}} in Gallery' },
          generate: { referenceImage: 'Reference Image' },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onFindInGallery = vi.fn();
const noop = vi.fn();

const CROPPED_REFERENCE: GenerateReferenceImage = {
  config: {
    image: {
      crop: {
        box: { height: 128, width: 128, x: 16, y: 16 },
        image: { height: 128, image_name: 'crop-intermediate.png', width: 128 },
        ratio: 1,
      },
      original: { image: { height: 512, image_name: 'original.png', width: 512 } },
    },
    type: 'external_reference_image',
  },
  id: 'cropped',
  isEnabled: true,
};

const renderCard = async (referenceImage: GenerateReferenceImage) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ReferenceImageCard
            count={1}
            index={0}
            referenceImage={referenceImage}
            selectedModel={undefined}
            onFindInGallery={onFindInGallery}
            onMove={noop}
            onPatch={noop}
            onRemove={noop}
            onUseSize={noop}
          />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

// Use media-specific action names to distinguish repeated cards accessibly.
const findButton = (): HTMLButtonElement | null =>
  document.querySelector<HTMLButtonElement>('button[aria-label="Find original.png in Gallery"]');

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  onFindInGallery.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
});

describe('reference image find-in-gallery badge', () => {
  it('reveals the original image a crop was taken from, not the crop upload', async () => {
    await renderCard(CROPPED_REFERENCE);

    // The thumbnail is showing the crop, which is what makes this worth pinning.
    expect(document.querySelector('img')?.src).toContain('crop-intermediate.png');

    await act(() => findButton()?.click());

    expect(onFindInGallery).toHaveBeenCalledWith('original.png');
  });

  it('stays available while the reference is toggled off', async () => {
    await renderCard({ ...CROPPED_REFERENCE, isEnabled: false });

    // Locate stays enabled for disabled references because it does not edit generation.
    expect(findButton()?.disabled).toBe(false);
  });
});
