import type { GenerateReferenceImage } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ReferenceImageCard } from './ReferenceImageCard';

// The real catalog, so a mistyped key shows up as the key and fails the label lookups below.
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onPatch = vi.fn();
const image = { original: { image: { height: 512, image_name: 'pose.png', width: 512 } } };
const STYLE_REFERENCE: GenerateReferenceImage = {
  config: { image, styleStrength: 1.6, type: 'krea2_reference_image' },
  id: 'pose',
  isEnabled: true,
};

/** Mirrors the section: each patch lands back on the rendered reference. */
const Card = ({ initial }: { initial: GenerateReferenceImage }) => {
  const [referenceImage, setReferenceImage] = useState(initial);
  const handlePatch = useCallback((id: string, patch: Partial<GenerateReferenceImage>) => {
    onPatch(id, patch);
    setReferenceImage((current) => ({ ...current, ...patch }));
  }, []);

  return (
    <ReferenceImageCard
      count={1}
      index={0}
      referenceImage={referenceImage}
      selectedModel={undefined}
      onFindInGallery={vi.fn()}
      onMove={vi.fn()}
      onPatch={handlePatch}
      onRemove={vi.fn()}
      onUseSize={vi.fn()}
    />
  );
};

const renderCard = async (initial: GenerateReferenceImage) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Card initial={initial} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const modeGroup = () => host?.querySelector<HTMLElement>('[data-scope="segment-group"][data-part="root"]');
const modeRadio = (label: 'Edit' | 'Style') =>
  [...(modeGroup()?.querySelectorAll<HTMLElement>('[data-part="item"]') ?? [])]
    .find((item) => item.textContent === label)
    ?.querySelector<HTMLInputElement>('input[type="radio"]') ?? null;
const clickMode = async (label: 'Edit' | 'Style') => {
  const item = [...(modeGroup()?.querySelectorAll<HTMLElement>('[data-part="item"]') ?? [])].find(
    (candidate) => candidate.textContent === label
  );

  await act(async () => {
    await userEvent.click(item!);
  });
};
const hasStrengthControl = () => host?.textContent?.includes('Style strength') ?? false;

beforeEach(() => {
  host = document.createElement('div');
  host.style.width = '360px';
  document.body.append(host);
  root = createRoot(host);
  onPatch.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
});

describe('Krea-2 reference mode switch', () => {
  it('switches a reference to Edit and back to Style, keeping its image and restarting the strength', async () => {
    await renderCard(STYLE_REFERENCE);

    expect(modeGroup()?.getAttribute('aria-label')).toBe('Mode');
    expect(modeRadio('Style')?.checked).toBe(true);
    expect(hasStrengthControl()).toBe(true);

    await clickMode('Edit');

    expect(onPatch).toHaveBeenLastCalledWith('pose', {
      config: { image, type: 'krea2_edit_reference_image' },
    });
    expect(modeRadio('Edit')?.checked).toBe(true);
    // Edit references have no strength; the card shows only the mode.
    expect(hasStrengthControl()).toBe(false);

    await clickMode('Style');

    expect(onPatch).toHaveBeenLastCalledWith('pose', {
      config: { image, styleStrength: 1, type: 'krea2_reference_image' },
    });
    expect(hasStrengthControl()).toBe(true);
  });

  it('switches with the arrow keys from the focused mode', async () => {
    await renderCard(STYLE_REFERENCE);

    await act(() => modeRadio('Style')?.focus());
    await act(async () => {
      await userEvent.keyboard('{ArrowRight}');
    });

    expect(onPatch).toHaveBeenLastCalledWith('pose', {
      config: { image, type: 'krea2_edit_reference_image' },
    });
    expect(document.activeElement).toBe(modeRadio('Edit'));
  });

  it('switches the mode of a switched-off reference, so it can join an edit set before it runs', async () => {
    await renderCard({ ...STYLE_REFERENCE, isEnabled: false });

    expect(modeRadio('Edit')?.disabled).toBe(false);

    await clickMode('Edit');

    expect(onPatch).toHaveBeenLastCalledWith('pose', {
      config: { image, type: 'krea2_edit_reference_image' },
    });
    expect(modeRadio('Edit')?.checked).toBe(true);
  });
});
