import type { GenerateReferenceImage } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ReferenceImageCard } from './ReferenceImageCard';

/** FLUX.3 names its references in the prompt; the card's tag button explains the tag and copies it where it can. */
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

const TAG = '<ref_image_0>';
const EXPLANATION = `Write ${TAG} in the prompt to refer to this image.`;

const REFERENCE: GenerateReferenceImage = {
  config: {
    image: { original: { image: { height: 512, image_name: 'fox.png', width: 512 } } },
    type: 'flux2_reference_image',
  },
  id: 'fox',
  isEnabled: true,
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let originalClipboard: PropertyDescriptor | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const noop = vi.fn();

const renderCard = () =>
  act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ReferenceImageCard
            count={1}
            index={0}
            promptTag={TAG}
            referenceImage={REFERENCE}
            selectedModel={undefined}
            onFindInGallery={noop}
            onMove={noop}
            onPatch={noop}
            onRemove={noop}
            onUseSize={noop}
          />
        </ChakraProvider>
      </I18nextProvider>
    )
  );

const tagButton = () => page.getByRole('button', { exact: true, name: `Copy prompt tag ${TAG}` });
const tooltips = () => [...document.querySelectorAll('[role="tooltip"]')].map((tooltip) => tooltip.textContent);

beforeEach(() => {
  originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await userEvent.unhover(document.body);
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  if (originalClipboard) {
    Object.defineProperty(navigator, 'clipboard', originalClipboard);
  } else {
    Reflect.deleteProperty(navigator, 'clipboard');
  }
});

describe('reference image prompt tag', () => {
  it('explains the tag and copies it', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    await renderCard();

    await expect.element(tagButton()).not.toHaveAttribute('aria-disabled');
    await act(() => userEvent.hover(tagButton()));
    await vi.waitFor(() => expect(tooltips()).toEqual([`${EXPLANATION} Click to copy.`]));
    await act(() => userEvent.click(tagButton()));

    expect(writeText).toHaveBeenCalledWith(TAG);
    await expect.element(page.getByText(`Copied ${TAG}`)).toBeInTheDocument();
  });

  it('still explains the tag, beside why it cannot copy, when the clipboard is unavailable', async () => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    await renderCard();
    const expected = `${EXPLANATION} Copying needs HTTPS or localhost`;

    await expect.element(tagButton()).toHaveAttribute('aria-disabled', 'true');
    await expect.element(tagButton()).toHaveAccessibleDescription(expected);
    await act(() => userEvent.hover(tagButton()));
    // One tooltip: the unavailable button's own, not a second hint around it.
    await vi.waitFor(() => expect(tooltips()).toEqual([expected]));
  });
});
