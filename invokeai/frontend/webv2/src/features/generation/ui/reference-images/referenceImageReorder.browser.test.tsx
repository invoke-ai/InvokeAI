import type { GenerateReferenceImage } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { moveReferenceImage } from '@features/generation/core/settings';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReferenceImageCard } from './ReferenceImageCard';

/** Browser coverage verifies arrow wiring and disabled endpoints beyond pure reorder tests. */
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          generate: {
            collapseReferenceImage: 'Collapse reference image',
            disableReferenceImage: 'Disable reference image',
            enableReferenceImage: 'Enable reference image',
            expandReferenceImage: 'Expand reference image',
            moveReferenceImageDown: 'Move reference image down',
            moveReferenceImageUp: 'Move reference image up',
            referenceImage: 'Reference Image',
            removeReferenceImage: 'Remove reference image',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onUseSize = vi.fn();
const onFindInGallery = vi.fn();
const onPatch = vi.fn();
const onRemove = vi.fn();

const buildReferenceImage = (id: string): GenerateReferenceImage => ({
  config: {
    image: { original: { image: { height: 512, image_name: `${id}.png`, width: 512 } } },
    type: 'external_reference_image',
  },
  id,
  isEnabled: true,
});

/** Mirrors the section: the move lands back on the rendered list. */
const Stack = ({ ids }: { ids: string[] }) => {
  const [referenceImages, setReferenceImages] = useState(() => ids.map(buildReferenceImage));
  const handleMove = useCallback((id: string, direction: -1 | 1) => {
    setReferenceImages((current) => [...moveReferenceImage(current, id, direction)]);
  }, []);

  return (
    <>
      {referenceImages.map((referenceImage, index) => (
        <ReferenceImageCard
          key={referenceImage.id}
          count={referenceImages.length}
          index={index}
          referenceImage={referenceImage}
          selectedModel={undefined}
          onFindInGallery={onFindInGallery}
          onMove={handleMove}
          onPatch={onPatch}
          onRemove={onRemove}
          onUseSize={onUseSize}
        />
      ))}
    </>
  );
};

const renderStack = async (ids: string[]) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Stack ids={ids} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const buttons = (label: string): HTMLButtonElement[] =>
  [...document.querySelectorAll(`button[aria-label="${label}"]`)] as HTMLButtonElement[];

/** Cards are titled by position, so the headings ARE the rendered order. */
const renderedOrder = (): string[] =>
  [...document.querySelectorAll('img')].map((image) => (image as HTMLImageElement).src.split('/').at(-2) ?? '');

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  onPatch.mockClear();
  onRemove.mockClear();
  onUseSize.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
});

describe('reference image reorder arrows', () => {
  it('moves a card through the stack and disables the arrows at each end', async () => {
    await renderStack(['first', 'second', 'third']);

    expect(renderedOrder()).toEqual(['first.png', 'second.png', 'third.png']);

    // Only the ends are frozen: the top card cannot rise, the bottom cannot fall.
    expect(buttons('Move reference image up').map((button) => button.disabled)).toEqual([true, false, false]);
    expect(buttons('Move reference image down').map((button) => button.disabled)).toEqual([false, false, true]);

    await act(() => {
      buttons('Move reference image down')[0]?.click();
    });

    expect(renderedOrder()).toEqual(['second.png', 'first.png', 'third.png']);

    // Up from the last slot is the inverse move, not another step down.
    await act(() => {
      buttons('Move reference image up')[2]?.click();
    });

    expect(renderedOrder()).toEqual(['second.png', 'third.png', 'first.png']);
  });

  it('hands focus to the arrow that stays live when a move lands on an end', async () => {
    await renderStack(['first', 'second', 'third']);

    // Hand off keyboard focus when the activated endpoint arrow becomes disabled.
    const up = buttons('Move reference image up')[1];

    await act(() => up?.focus());
    await act(() => up?.click());

    expect(renderedOrder()).toEqual(['second.png', 'first.png', 'third.png']);
    expect(document.activeElement).toBe(buttons('Move reference image down')[0]);
    expect((document.activeElement as HTMLButtonElement).disabled).toBe(false);

    // Same at the far end: the down arrow of the card that lands last.
    const down = buttons('Move reference image down')[1];

    await act(() => down?.focus());
    await act(() => down?.click());

    expect(renderedOrder()).toEqual(['second.png', 'third.png', 'first.png']);
    expect(document.activeElement).toBe(buttons('Move reference image up')[2]);
    expect((document.activeElement as HTMLButtonElement).disabled).toBe(false);
  });

  it('leaves a lone reference image with both arrows disabled', async () => {
    await renderStack(['only']);

    expect(buttons('Move reference image up')[0]?.disabled).toBe(true);
    expect(buttons('Move reference image down')[0]?.disabled).toBe(true);
  });
});
