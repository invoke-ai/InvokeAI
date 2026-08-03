/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryItemSearch } from './GalleryItemSearch';
import { GalleryWidgetContext } from './GalleryWidgetContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string) => key,
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderSearch = async () => {
  const contextValue = {
    actions: { setSearchTerm: vi.fn() },
    gallery: { searchTerm: '' },
  } as unknown as GalleryWidgetContextValue;

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={contextValue}>
          <GalleryItemSearch />
        </GalleryWidgetContext>
      </ChakraProvider>
    );
  });
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  document.querySelectorAll('[data-scope="popover"][data-part="positioner"]').forEach((element) => element.remove());
  host?.remove();
  host = null;
  root = null;
});

describe('GalleryItemSearch help', () => {
  it('shows the relative value in a valid prefixed token', async () => {
    await renderSearch();
    const trigger = host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.searchHelpTitle"]');

    await act(async () => {
      trigger?.click();
      await Promise.resolve();
    });

    const examples = Array.from(document.querySelectorAll('code')).map((element) => element.textContent);

    expect(examples).toContain('from:7d');
    expect(examples).not.toContain('7d');
  });
});
