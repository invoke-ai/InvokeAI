import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { GalleryStatusChip } from './GalleryWidgetView';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      key === 'widgets.gallery.statusChip' ? `Gallery: ${String(values?.count)} items` : key,
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('renders the compact gallery status through the localized status-chip key', async () => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryStatusChip count={12} />
      </ChakraProvider>
    )
  );

  expect(document.body.textContent).toContain('Gallery: 12 items');
  expect(document.body.textContent).not.toContain('widgets.gallery.statusChip');
});
