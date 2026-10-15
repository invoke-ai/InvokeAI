import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ModelImageUpload } from './ModelImageUpload';

const dependencies = vi.hoisted(() => ({ deleteModelImage: vi.fn() }));

vi.mock('@features/models/data/api', () => ({
  deleteModelImage: dependencies.deleteModelImage,
  getModelImageUrl: () => '/model-image.png',
  updateModelImage: vi.fn(),
}));
vi.mock('@features/models/data/modelsStore', () => ({
  markCoverImageChanged: vi.fn(),
  useModelsSelector: (selector: (snapshot: { coverImageVersions: Record<string, number> }) => unknown) =>
    selector({ coverImageVersions: {} }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = { cover_image: 'cover.png', key: 'model-a', name: 'Model A' };
const NOOP = () => {};

/** The real picker trigger on the Launchpad: no workbench, so the tile supplies its own gallery host. */
describe('ModelImageUpload gallery picker', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    dependencies.deleteModelImage.mockReset();
    accountLifecycle.activate('model-image-picker', ':user:model-image-picker');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <ModelImageUpload model={model} onError={NOOP} onUpdated={NOOP} />
          </QueryClientProvider>
        </ChakraProvider>
      );
    });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  // The slot button opens the picker; its name switches between choose and replace with the cover it shows.
  const tile = () => host.querySelector<HTMLElement>('button[aria-haspopup="dialog"]')!;
  const picker = () => document.querySelector('[role="dialog"][aria-label="widgets.gallery.picker.chooseImage"]');

  it('opens the gallery picker from the keyboard', async () => {
    await act(() => tile().focus());
    await userEvent.keyboard('{Enter}');

    await vi.waitFor(() => expect(picker()).not.toBeNull());
    expect(tile().getAttribute('aria-expanded')).toBe('true');
  });

  it('keeps focus on the tile and does not open the picker while the cover is being removed', async () => {
    dependencies.deleteModelImage.mockReturnValue(new Promise(NOOP));
    const remove = host.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.picker.removeImage"]')!;

    await act(() => remove.click());
    await vi.waitFor(() => expect(tile().getAttribute('aria-busy')).toBe('true'));
    // Remove handed focus to the tile; busy must not disable it out from under that focus (fixed up next frame).
    await new Promise(requestAnimationFrame);
    await new Promise(requestAnimationFrame);
    expect(document.activeElement).toBe(tile());

    await act(() => tile().click());

    expect(picker()).toBeNull();
    expect(tile().getAttribute('aria-expanded')).toBe('false');
  });
});
