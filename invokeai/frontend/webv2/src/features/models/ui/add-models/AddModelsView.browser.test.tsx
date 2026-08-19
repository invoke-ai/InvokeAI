import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AddModelsView } from './AddModelsView';

/**
 * The Add Models box is local state seeded once from the models UI store, so a
 * requirement link elsewhere in the app can open this view already searching —
 * without that search outliving the view the way account-scoped store state
 * would. This suite owns exactly that handover; everything else the view does
 * is stubbed out.
 */

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

vi.mock('@features/models/data/startersStore', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureStartersLoaded: vi.fn(),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ error: null, response: { starter_bundles: {}, starter_models: [] }, status: 'loaded' }),
}));

vi.mock('@features/models/data/externalProvidersStore', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureExternalProvidersLoaded: () => Promise.resolve(),
  useExternalProvidersSelector: (selector: (snapshot: unknown) => unknown) => selector({ configs: [] }),
}));

// Spread the original: sibling modules in this view's graph import other
// members of it, and a narrow factory would break their imports outright.
vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getHuggingFaceModels: vi.fn(),
  scanFolderForModels: vi.fn(),
}));

vi.mock('./useInstallActions', () => ({
  useInstallActions: () => ({ install: vi.fn(), installMany: vi.fn(), pendingSources: new Set() }),
}));

vi.mock('@features/models/ui/useModelsNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('AddModelsView search seed', () => {
  let host: HTMLDivElement;
  let root: Root;

  const searchBox = () => document.querySelector<HTMLInputElement>('input[aria-label="models.searchOrAdd"]');

  const mount = async () => {
    root = createRoot(host);

    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <AddModelsView />
          </ChakraProvider>
        </StrictMode>
      );
      // Chakra's observer-driven commits land a task after the render that
      // armed them; awaiting inside this scope keeps them in it.
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };

  const unmount = async () => {
    await act(() => root.unmount());
  };

  beforeEach(async () => {
    host = document.createElement('div');
    document.body.append(host);

    const store = await import('@features/models/ui/uiStore');
    store.clearAddModelsSeed();
  });

  afterEach(() => {
    host.remove();
  });

  it('opens searching for a seeded model, then forgets it on the next open', async () => {
    const { requestAddModelsSearch } = await import('@features/models/ui/uiStore');

    requestAddModelsSearch('Wan 2.2 I2V A14B');
    await mount();

    // StrictMode double-invokes the `useState` initializer, so the read has to
    // survive being run twice — this is the case that catches a take-and-clear
    // initializer.
    expect(searchBox()?.value).toBe('Wan 2.2 I2V A14B');

    await unmount();
    await mount();

    // A tab switch used to reset the box, and still does: the seed was
    // one-shot, not account-lived state that keeps filtering hours later.
    expect(searchBox()?.value).toBe('');

    await unmount();
  });

  it('opens empty when nothing asked it to search', async () => {
    await mount();

    expect(searchBox()?.value).toBe('');

    await unmount();
  });
});
