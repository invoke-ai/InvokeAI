/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-object-as-prop -- Test fixtures deliberately vary for each render. */
import type { StarterModel } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { StarterList } from './StarterList';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SOURCE = 'https://example.com/model.safetensors';
const PAGE = 'https://example.com/model';
const noop = () => undefined;

const contextMenuEvent = () =>
  new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 100, clientY: 100 });

describe('starter row source context menus', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    accountLifecycle.activate('starter-list-test', ':user:starter-list-test');
    setModelsSnapshotForTests({ models: [], status: 'loaded' });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    setModelsSnapshotForTests({ models: [], status: 'idle' });
    accountLifecycle.invalidate();
  });

  const render = (source_urls: string[]) => {
    const model: StarterModel = {
      base: 'sdxl',
      description: 'Example model',
      is_installed: false,
      name: 'Example',
      source: SOURCE,
      source_urls,
      type: 'main',
    };
    const models = [model];
    const response = { starter_bundles: {}, starter_models: models };
    return act(() =>
      root.render(
        <ChakraProvider value={system}>
          <StarterList
            configuredExternalProviders={new Set()}
            isInstallable
            loadError={null}
            models={models}
            onConfigureExternalProvider={noop}
            onInstall={noop}
            pendingSources={new Set()}
            response={response}
            selectedBundleSources={undefined}
            status="loaded"
          />
        </ChakraProvider>
      )
    );
  };

  it('keeps native link actions while row right-click still opens the install menu', async () => {
    await render([PAGE]);
    const link = host.querySelector('a')!;
    const linkEvent = contextMenuEvent();
    await act(() => link.dispatchEvent(linkEvent));
    expect(linkEvent.defaultPrevented).toBe(false);
    expect(document.querySelector('[role="menuitem"]')).toBeNull();

    const rowEvent = contextMenuEvent();
    await act(() => host.querySelector('[data-list-row]')!.dispatchEvent(rowEvent));
    expect(rowEvent.defaultPrevented).toBe(true);
    await expect.poll(() => document.querySelector('[role="menuitem"]')?.textContent).toContain('models.install');
  });

  it('keeps native link actions for links in the source menu', async () => {
    await render([PAGE, 'https://example.org/quantized']);
    const trigger = host.querySelector<HTMLButtonElement>('button[aria-label="models.visitModelSource"]')!;
    await userEvent.click(trigger);
    await expect.poll(() => document.querySelectorAll('a[role="menuitem"]').length).toBe(2);
    const linkEvent = contextMenuEvent();
    await act(() => document.querySelector('a[role="menuitem"]')!.dispatchEvent(linkEvent));
    expect(linkEvent.defaultPrevented).toBe(false);
    expect(document.querySelector('[role="menuitem"][data-value="install"]')).toBeNull();
    expect(host.querySelector('[data-list-row]')?.hasAttribute('data-menu-open')).toBe(false);
  });
});
