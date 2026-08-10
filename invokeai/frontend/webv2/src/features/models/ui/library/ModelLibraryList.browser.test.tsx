import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DEFAULT_LIBRARY_FILTERS, type ModelLibraryFilters } from '@features/models/core/library';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelLibraryList } from './ModelLibraryList';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = (key: string, type: string): ModelConfig =>
  ({
    base: 'sdxl',
    file_size: 1024,
    format: 'checkpoint',
    hash: `hash-${key}`,
    key,
    name: `Model ${key}`,
    path: `/models/${key}.safetensors`,
    source: `/models/${key}.safetensors`,
    source_type: 'path',
    type,
  }) as ModelConfig;

const library: ModelConfig[] = [
  ...Array.from({ length: 6 }, (_, i) => model(`main-${i}`, 'main')),
  ...Array.from({ length: 3 }, (_, i) => model(`lora-${i}`, 'lora')),
];

const noop = () => undefined;

const Harness = ({ filters }: { filters: ModelLibraryFilters }) => (
  <ChakraProvider value={system}>
    <div style={{ display: 'flex', flexDirection: 'column', height: '800px', width: '360px' }}>
      <ModelLibraryList
        activeModelKey={null}
        filters={filters}
        instanceId="manager"
        selectedKeys={new Set<string>()}
        onActivate={noop}
        onToggleSelected={noop}
      />
    </div>
  </ChakraProvider>
);

describe('ModelLibraryList filter transitions', () => {
  let host: HTMLDivElement;
  let root: Root;

  const renderWithFilters = async (filters: ModelLibraryFilters) => {
    await act(async () => {
      root.render(<Harness filters={filters} />);
      // Flush the deferred filters and the virtualizer's measurement passes.
      await Promise.resolve();
    });
    await act(async () => {
      await new Promise((resolve) => {
        requestAnimationFrame(() => resolve(undefined));
      });
    });
  };

  const renderedModelNames = () =>
    [...host.querySelectorAll<HTMLButtonElement>('button[type="button"]')].map(
      (button) => button.textContent.match(/Model [a-z]+-\d+/)?.[0] ?? ''
    );

  beforeEach(() => {
    accountLifecycle.activate('library-list-test-a', ':user:library-list-test-a');
    setModelsSnapshotForTests({ models: library, status: 'loaded' });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('restores every row after narrowing to a type and resetting to all', async () => {
    await renderWithFilters(DEFAULT_LIBRARY_FILTERS);
    expect(renderedModelNames()).toHaveLength(9);

    await renderWithFilters({ ...DEFAULT_LIBRARY_FILTERS, typeFilter: 'lora' });
    expect(renderedModelNames()).toEqual(['Model lora-0', 'Model lora-1', 'Model lora-2']);

    await renderWithFilters(DEFAULT_LIBRARY_FILTERS);
    const names = renderedModelNames();

    expect(names).toHaveLength(9);
    expect(names).toContain('Model main-5');
    expect(names).toContain('Model lora-2');
  });
});
