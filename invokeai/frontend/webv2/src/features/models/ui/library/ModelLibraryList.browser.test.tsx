import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DEFAULT_LIBRARY_FILTERS, type ModelLibraryFilters } from '@features/models/core/library';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

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

describe('ModelLibraryList keyboard context menu', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    accountLifecycle.activate('library-menu-test-a', ':user:library-menu-test-a');
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

  it('opens the row menu from the keyboard and returns focus to the row when it closes', async () => {
    await act(() => root.render(<Harness filters={DEFAULT_LIBRARY_FILTERS} />));
    await act(async () => {
      await Promise.resolve();
    });

    const row = host.querySelector<HTMLElement>('[data-list-row="main-1"]')!;
    const primary = row.querySelector<HTMLButtonElement>('[data-list-primary]')!;

    await act(() => primary.focus());
    // Keyboard-invoked context menus arrive as a contextmenu event with no pointer position.
    await act(() => {
      row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 0, clientY: 0 }));
    });
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).not.toBeNull());
    await vi.waitFor(() =>
      expect(document.querySelector('[role="menu"]')!.contains(document.activeElement)).toBe(true)
    );

    await userEvent.keyboard('{Escape}');
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).toBeNull());
    expect(document.activeElement).toBe(primary);
  });
});

describe('ModelLibraryList pinned group header', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    accountLifecycle.activate('library-pin-test-a', ':user:library-pin-test-a');
    // Enough rows in both groups that the second group's header can reach the
    // top of an 800px viewport with the first group still inside the
    // virtualizer's overscan window.
    setModelsSnapshotForTests({
      models: [
        ...Array.from({ length: 20 }, (_, i) => model(`main-${i}`, 'main')),
        ...Array.from({ length: 20 }, (_, i) => model(`lora-${i}`, 'lora')),
      ],
      status: 'loaded',
    });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  const settleFrame = async () => {
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      await new Promise((resolve) => {
        requestAnimationFrame(() => resolve(undefined));
      });
    });
  };

  const pinnedHeaderText = () => host.querySelector<HTMLElement>('[data-list-pinned-header]')?.textContent ?? '';

  it('swaps the pinned header when a group scrolls under it, not an overscan later', async () => {
    await act(() => root.render(<Harness filters={DEFAULT_LIBRARY_FILTERS} />));
    await settleFrame();

    expect(host.querySelector('[role="list"][aria-label="models.library"]')).not.toBeNull();
    const viewport = host.querySelector<HTMLElement>('[data-list-viewport]');

    expect(viewport).not.toBeNull();
    expect(pinnedHeaderText()).toContain('Main Models');

    const scrollTo = async (top: number) => {
      await act(async () => {
        viewport!.scrollTop = top;
        viewport!.dispatchEvent(new Event('scroll'));
        await new Promise((resolve) => {
          requestAnimationFrame(() => resolve(undefined));
        });
      });
    };

    // Rows: main header (32px) + 20 main rows (56px each) put the LoRAs header
    // at 1152px; one row past it, the first visible row is a lora.
    await scrollTo(1208);
    expect(pinnedHeaderText()).toContain('LoRAs');

    await scrollTo(0);
    expect(pinnedHeaderText()).toContain('Main Models');
  });
});
