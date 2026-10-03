import type { GenerateLora } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ConceptList, type ConceptModelPort, ConceptRow, type ConceptRowProps } from './ConceptRow';

const PIXEL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+XcX1WQAAAABJRU5ErkJggg==';
const openInModelManager = vi.fn();
const MODELS: ConceptModelPort = {
  getBaseColorPalette: () => 'gray',
  getBaseLabel: (base) => base,
  getImageUrl: () => PIXEL,
  openInModelManager,
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const makeLora = (model: Partial<GenerateLora['model']> = {}): GenerateLora => ({
  isEnabled: true,
  model: { base: 'sdxl', key: 'lora-1', name: 'Ink Wash', type: 'lora', ...model },
  weight: 0.75,
});

type Handlers = Partial<Pick<ConceptRowProps, 'models' | 'onRemove' | 'onUpdate'>>;

const renderList = async (loras: GenerateLora[], handlers: Handlers = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ConceptList label="Concepts">
          {loras.map((lora) => (
            <ConceptRow
              key={lora.model.key}
              models={handlers.models ?? MODELS}
              lora={lora}
              onRemove={handlers.onRemove ?? vi.fn()}
              onUpdate={handlers.onUpdate ?? vi.fn()}
            />
          ))}
        </ConceptList>
      </ChakraProvider>
    )
  );

  return host;
};

const render = (lora: GenerateLora, handlers: Handlers = {}) => renderList([lora], handlers);

const menuLabels = () => [...document.querySelectorAll('[role="menuitem"]')].map((item) => item.textContent);

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ConceptRow', () => {
  it('shows the model cover as its thumbnail', async () => {
    const row = await render(makeLora({ cover_image: 'cover.png' }));

    await expect.poll(() => row.querySelector('img')?.getAttribute('data-state')).toBe('visible');
    expect(row.querySelector('img')?.getAttribute('src')).toBe(PIXEL);
  });

  it('falls back to a model icon without a cover', async () => {
    const row = await render(makeLora());

    expect(row.querySelector('img')).toBeNull();
    expect(row.querySelector('.lucide-box')).not.toBeNull();
  });

  it('reports toggles, weight steps, and removal against the model key', async () => {
    const onUpdate = vi.fn();
    const onRemove = vi.fn();
    const row = await render(makeLora(), { onRemove, onUpdate });

    await act(() => userEvent.click(row.querySelector<HTMLElement>('[data-part="control"]')!));
    expect(onUpdate).toHaveBeenLastCalledWith('lora-1', { isEnabled: false });

    row.querySelector<HTMLElement>('[role="slider"]')!.focus();
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(onUpdate).toHaveBeenLastCalledWith('lora-1', { weight: 0.8 });

    await act(() =>
      userEvent.click(row.querySelector<HTMLElement>('[aria-label="widgets.generate.removeConceptNamed"]')!)
    );
    expect(onRemove).toHaveBeenCalledWith('lora-1');
  });

  it('offers model, weight, toggle, and removal actions from the row context menu', async () => {
    const onUpdate = vi.fn();
    const onRemove = vi.fn();
    const row = await render({ ...makeLora(), weight: 1.25 }, { onRemove, onUpdate });
    const openMenu = async () => {
      await act(() => userEvent.click(row.querySelector<HTMLElement>('[data-list-primary]')!, { button: 'right' }));
      await expect.poll(() => document.querySelector('[role="menu"]')).not.toBeNull();
    };
    const choose = async (label: string) => {
      const item = [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
        (candidate) => candidate.textContent === label
      );
      await act(() => userEvent.click(item!));
    };

    await openMenu();
    await choose('widgets.generate.conceptMenu.openInModelManager');
    expect(openInModelManager).toHaveBeenCalledWith('lora-1');

    await openMenu();
    await choose('widgets.generate.resetToConceptDefault');
    expect(onUpdate).toHaveBeenLastCalledWith('lora-1', { weight: 0.75 });

    await openMenu();
    await choose('widgets.generate.conceptMenu.disable');
    expect(onUpdate).toHaveBeenLastCalledWith('lora-1', { isEnabled: false });

    await openMenu();
    await choose('widgets.generate.conceptMenu.remove');
    expect(onRemove).toHaveBeenCalledWith('lora-1');
  });

  it('opens the menu from a right-click but not from a normal click', async () => {
    const row = await render(makeLora());
    const primary = row.querySelector<HTMLElement>('[data-list-primary]')!;

    await act(() => userEvent.click(primary));
    expect(document.querySelector('[role="menu"]')).toBeNull();

    await act(() => userEvent.click(primary, { button: 'right' }));
    await expect.poll(() => document.querySelector('[role="menu"]')).not.toBeNull();
  });

  it('moves the open menu to a second row right-clicked while the first is open', async () => {
    const list = await renderList([
      makeLora(),
      makeLora({ key: 'lora-2', name: 'Charcoal' }),
      makeLora({ key: 'lora-3', name: 'Pastel' }),
    ]);
    const rows = list.querySelectorAll<HTMLElement>('[data-list-primary]');
    const openRow = () => list.querySelector('[data-menu-open] [data-list-primary]');

    await act(() => userEvent.click(rows[0]!, { button: 'right' }));
    await expect.poll(openRow).toBe(rows[0]);
    await act(() => userEvent.click(rows[2]!, { button: 'right' }));
    await expect.poll(openRow).toBe(rows[2]);
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });

    expect(document.querySelectorAll('[role="menu"][data-state="open"]')).toHaveLength(1);
    expect(openRow()).toBe(rows[2]);
  });

  it('omits the Model Manager link when the session may not manage models', async () => {
    const row = await render(makeLora(), { models: { ...MODELS, openInModelManager: undefined } });

    await act(() => userEvent.click(row.querySelector<HTMLElement>('[data-list-primary]')!, { button: 'right' }));
    await expect.poll(() => document.querySelector('[role="menu"]')).not.toBeNull();

    expect(menuLabels()).not.toContain('widgets.generate.conceptMenu.openInModelManager');
    expect(menuLabels()).toContain('widgets.generate.conceptMenu.remove');
  });

  it('moves keyboard focus to the neighbouring row when a row is removed', async () => {
    const onRemove = vi.fn();
    const list = await renderList([makeLora(), makeLora({ key: 'lora-2', name: 'Charcoal' })], { onRemove });
    const [first, second] = list.querySelectorAll<HTMLElement>('[data-list-primary]');

    // A keyboard-invoked context menu (Shift+F10 or the Menu key) reports no pointer position.
    first!.focus();
    await act(() => {
      first!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.poll(() => document.querySelector('[role="menu"]')).not.toBeNull();
    const remove = [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
      (item) => item.textContent === 'widgets.generate.conceptMenu.remove'
    );
    await act(() => userEvent.click(remove!));

    expect(onRemove).toHaveBeenCalledWith('lora-1');
    await expect.poll(() => document.activeElement).toBe(second);
  });
});
