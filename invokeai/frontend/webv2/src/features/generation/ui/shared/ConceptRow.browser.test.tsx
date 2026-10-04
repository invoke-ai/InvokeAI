import type { GenerateLora } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act, Profiler, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ConceptList, type ConceptModelPort, ConceptRow, type ConceptRowProps } from './ConceptRow';
import { GenerateFieldContextMenu } from './GenerateFieldContextMenu';

const rowCommits = new Map<string, number>();
const countRowCommit = (id: string) => {
  rowCommits.set(id, (rowCommits.get(id) ?? 0) + 1);
};

const STEPS_STYLE = { marginTop: 200 };
const copySteps = () => '30';

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
  if (!host) {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  }
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ConceptList label="Concepts" projectId="project-1">
          {loras.map((lora) => (
            <Profiler key={lora.model.key} id={lora.model.key} onRender={countRowCommit}>
              <ConceptRow
                models={handlers.models ?? MODELS}
                lora={lora}
                onRemove={handlers.onRemove ?? vi.fn()}
                onUpdate={handlers.onUpdate ?? vi.fn()}
              />
            </Profiler>
          ))}
        </ConceptList>
        <GenerateFieldContextMenu copyValue={copySteps}>
          <div data-testid="steps-field" style={STEPS_STYLE}>
            Steps
          </div>
        </GenerateFieldContextMenu>
      </ChakraProvider>
    )
  );

  return host;
};

const render = (lora: GenerateLora, handlers: Handlers = {}) => renderList([lora], handlers);

const menuLabels = () =>
  [...document.querySelectorAll('[role="menu"][data-state="open"] [role="menuitem"]')].map((item) => item.textContent);

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ConceptRow', () => {
  it.each(['remove', 'toggle'] as const)(
    'uses the current row handler for a menu %s after settings change',
    async (action) => {
      const onChange = vi.fn();
      const first = makeLora();
      const second = makeLora({ key: 'lora-2', name: 'Chalk' });
      const renderValues = (loras: GenerateLora[]) =>
        renderList(loras, {
          onRemove: (key) => onChange(loras.filter((lora) => lora.model.key !== key)),
          onUpdate: (key, update) =>
            onChange(loras.map((lora) => (lora.model.key === key ? { ...lora, ...update } : lora))),
        });
      const list = await renderValues([first, second]);
      await act(() =>
        userEvent.click(list.querySelectorAll<HTMLElement>('[data-list-primary]')[1]!, { button: 'right' })
      );
      await expect.poll(() => document.querySelector('[role="menu"]')).not.toBeNull();
      await renderValues([
        { ...first, weight: 1.25 },
        { ...second, isEnabled: false },
      ]);
      const label = `widgets.generate.conceptMenu.${action === 'remove' ? 'remove' : 'enable'}`;
      const menuItem = () =>
        [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find((item) => item.textContent === label);
      await expect.poll(menuItem).toBeDefined();
      await act(() => userEvent.click(menuItem()!));
      expect(onChange).toHaveBeenLastCalledWith(
        action === 'remove' ? [{ ...first, weight: 1.25 }] : [{ ...first, weight: 1.25 }, second]
      );
    }
  );

  it.each(['weight', 'steps'] as const)(
    'hands a right-click between row and %s menus without losing the next menu',
    async (field) => {
      const list = await render(makeLora());
      const primary = list.querySelector<HTMLElement>('[data-list-primary]')!;
      const control = list.querySelector<HTMLElement>(
        field === 'weight' ? '[data-scope="scrubber"]' : '[data-testid="steps-field"]'
      )!;
      await act(() => userEvent.click(primary, { button: 'right' }));
      await expect.poll(menuLabels).toContain('widgets.generate.conceptMenu.remove');
      await act(() => userEvent.click(control, { button: 'right', position: { x: 8, y: 8 } }));
      await expect.poll(menuLabels).toContain('widgets.generate.copyValue');
      expect(document.querySelectorAll('[role="menu"][data-state="open"]')).toHaveLength(1);
      await act(() => userEvent.click(primary, { button: 'right' }));
      await expect.poll(menuLabels).toContain('widgets.generate.conceptMenu.remove');
      expect(document.querySelectorAll('[role="menu"][data-state="open"]')).toHaveLength(1);
    }
  );

  it('keeps a drag local to its row and commits its final weight once', async () => {
    const onUpdate = vi.fn();
    const list = await renderList([makeLora(), makeLora({ key: 'lora-2', name: 'Chalk' })], { onUpdate });
    const frame = list.querySelector<HTMLElement>('[data-scope="scrubber"]')!;
    const rect = frame.getBoundingClientRect();
    const pointer = (target: EventTarget, type: string, fraction: number) =>
      act(() => {
        target.dispatchEvent(
          new PointerEvent(type, {
            bubbles: true,
            button: 0,
            clientX: rect.left + 8 + fraction * (rect.width - 16),
            pointerId: 1,
          })
        );
      });

    await pointer(frame, 'pointerdown', 0.2);
    rowCommits.clear();
    for (let step = 1; step <= 10; step += 1) {
      await pointer(window, 'pointermove', 0.2 + step * 0.05);
    }
    await pointer(window, 'pointerup', 0.7);
    const finalWeight = Number(frame.querySelector('[role="slider"]')!.getAttribute('aria-valuenow'));

    expect(finalWeight).not.toBe(0.75);
    expect(rowCommits.get('lora-1')).toBeGreaterThan(0);
    expect(rowCommits.get('lora-2') ?? 0).toBe(0);
    expect(onUpdate).not.toHaveBeenCalled();
    await expect.poll(() => onUpdate.mock.calls).toEqual([['lora-1', { weight: finalWeight }]]);
    await act(() => flushWorkbenchDrafts());
    expect(onUpdate).toHaveBeenCalledTimes(1);
  });

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
    expect(onUpdate).toHaveBeenCalledTimes(1);
    await act(() => flushWorkbenchDrafts());
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
    await expect.poll(() => onUpdate.mock.lastCall).toEqual(['lora-1', { weight: 0.75 }]);

    await openMenu();
    await choose('widgets.generate.conceptMenu.disable');
    expect(onUpdate).toHaveBeenLastCalledWith('lora-1', { isEnabled: false });

    await openMenu();
    await choose('widgets.generate.conceptMenu.remove');
    expect(onRemove).toHaveBeenCalledWith('lora-1');
  });

  it('animates the row menu out with its actions instead of unmounting them on close', async () => {
    const row = await render(makeLora());
    await act(() => userEvent.click(row.querySelector<HTMLElement>('[data-list-primary]')!, { button: 'right' }));
    await expect.poll(menuLabels).toContain('widgets.generate.conceptMenu.remove');
    const menu = document.querySelector('[role="menu"]')!;

    const frames = closingFrames(await recordDialogExit(menu, () => act(() => userEvent.keyboard('{Escape}'))));
    await expect.poll(() => document.querySelector('[role="menu"]')).toBeNull();

    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toContain('widgets.generate.conceptMenu.remove');
    }
  });

  it('does not open row actions on a pointer click', async () => {
    const row = await render(makeLora());

    await act(() => userEvent.click(row.querySelector<HTMLElement>('[data-list-primary]')!));

    expect(document.querySelector('[role="menu"][data-state="open"]')).toBeNull();
    expect(row.querySelector('[data-list-primary]')?.getAttribute('aria-expanded')).toBe('false');
  });

  it.each(['{Enter}', ' '])('opens row actions on %s and restores focus on Escape', async (key) => {
    const row = await render(makeLora());
    const primary = row.querySelector<HTMLElement>('[data-list-primary]')!;

    expect(primary.getAttribute('aria-haspopup')).toBe('menu');
    primary.focus();
    await act(() => userEvent.keyboard(key));
    await expect.poll(menuLabels).toContain('widgets.generate.conceptMenu.remove');
    expect(primary.getAttribute('aria-expanded')).toBe('true');
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(() => document.activeElement).toBe(primary);
    expect(primary.getAttribute('aria-expanded')).toBe('false');
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

  it('hands focus to the picker the last row was excluded from once that row is removed', async () => {
    // A picker whose only candidate is applied is disabled until the removal renders, and the control before the
    // whole section must not take focus in its place.
    const Section = () => {
      const [loras, setLoras] = useState([makeLora()]);
      const removeAll = useCallback(() => setLoras([]), []);

      return (
        <ChakraProvider value={system}>
          <button type="button">Before the section</button>
          <div>
            <button disabled={loras.length > 0} type="button">
              Add concept
            </button>
            {loras.length > 0 ? (
              <ConceptList label="Concepts" projectId="project-1">
                {loras.map((lora) => (
                  <ConceptRow
                    key={lora.model.key}
                    models={MODELS}
                    lora={lora}
                    onRemove={removeAll}
                    onUpdate={vi.fn()}
                  />
                ))}
              </ConceptList>
            ) : null}
            <button type="button">After the list</button>
          </div>
        </ChakraProvider>
      );
    };
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => root?.render(<Section />));
    const remove = host.querySelector<HTMLElement>('[aria-label="widgets.generate.removeConceptNamed"]');

    remove!.focus();
    await act(() => userEvent.click(remove!));

    await expect.poll(() => document.activeElement?.textContent).toBe('Add concept');
  });
});
