import type { CanvasBlendMode, CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { getDocumentLayer } from '@workbench/canvas-engine/api';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { commitPreparedEdit } from '@workbench/widgets/canvas/useStructuralCommit';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({
  listeners: new Set<() => void>(),
  project: null as Project | null,
}));

vi.mock('@workbench/WorkbenchContext', async () => {
  const { useSyncExternalStore } = await import('react');
  const subscribe = (listener: () => void) => {
    harness.listeners.add(listener);
    return () => harness.listeners.delete(listener);
  };
  return {
    useActiveProjectSelector: (selector: (project: unknown) => unknown) =>
      selector(useSyncExternalStore(subscribe, () => harness.project)),
    useOptionalWorkbenchCommands: () => null,
  };
});
const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));

import { LayerBlendRow } from './LayerBlendRow';

const i18n = createInstance();
beforeAll(async () => {
  const translation = (await (await fetch('/locales/en.json')).json()) as Record<string, unknown>;
  await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** The structural engine over the real reducer and history; previews flush synchronously, as a settled frame would. */
let stub: StructuralEngineStub;
let unsubscribe: () => void = () => undefined;

const publish = (): void => {
  harness.project = stub.project();
  harness.listeners.forEach((listener) => listener());
};

const createStub = (selectedLayerId: string | null = 'r1'): void => {
  unsubscribe();
  stub = createStructuralEngineStub({
    layers: [layerContract('r1'), layerContract('r2')],
    schedulePreview: (flush) => {
      flush();
      return () => undefined;
    },
    selectedLayerId,
  });
  unsubscribe = stub.subscribe(publish);
  publish();
};

/** Records a blend-mode step from elsewhere (a hotkey, a script), as the user's history to undo. */
const commitBlendMode = (id: string, blendMode: CanvasBlendMode): void => {
  const outcome = commitPreparedEdit(stub.engine, 'Blend mode', (model) =>
    model.prepare({ id, patch: { blendMode }, type: 'patch' })
  );
  expect(outcome).toEqual({ status: 'committed' });
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/** The row's engine surface: the stub's structural half plus an unlocked interaction store. */
const rowEngine = () => ({
  ...stub.engine,
  exports: {},
  interaction: { get: () => false, subscribe: () => () => undefined },
});

const mount = async (width = 280): Promise<void> => {
  host = document.createElement('div');
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);
  const engine = rowEngine();
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <LayerBlendRow engine={engine as never} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const unmount = async (): Promise<void> => {
  await act(() => root?.unmount());
  root = null;
};

const layer = (id = 'r1'): CanvasLayerContract => getDocumentLayer(stub.document(), id)!;
const trigger = () => page.getByRole('combobox', { name: 'Blend mode' });
const option = (name: string) => page.getByRole('option', { exact: true, name });
const commits = () => stub.commits;

beforeEach(() => {
  createStub();
});

afterEach(async () => {
  await unmount();
  host?.remove();
  host = null;
  vi.clearAllMocks();
  // Prototype spies (pointer lock) must not leak into later tests or files sharing the browser.
  vi.restoreAllMocks();
});

describe('blend mode preview', () => {
  it('previews the hovered mode and restores the original when the menu closes without a choice', async () => {
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Multiply'));

    await expect.poll(() => layer().blendMode).toBe('multiply');
    // The trigger and checked option keep naming the committed mode while the canvas previews another.
    await expect.element(trigger()).toHaveTextContent('Normal');
    await expect.element(option('Normal')).toHaveAttribute('aria-selected', 'true');

    await userEvent.hover(option('Screen'));
    await expect.poll(() => layer().blendMode).toBe('screen');

    await userEvent.keyboard('{Escape}');

    await expect.poll(() => layer().blendMode).toBe('normal');
    expect(commits()).toHaveLength(0);
    expect(stub.history.canUndo()).toBe(false);
    expect(notify.error).not.toHaveBeenCalled();
  });

  it('previews the keyboard highlight and records one step from the original when it is chosen', async () => {
    await mount();
    await userEvent.click(trigger());
    await expect.element(option('Normal')).toBeVisible();
    await userEvent.keyboard('{ArrowDown}{ArrowDown}');

    await expect.poll(() => layer().blendMode).toBe('screen');

    await userEvent.keyboard('{Enter}');

    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ patch: { blendMode: 'screen' } });
    expect(commits()[0]!.inverse).toMatchObject({ patch: { blendMode: 'normal' } });
    await expect.element(trigger()).toHaveTextContent('Screen');
    expect(layer().blendMode).toBe('screen');
    await stub.engine.history.undo();
    expect(layer().blendMode).toBe('normal');
  });

  it('records a clicked option once, even after previewing others on the way', async () => {
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Multiply'));
    await userEvent.hover(option('Overlay'));
    await userEvent.click(option('Overlay'));

    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.inverse).toMatchObject({ id: 'r1', patch: { blendMode: 'normal' } });
    expect(layer().blendMode).toBe('overlay');
    expect(stub.history.entries().past).toEqual(['Blend mode']);
  });

  it('restores the original when the menu unmounts or the selection moves while it is open', async () => {
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Darken'));
    await expect.poll(() => layer().blendMode).toBe('darken');

    await act(() => {
      stub.ctx.dispatch({ id: 'r2', type: 'setCanvasSelectedLayer' }, 'system');
    });

    expect(layer('r1').blendMode).toBe('normal');
    expect(page.getByRole('option').query()).toBeNull();

    await userEvent.click(trigger());
    await userEvent.hover(option('Lighten'));
    await expect.poll(() => layer('r2').blendMode).toBe('lighten');
    await unmount();

    expect(layer('r2').blendMode).toBe('normal');
    expect(commits()).toHaveLength(0);
  });

  it('keeps an undo that lands while a mode is hovered: Escape restores nothing over it', async () => {
    commitBlendMode('r1', 'screen');
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Multiply'));
    await expect.poll(() => layer().blendMode).toBe('multiply');

    await act(() => stub.engine.history.undo());

    expect(layer().blendMode).toBe('normal');

    await userEvent.keyboard('{Escape}');
    await expect.poll(() => page.getByRole('option').query()).toBeNull();

    // The undone step stays undone: the document, the undo stack and the redo stack agree.
    expect(layer().blendMode).toBe('normal');
    expect(stub.history.canUndo()).toBe(false);
    expect(stub.history.canRedo()).toBe(true);
    expect(notify.error).not.toHaveBeenCalled();
  });

  it('records a choice made after an undo landed mid-menu from the undone mode, not the one the menu opened on', async () => {
    commitBlendMode('r1', 'screen');
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Multiply'));
    await expect.poll(() => layer().blendMode).toBe('multiply');
    await act(() => stub.engine.history.undo());

    await userEvent.click(option('Multiply'));

    expect(layer().blendMode).toBe('multiply');
    expect(commits().at(-1)!.inverse).toMatchObject({ id: 'r1', patch: { blendMode: 'normal' } });
    await stub.engine.history.undo();
    expect(layer().blendMode).toBe('normal');
  });

  it('names the mode an undo landed on under the open menu, and records a choice of the pre-undo mode from it', async () => {
    commitBlendMode('r1', 'screen');
    await mount();
    await userEvent.click(trigger());
    await expect.element(trigger()).toHaveTextContent('Screen');
    await userEvent.hover(option('Multiply'));
    await expect.poll(() => layer().blendMode).toBe('multiply');
    await act(() => stub.engine.history.undo());

    await expect.element(trigger()).toHaveTextContent('Normal');
    await expect.element(option('Normal')).toHaveAttribute('aria-selected', 'true');

    await userEvent.click(option('Screen'));

    expect(layer().blendMode).toBe('screen');
    expect(commits().at(-1)!.inverse).toMatchObject({ id: 'r1', patch: { blendMode: 'normal' } });
  });

  it('ends a hovered preview before a delete from elsewhere, so undoing the delete restores the committed mode', async () => {
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Multiply'));
    await expect.poll(() => layer().blendMode).toBe('multiply');

    await act(() => {
      const outcome = commitPreparedEdit(stub.engine, 'Delete', (model) =>
        model.prepare({ ids: ['r1'], type: 'remove' })
      );
      expect(outcome).toEqual({ status: 'committed' });
    });

    expect(getDocumentLayer(stub.document(), 'r1')).toBeNull();
    await stub.engine.history.undo();
    expect(layer('r1').blendMode).toBe('normal');
    expect(stub.history.entries()).toEqual({ future: ['Delete'], past: [] });
  });
});

describe('opacity input', () => {
  const opacityInput = () => page.getByRole('spinbutton', { exact: true, name: 'Opacity' });

  it('records a held arrow key as one step from where it started', async () => {
    await mount();
    await act(async () => {
      (opacityInput().element() as HTMLInputElement).focus();
      await userEvent.keyboard('{ArrowDown>4/}');
    });

    expect(layer().opacity).toBeCloseTo(0.96, 5);
    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.96 } });
    expect(commits()[0]!.inverse).toMatchObject({ id: 'r1', patch: { opacity: 1 } });
  });

  it('scrubs from its drag handle and records the result once when the mouse button lifts', async () => {
    await mount();
    // The scrubber locks the pointer once a real click has activated the page; the harness lock steals focus.
    vi.spyOn(Element.prototype, 'requestPointerLock').mockImplementation(() => Promise.resolve());
    const scrubber = host!.querySelector<HTMLElement>('[data-scope="number-input"][data-part="scrubber"]')!;
    await act(() =>
      scrubber.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 100, clientY: 100 }))
    );
    for (let step = 1; step <= 5; step += 1) {
      await act(() =>
        document.dispatchEvent(
          new MouseEvent('mousemove', { bubbles: true, clientX: 100 - step * 4, clientY: 100, movementX: -4 })
        )
      );
    }

    // Each step previews on the layer; nothing is recorded until the release.
    expect(layer().opacity).toBeCloseTo(0.95, 5);
    expect(commits()).toHaveLength(0);
    await act(() =>
      document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, button: 0, clientX: 80, clientY: 100 }))
    );
    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.95 } });
    expect(commits()[0]!.inverse).toMatchObject({ id: 'r1', patch: { opacity: 1 } });
  });

  it('previews a typed value and records it once on Enter', async () => {
    await mount();
    await userEvent.tripleClick(opacityInput());
    await userEvent.keyboard('45');

    expect(layer().opacity).toBeCloseTo(0.45, 5);
    expect(commits()).toHaveLength(0);

    await userEvent.keyboard('{Enter}');

    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ patch: { opacity: 0.45 } });
    expect(commits()[0]!.inverse).toMatchObject({ patch: { opacity: 1 } });
  });

  it('records a still-pending edit when the row unmounts', async () => {
    await mount();
    await userEvent.tripleClick(opacityInput());
    await userEvent.keyboard('30');
    expect(commits()).toHaveLength(0);

    await unmount();

    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.3 } });
  });

  it('disables both controls without an editable selection', async () => {
    createStub(null);
    await mount();

    await expect.element(trigger()).toBeDisabled();
    await expect.element(opacityInput()).toBeDisabled();
  });
});
