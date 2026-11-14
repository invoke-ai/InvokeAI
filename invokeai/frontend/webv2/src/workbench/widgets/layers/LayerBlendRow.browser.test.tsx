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

/** The row's engine surface: the stub's structural half plus an interaction store, unlocked unless `locked`. */
const rowEngine = (locked: boolean) => ({
  ...stub.engine,
  exports: {},
  interaction: { get: () => locked, subscribe: () => () => undefined },
});

const mount = async (width = 280, locked = false): Promise<void> => {
  host = document.createElement('div');
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);
  const engine = rowEngine(locked);
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

const afterFrames = async (count: number): Promise<void> => {
  for (let frame = 0; frame < count; frame += 1) {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve());
    });
  }
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

  it('shows the undone mode, not the one the menu opened on, when the highlight leaves the list after an undo', async () => {
    commitBlendMode('r1', 'screen');
    const committed = commits().length;
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Multiply'));
    await expect.poll(() => layer().blendMode).toBe('multiply');
    await act(() => stub.engine.history.undo());
    expect(layer().blendMode).toBe('normal');

    await userEvent.unhover(option('Multiply'));
    await userEvent.hover(trigger());

    // Nothing was previewed on the undone document, so leaving the options previews nothing over it.
    await new Promise((resolve) => {
      setTimeout(resolve, 50);
    });
    expect(layer().blendMode).toBe('normal');
    expect(commits()).toHaveLength(committed);
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
    expect(stub.history.entries()).toEqual({ evicted: 0, future: ['Delete'], past: [] });
  });
});

describe('opacity input', () => {
  const opacityInput = () => page.getByRole('spinbutton', { exact: true, name: 'Opacity' });
  const sliderTrigger = () => page.getByRole('button', { exact: true, name: 'Opacity slider' });
  const slider = () => page.getByRole('slider', { exact: true, name: 'Opacity' });

  /** Opens the dropdown slider; `pointer` builds a mouse event at a fraction of the track. */
  const openSlider = async () => {
    await userEvent.click(sliderTrigger());
    await expect.element(slider()).toBeVisible();
    const control = document.querySelector<HTMLElement>('[data-scope="slider"][data-part="control"]')!;
    const { height, left, top, width } = control.getBoundingClientRect();
    const pointer = (type: string, fraction: number) =>
      new PointerEvent(type, {
        bubbles: true,
        button: 0,
        buttons: type === 'pointerup' ? 0 : 1,
        clientX: left + width * fraction,
        clientY: top + height / 2,
        isPrimary: true,
        pointerId: 1,
        pointerType: 'mouse',
      });
    return { control, pointer };
  };

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

  it('keeps a typed value pending while the caret moves, and records only the final value', async () => {
    await mount();
    const input = opacityInput().element() as HTMLInputElement;
    // Zag restores the caret one frame after each value change, so a caret move made sooner is undone. Let each
    // keystroke settle before moving the caret, as a person typing would.
    const typeSettled = async (text: string) => {
      await userEvent.keyboard(text);
      await act(() => afterFrames(2));
    };
    await userEvent.tripleClick(opacityInput());
    await typeSettled('5');
    await typeSettled('{ArrowLeft}');
    expect(input.selectionStart).toBe(0);
    await typeSettled('2');
    await expect.element(opacityInput()).toHaveValue('25');
    await typeSettled('{ArrowRight}');
    expect(input.selectionStart).toBe(2);

    expect(layer().opacity).toBeCloseTo(0.25, 5);
    expect(commits()).toHaveLength(0);

    await userEvent.keyboard('{Enter}');

    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.25 } });
    expect(commits()[0]!.inverse).toMatchObject({ id: 'r1', patch: { opacity: 1 } });
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

  it('previews each step of a slider drag and records the result once when the pointer lifts', async () => {
    await mount();
    const { control, pointer } = await openSlider();

    await act(() => control.dispatchEvent(pointer('pointerdown', 0.8)));
    const seen = [layer().opacity];
    for (const fraction of [0.7, 0.6, 0.5]) {
      await act(() => document.dispatchEvent(pointer('pointermove', fraction)));
      seen.push(layer().opacity);
    }

    // Every step reaches the layer while nothing is recorded until the release.
    expect(new Set(seen.map((value) => value.toFixed(2))).size).toBe(4);
    expect(layer().opacity).toBeCloseTo(0.5, 1);
    await expect.element(opacityInput()).toHaveValue(String(Math.round(layer().opacity * 100)));
    expect(commits()).toHaveLength(0);

    await act(() => document.dispatchEvent(pointer('pointerup', 0.5)));

    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: layer().opacity } });
    expect(commits()[0]!.inverse).toMatchObject({ id: 'r1', patch: { opacity: 1 } });
    expect(stub.history.entries().past).toEqual(['Opacity']);
  });

  it('records a drag once when Escape closes the slider mid-drag, ignoring the moves that follow', async () => {
    await mount();
    const { control, pointer } = await openSlider();

    await act(() => control.dispatchEvent(pointer('pointerdown', 0.8)));
    await act(() => document.dispatchEvent(pointer('pointermove', 0.6)));
    const atClose = layer().opacity;
    expect(atClose).toBeCloseTo(0.6, 1);

    await userEvent.keyboard('{Escape}');
    await act(() => document.dispatchEvent(pointer('pointermove', 0.3)));
    await act(() => document.dispatchEvent(pointer('pointerup', 0.3)));
    await expect.poll(() => slider().query()).toBeNull();

    expect(layer().opacity).toBe(atClose);
    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: atClose } });
    // Nothing is left pending for a later flush to record.
    await unmount();
    expect(commits()).toHaveLength(1);
    expect(stub.history.entries().past).toEqual(['Opacity']);
  });

  it('opens the slider from the keyboard, records a held key once, and returns focus on Escape', async () => {
    await mount();
    await act(async () => {
      (sliderTrigger().element() as HTMLElement).focus();
      await userEvent.keyboard('{Enter}');
    });
    await expect.element(slider()).toHaveFocus();

    await userEvent.keyboard('{ArrowLeft>3/}');

    expect(layer().opacity).toBeCloseTo(0.97, 5);
    expect(commits()).toHaveLength(1);
    expect(commits()[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.97 } });
    await expect.element(slider()).toHaveAttribute('aria-valuetext', '97%');

    await userEvent.keyboard('{Escape}');

    await expect.poll(() => slider().query()).toBeNull();
    await expect.element(sliderTrigger()).toHaveFocus();
    expect(commits()).toHaveLength(1);
  });

  it('disables every control without an editable selection', async () => {
    createStub(null);
    await mount();

    await expect.element(trigger()).toBeDisabled();
    await expect.element(opacityInput()).toBeDisabled();
    await expect.element(sliderTrigger()).toBeDisabled();
  });

  it('disables the field and the slider trigger while canvas editing is locked', async () => {
    await mount(280, true);

    await expect.element(opacityInput()).toBeDisabled();
    await expect.element(sliderTrigger()).toBeDisabled();
  });
});
