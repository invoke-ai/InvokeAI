import type {
  CanvasLayerContract,
  CanvasLayerPreviewMutation,
  PreparedDocumentEdit,
} from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createDocumentModel } from '@workbench/canvas-engine/api';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({
  layers: [] as CanvasLayerContract[],
  listeners: new Set<() => void>(),
  project: null as { canvas: { document: unknown } } | null,
  selectedLayerId: null as string | null,
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

const publish = (): void => {
  harness.project = { canvas: { document: documentFrom(harness.layers, harness.selectedLayerId) } };
  harness.listeners.forEach((listener) => listener());
};

/** Applies an `updateCanvasLayer` patch the way the reducer does, so the row re-renders from the document. */
const applyMutation = (mutation: unknown): void => {
  const { id, patch, type } = mutation as { id: string; patch: Partial<CanvasLayerContract>; type: string };
  if (type !== 'updateCanvasLayer') {
    throw new Error(`Unexpected mutation ${type}`);
  }
  harness.layers = harness.layers.map((layer) =>
    layer.id === id ? ({ ...layer, ...patch } as CanvasLayerContract) : layer
  );
  publish();
};

const previews: CanvasLayerPreviewMutation[] = [];
const commits: PreparedDocumentEdit[] = [];

/** One engine preview session at a time, as the engine owns them: a newer session ends the last. */
const createEngine = () => {
  let current: object | null = null;
  return {
    document: {
      model: () => createDocumentModel(harness.project!.canvas.document as never, { editRevision: 0, projectId: 'p' }),
    },
    exports: {},
    interaction: { get: () => false, subscribe: () => () => undefined },
    layers: {
      beginStructuralPreview: () => {
        const session = {
          apply: (action: CanvasLayerPreviewMutation) => {
            if (current !== session) {
              return false;
            }
            previews.push(action);
            applyMutation(action);
            return true;
          },
          cancel: (restore?: CanvasLayerPreviewMutation) => {
            if (current === session) {
              current = null;
              if (restore) {
                applyMutation(restore);
              }
            }
          },
          commit: (_label: string, edit: PreparedDocumentEdit) => {
            if (current !== session) {
              return { status: 'busy' as const };
            }
            current = null;
            commits.push(edit);
            applyMutation(edit.forward);
            return { status: 'committed' as const };
          },
        };
        current = session;
        return session;
      },
      commitPrepared: (_label: string, edit: PreparedDocumentEdit) => {
        current = null;
        commits.push(edit);
        applyMutation(edit.forward);
        return { status: 'committed' as const };
      },
    },
    projectId: 'p',
  };
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const mount = async (width = 280): Promise<void> => {
  host = document.createElement('div');
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);
  const engine = createEngine();
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

const layer = (id = 'r1'): CanvasLayerContract => harness.layers.find((candidate) => candidate.id === id)!;
const trigger = () => page.getByRole('combobox', { name: 'Blend mode' });
const option = (name: string) => page.getByRole('option', { exact: true, name });

beforeEach(() => {
  harness.layers = [layerContract('r1'), layerContract('r2')];
  harness.selectedLayerId = 'r1';
  publish();
});

afterEach(async () => {
  await unmount();
  host?.remove();
  host = null;
  previews.length = 0;
  commits.length = 0;
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
    expect(commits).toHaveLength(0);
    expect(notify.error).not.toHaveBeenCalled();
  });

  it('previews the keyboard highlight and records one step from the original when it is chosen', async () => {
    await mount();
    await userEvent.click(trigger());
    await expect.element(option('Normal')).toBeVisible();
    await userEvent.keyboard('{ArrowDown}{ArrowDown}');

    await expect.poll(() => layer().blendMode).toBe('screen');

    await userEvent.keyboard('{Enter}');

    expect(commits).toHaveLength(1);
    expect(commits[0]!.forward).toMatchObject({ patch: { blendMode: 'screen' } });
    expect(commits[0]!.inverse).toMatchObject({ patch: { blendMode: 'normal' } });
    await expect.element(trigger()).toHaveTextContent('Screen');
    expect(layer().blendMode).toBe('screen');
  });

  it('records a clicked option once, even after previewing others on the way', async () => {
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Multiply'));
    await userEvent.hover(option('Overlay'));
    await userEvent.click(option('Overlay'));

    expect(commits).toHaveLength(1);
    expect(commits[0]!.inverse).toMatchObject({ id: 'r1', patch: { blendMode: 'normal' } });
    expect(layer().blendMode).toBe('overlay');
  });

  it('restores the original when the menu unmounts or the selection moves while it is open', async () => {
    await mount();
    await userEvent.click(trigger());
    await userEvent.hover(option('Darken'));
    await expect.poll(() => layer().blendMode).toBe('darken');

    await act(() => {
      harness.selectedLayerId = 'r2';
      publish();
    });

    expect(layer('r1').blendMode).toBe('normal');
    expect(page.getByRole('option').query()).toBeNull();

    await userEvent.click(trigger());
    await userEvent.hover(option('Lighten'));
    await expect.poll(() => layer('r2').blendMode).toBe('lighten');
    await unmount();

    expect(layer('r2').blendMode).toBe('normal');
    expect(commits).toHaveLength(0);
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
    expect(commits).toHaveLength(1);
    expect(commits[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.96 } });
    expect(commits[0]!.inverse).toMatchObject({ id: 'r1', patch: { opacity: 1 } });
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
    expect(commits).toHaveLength(0);
    await act(() =>
      document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, button: 0, clientX: 80, clientY: 100 }))
    );
    expect(commits).toHaveLength(1);
    expect(commits[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.95 } });
    expect(commits[0]!.inverse).toMatchObject({ id: 'r1', patch: { opacity: 1 } });
  });

  it('previews a typed value and records it once on Enter', async () => {
    await mount();
    await userEvent.tripleClick(opacityInput());
    await userEvent.keyboard('45');

    expect(layer().opacity).toBeCloseTo(0.45, 5);
    expect(commits).toHaveLength(0);

    await userEvent.keyboard('{Enter}');

    expect(commits).toHaveLength(1);
    expect(commits[0]!.forward).toMatchObject({ patch: { opacity: 0.45 } });
    expect(commits[0]!.inverse).toMatchObject({ patch: { opacity: 1 } });
  });

  it('records a still-pending edit when the row unmounts', async () => {
    await mount();
    await userEvent.tripleClick(opacityInput());
    await userEvent.keyboard('30');
    expect(commits).toHaveLength(0);

    await unmount();

    expect(commits).toHaveLength(1);
    expect(commits[0]!.forward).toMatchObject({ id: 'r1', patch: { opacity: 0.3 } });
  });

  it('disables both controls without an editable selection', async () => {
    harness.selectedLayerId = null;
    publish();
    await mount();

    await expect.element(trigger()).toBeDisabled();
    await expect.element(opacityInput()).toBeDisabled();
  });
});
