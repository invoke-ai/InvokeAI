import { ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({ useWorkflowPreferencesSelector: () => false }));
vi.mock('@xyflow/react', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useReactFlow: () => ({ fitView: vi.fn(), zoomIn: vi.fn(), zoomOut: vi.fn() }),
}));

const { EditorToolbar } = await import('./EditorToolbar');

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const render = async (nodeOpacity: number) => {
  applyThemeToRoot('mono');
  host = document.createElement('div');
  host.style.cssText = 'height:520px;position:relative;width:320px';
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <EditorToolbar nodeOpacity={nodeOpacity} tool="pan" onNodeOpacityChange={vi.fn()} onToolChange={vi.fn()} />
      </ChakraProvider>
    );
  });
};

/** Compare rasterized fills because equivalent token/color-mix colors serialize into different color spaces. */
const paintedFill = (element: Element): string => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { willReadFrequently: true })!;

  context.fillStyle = getComputedStyle(element).backgroundColor;
  context.fillRect(0, 0, 1, 1);

  return [...context.getImageData(0, 0, 1, 1).data].join(',');
};

const buttonBoxes = (): string[] =>
  [...host!.querySelectorAll('[role="toolbar"] button')].map((button) => {
    const rect = button.getBoundingClientRect();

    return `${rect.width}x${rect.height}`;
  });

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('editor toolbar', () => {
  it('keeps every button on one shared square toolbar box', async () => {
    // Match custom control sizing; one wider child stretches every button in the column toolbar.
    await render(1);

    expect(new Set(buttonBoxes())).toEqual(new Set(['28x28']));
  });

  it('offers to update outdated nodes only while there are some, without breaking the toolbar grid', async () => {
    await render(1);
    expect(host!.querySelector('button[aria-label="nodes.updateAllNodes"]')).toBeNull();

    const onUpdateNodes = vi.fn();

    await settle(() => {
      root?.render(
        <ChakraProvider value={system}>
          <EditorToolbar
            nodeOpacity={1}
            tool="pan"
            updatableNodeCount={2}
            onNodeOpacityChange={vi.fn()}
            onToolChange={vi.fn()}
            onUpdateNodes={onUpdateNodes}
          />
        </ChakraProvider>
      );
    });

    const update = host!.querySelector<HTMLButtonElement>('button[aria-label="nodes.updateAllNodes"]')!;

    expect(update).not.toBeNull();
    expect(new Set(buttonBoxes())).toEqual(new Set(['28x28']));
    update.focus();
    await settle(() => update.click());
    expect(onUpdateNodes).toHaveBeenCalledOnce();
    // The button leaves with the last outdated node; focus is already on its neighbour.
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Fit view');
  });

  it('states node opacity the way the tool buttons state themselves', async () => {
    await render(0.5);
    const opacity = host!.querySelector<HTMLButtonElement>('button[aria-label="Node opacity"]')!;
    const activeTool = host!.querySelector<HTMLButtonElement>(
      'button[aria-pressed="true"]:not([aria-label="Node opacity"])'
    )!;

    expect(opacity.getAttribute('aria-pressed')).toBe('true');
    expect(paintedFill(opacity)).toBe(paintedFill(activeTool));
    expect(new Set(buttonBoxes())).toEqual(new Set(['28x28']));
  });
});
