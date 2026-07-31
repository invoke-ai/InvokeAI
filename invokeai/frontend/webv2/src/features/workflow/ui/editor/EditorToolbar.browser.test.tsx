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
  it('keeps every button on the shared 32px toolbar box', async () => {
    // The node-opacity button is hand-rolled rather than a ToolbarButton, so it
    // never picked up the primitive's size. That is not a local defect: Toolbar
    // is a column Stack, and its default stretch alignment let one `sm` button
    // widen every sibling to 36px while they kept their 32px height, turning
    // the whole strip into rectangles four pixels wider than the canvas one.
    await render(1);

    expect(new Set(buttonBoxes())).toEqual(new Set(['32x32']));
  });

  it('states node opacity the way the tool buttons state themselves', async () => {
    // Its "on" used to be a bespoke accent icon colour, while every other
    // button in the strip fills. One control type, one vocabulary.
    await render(0.5);
    const opacity = host!.querySelector<HTMLButtonElement>('button[aria-label="Node opacity"]')!;
    const activeTool = host!.querySelector<HTMLButtonElement>(
      'button[aria-pressed="true"]:not([aria-label="Node opacity"])'
    )!;

    expect(opacity.getAttribute('aria-pressed')).toBe('true');
    expect(getComputedStyle(opacity).backgroundColor).toBe(getComputedStyle(activeTool).backgroundColor);
    expect(new Set(buttonBoxes())).toEqual(new Set(['32x32']));
  });
});
