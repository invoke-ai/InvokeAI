import { ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const toolbarMocks = vi.hoisted(() => ({
  exportWorkflowAsPng: vi.fn(),
  error: vi.fn(),
  fitView: vi.fn(),
  getNodes: vi.fn(() => [{ id: 'seed' }]),
  getNodesBounds: vi.fn(() => ({ x: 12, y: 34, width: 56, height: 78 })),
  info: vi.fn(),
  onExportPendingChange: vi.fn(),
  success: vi.fn(),
  zoomIn: vi.fn(),
  zoomOut: vi.fn(),
  workflowName: 'Test workflow',
}));

vi.mock('@features/workflow/ui/WorkflowUiContext', () => ({
  useWorkflowNotifications: () => ({
    error: toolbarMocks.error,
    info: toolbarMocks.info,
    success: toolbarMocks.success,
  }),
  useWorkflowPreferencesSelector: () => false,
  useWorkflowProjectSelector: (selector: (project: unknown) => unknown) =>
    selector({ activeWorkflow: { document: { name: toolbarMocks.workflowName } } }),
}));
vi.mock('@xyflow/react', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useReactFlow: () => toolbarMocks,
}));
vi.mock('./workflowImageExport', () => ({ exportWorkflowAsPng: toolbarMocks.exportWorkflowAsPng }));

const { EditorToolbar } = await import('./EditorToolbar');

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

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

beforeEach(() => {
  vi.clearAllMocks();
  toolbarMocks.workflowName = 'Test workflow';
});

const render = async (nodeOpacity: number) => {
  applyThemeToRoot('mono');
  host = document.createElement('div');
  host.className = 'react-flow';
  host.style.cssText = 'height:520px;position:relative;width:320px';
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <EditorToolbar
            nodeOpacity={nodeOpacity}
            tool="pan"
            onExportPendingChange={toolbarMocks.onExportPendingChange}
            onNodeOpacityChange={vi.fn()}
            onToolChange={vi.fn()}
          />
        </ChakraProvider>
      </I18nextProvider>
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
    expect(host!.querySelector('button[aria-label^="Update "]')).toBeNull();

    const onUpdateNodes = vi.fn();

    await settle(() => {
      root?.render(
        <I18nextProvider i18n={i18n}>
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
        </I18nextProvider>
      );
    });

    const update = host!.querySelector<HTMLButtonElement>('button[aria-label="Update 2 outdated nodes"]')!;

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

  it('offers a camera action for exporting the workflow image', async () => {
    toolbarMocks.exportWorkflowAsPng.mockResolvedValue(undefined);
    await render(1);

    const camera = host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!;

    expect(camera).not.toBeNull();
    await settle(() => camera.click());
    await vi.waitFor(() => expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledOnce());

    expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledWith({
      bounds: { x: 12, y: 34, width: 56, height: 78 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: host,
      workflowName: 'Test workflow',
    });
    expect(toolbarMocks.onExportPendingChange.mock.calls).toEqual([[true], [false]]);
    expect(camera.disabled).toBe(false);
  });

  it('shows the translated label as a camera tooltip', async () => {
    await render(1);

    const camera = host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!;
    await act(async () => {
      await userEvent.hover(camera);
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 500);
      });
    });

    const tooltip = [...document.querySelectorAll('[role="tooltip"]')].find(
      (element) => element.textContent === 'Download workflow as PNG'
    );

    expect(tooltip).not.toBeUndefined();
  });

  it('reports an export failure and releases the camera action', async () => {
    toolbarMocks.exportWorkflowAsPng.mockRejectedValueOnce(new Error('rasterization failed'));
    await render(1);

    const camera = host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!;
    await settle(() => camera.click());
    await vi.waitFor(() => expect(toolbarMocks.error).toHaveBeenCalledWith('Could not export workflow image.'));

    expect(toolbarMocks.error).toHaveBeenCalledWith('Could not export workflow image.');
    expect(toolbarMocks.onExportPendingChange.mock.calls).toEqual([[true], [false]]);
    expect(camera.disabled).toBe(false);
  });

  it('uses the translated untitled workflow name for an unnamed export', async () => {
    toolbarMocks.workflowName = '';
    toolbarMocks.exportWorkflowAsPng.mockResolvedValue(undefined);
    await render(1);

    await settle(() =>
      host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!.click()
    );
    await vi.waitFor(() => expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledOnce());

    expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledWith(
      expect.objectContaining({ fallbackWorkflowName: 'Untitled Workflow', workflowName: '' })
    );
  });
});
