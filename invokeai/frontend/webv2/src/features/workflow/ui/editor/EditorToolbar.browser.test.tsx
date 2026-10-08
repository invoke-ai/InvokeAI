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
  onExportPrepare: vi.fn(),
  onExportComplete: vi.fn(),
  preflightWorkflowImageExport: vi.fn(),
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
vi.mock('./workflowImageExport', () => ({
  exportWorkflowAsPng: toolbarMocks.exportWorkflowAsPng,
  preflightWorkflowImageExport: toolbarMocks.preflightWorkflowImageExport,
}));

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
  toolbarMocks.preflightWorkflowImageExport.mockReturnValue(null);
  toolbarMocks.workflowName = 'Test workflow';
});

const render = async (nodeOpacity: number) => {
  applyThemeToRoot('mono');
  host = document.createElement('div');
  host.className = 'react-flow';
  host.style.cssText = 'height:520px;position:relative;width:320px';
  document.body.append(host);
  root = createRoot(host);
  toolbarMocks.onExportPrepare.mockResolvedValue(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <EditorToolbar
            nodeOpacity={nodeOpacity}
            tool="pan"
            onExportPrepare={toolbarMocks.onExportPrepare}
            onExportComplete={toolbarMocks.onExportComplete}
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
              onExportPrepare={toolbarMocks.onExportPrepare}
              onExportComplete={toolbarMocks.onExportComplete}
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

  it('captures the prepared export view and keeps one busy camera action until rasterization finishes', async () => {
    let finish: () => void = () => {};
    toolbarMocks.exportWorkflowAsPng.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = () => resolve({ status: 'exported', reduced: false, width: 112, height: 156 });
        })
    );
    await render(1);
    const exportElement = document.createElement('div');
    toolbarMocks.onExportPrepare.mockResolvedValueOnce(exportElement);

    const camera = host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!;

    expect(camera).not.toBeNull();
    expect(camera.getAttribute('aria-busy')).toBe('false');
    camera.focus();
    await settle(() => {
      camera.click();
      camera.click();
    });
    await vi.waitFor(() => expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledOnce());

    expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledWith({
      bounds: { x: 12, y: 34, width: 56, height: 78 },
      fallbackWorkflowName: 'Untitled Workflow',
      flowElement: exportElement,
      workflowName: 'Test workflow',
    });
    expect(camera.disabled).toBe(true);
    expect(camera.getAttribute('aria-busy')).toBe('true');
    await settle(() => camera.click());
    expect(toolbarMocks.onExportPrepare).toHaveBeenCalledOnce();
    expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledOnce();
    expect(toolbarMocks.onExportComplete).not.toHaveBeenCalled();
    finish();
    await vi.waitFor(() => expect(toolbarMocks.onExportComplete).toHaveBeenCalledOnce());
    expect(camera.disabled).toBe(false);
    expect(camera.getAttribute('aria-busy')).toBe('false');
    expect(document.activeElement).toBe(camera);
    expect(toolbarMocks.error).not.toHaveBeenCalled();
    expect(toolbarMocks.info).not.toHaveBeenCalled();
  });

  it.each([
    {
      outcome: { status: 'exported', reduced: true, width: 9000, height: 489 },
      notification: 'info',
      args: [
        'Workflow image exported at reduced resolution',
        'The workflow is too large to export at screen resolution, so the image is 9000 × 489 px.',
      ],
    },
    {
      outcome: { status: 'too-large' },
      notification: 'error',
      args: [
        'Workflow is too large to export as an image',
        'Even at reduced resolution the image would exceed what the browser can render safely. Export the workflow JSON instead.',
      ],
    },
    {
      outcome: { status: 'busy' },
      notification: 'info',
      args: ['A previous workflow image export is still finishing. Try again shortly.'],
    },
  ] as const)('reports a $outcome.status export outcome', async ({ args, notification, outcome }) => {
    toolbarMocks.exportWorkflowAsPng.mockResolvedValueOnce(outcome);
    await render(1);

    const camera = host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!;
    await settle(() => camera.click());
    await vi.waitFor(() => expect(toolbarMocks.onExportComplete).toHaveBeenCalledOnce());

    expect(toolbarMocks[notification]).toHaveBeenCalledExactlyOnceWith(...args);
    expect(toolbarMocks[notification === 'info' ? 'error' : 'info']).not.toHaveBeenCalled();
    expect(camera.disabled).toBe(false);
  });

  it('refuses an oversized workflow from the live bounds without preparing the export view', async () => {
    toolbarMocks.preflightWorkflowImageExport.mockReturnValueOnce({ status: 'too-large' });
    await render(1);

    const camera = host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!;
    await settle(() => camera.click());
    await vi.waitFor(() => expect(toolbarMocks.onExportComplete).toHaveBeenCalledOnce());

    expect(toolbarMocks.preflightWorkflowImageExport).toHaveBeenCalledWith({ x: 12, y: 34, width: 56, height: 78 });
    expect(toolbarMocks.onExportPrepare).not.toHaveBeenCalled();
    expect(toolbarMocks.exportWorkflowAsPng).not.toHaveBeenCalled();
    expect(toolbarMocks.error).toHaveBeenCalledExactlyOnceWith(
      'Workflow is too large to export as an image',
      'Even at reduced resolution the image would exceed what the browser can render safely. Export the workflow JSON instead.'
    );
    expect(camera.disabled).toBe(false);
  });

  it('exports silently when the output is reduced but still at least screen resolution', async () => {
    toolbarMocks.exportWorkflowAsPng.mockResolvedValueOnce({
      status: 'exported',
      reduced: false,
      width: 9000,
      height: 5000,
    });
    await render(1);

    await settle(() =>
      host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!.click()
    );
    await vi.waitFor(() => expect(toolbarMocks.onExportComplete).toHaveBeenCalledOnce());

    expect(toolbarMocks.info).not.toHaveBeenCalled();
    expect(toolbarMocks.error).not.toHaveBeenCalled();
  });

  it('stays silent when the editor unmounts during an export', async () => {
    let finish: () => void = () => {};
    toolbarMocks.exportWorkflowAsPng.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = () => resolve({ status: 'canceled' });
        })
    );
    await render(1);

    await settle(() =>
      host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!.click()
    );
    await vi.waitFor(() => expect(toolbarMocks.exportWorkflowAsPng).toHaveBeenCalledOnce());
    await settle(() => root?.unmount());
    root = null;
    finish();
    await vi.waitFor(() => expect(toolbarMocks.onExportComplete).toHaveBeenCalledOnce());

    expect(toolbarMocks.error).not.toHaveBeenCalled();
    expect(toolbarMocks.info).not.toHaveBeenCalled();
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

  it.each(['prepare', 'rasterize'])('reports a %s failure and releases the camera action', async (stage) => {
    await render(1);
    (stage === 'prepare' ? toolbarMocks.onExportPrepare : toolbarMocks.exportWorkflowAsPng).mockRejectedValueOnce(
      new Error('export failed')
    );

    const camera = host!.querySelector<HTMLButtonElement>('button[aria-label="Download workflow as PNG"]')!;
    await settle(() => camera.click());
    await vi.waitFor(() => expect(toolbarMocks.error).toHaveBeenCalledWith('Could not export workflow image.'));

    expect(toolbarMocks.error).toHaveBeenCalledWith('Could not export workflow image.');
    expect(toolbarMocks.onExportComplete).toHaveBeenCalledOnce();
    expect(camera.disabled).toBe(false);
  });

  it('uses the translated untitled workflow name for an unnamed export', async () => {
    toolbarMocks.workflowName = '';
    toolbarMocks.exportWorkflowAsPng.mockResolvedValue({ status: 'exported', reduced: false, width: 112, height: 156 });
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
