/* oxlint-disable react-perf/jsx-no-new-function-as-prop */

import type * as PsdExportModule from '@workbench/canvas-engine/export/psdExport';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { LayerRowCommands } from './layerRowCommands';

import { LayerStackHeader } from './LayerStackHeader';
import { LayerStackMenu } from './LayerStackMenu';

const harness = vi.hoisted(() => ({
  engine: null as CanvasEngine | null,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
  execute: vi.fn<typeof PsdExportModule.executePsdExport>(),
}));
const activeProject = () => harness.store!.getState().projects[0]!;

// Reuse a real project and engine without starting account persistence or the canvas registry.
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => activeProject().id,
  useActiveProjectName: () => activeProject().name,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject()),
}));
vi.mock('@workbench/canvas-operations/react', () => ({ useCanvasEngine: () => harness.engine }));
// Hold only the expensive export boundary; the controller, status stores, action hook and UI are real.
vi.mock('@workbench/canvas-engine/export/psdExport', async (importOriginal) => ({
  ...(await importOriginal<typeof PsdExportModule>()),
  executePsdExport: harness.execute,
}));

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;
let pixels: Blob;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
  const canvas = new OffscreenCanvas(16, 16);
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#fff';
  context.fillRect(0, 0, 16, 16);
  pixels = await canvas.convertToBlob({ type: 'image/png' });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  harness.engine?.lifecycle.dispose();
  harness.engine = null;
  harness.store = null;
  harness.execute.mockReset();
  root = null;
  host = null;
});

const noop = (): void => {};
const anchor = { x: 40, y: 80, width: 1, height: 1 };
const commands = { focus: noop, keyDown: noop, toggleCollapse: noop } as unknown as LayerRowCommands;

const renderActions = async () => {
  const initial = createInitialWorkbenchState();
  const base = initial.projects[0]!;
  const layer = layerContract('art', 'raster', {
    source: { image: { height: 16, imageName: 'art.png', width: 16 }, type: 'image' },
  });
  const store = createWorkbenchStore({
    ...initial,
    projects: [
      { ...base, canvas: { ...base.canvas, document: { ...base.canvas.document, stacks: stacksFrom([layer]) } } },
    ],
  });
  harness.store = store;
  const projectId = store.getState().activeProjectId;
  const engine = createCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(pixels),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: noop,
  });
  harness.engine = engine;
  host = document.createElement('div');
  host.style.cssText = 'width: 360px; height: 40px';
  document.body.append(host);
  root = createRoot(host);
  const render = (menu = false) =>
    act(() =>
      root!.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <LayerStackHeader
              collapsed={false}
              commands={commands}
              document={activeProject().canvas.document}
              editingLocked={false}
              engine={engine}
              focused
              leafCount={1}
              posInSet={1}
              rowKey="stack:raster"
              setSize={1}
              stack="raster"
            />
            {menu && (
              <LayerStackMenu
                anchor={anchor}
                document={activeProject().canvas.document}
                editingLocked={false}
                engine={engine}
                stack="raster"
                onClose={() => void render()}
              />
            )}
          </ChakraProvider>
        </I18nextProvider>
      )
    );
  await render();
  return { engine, openMenu: () => render(true) };
};

it.each(['header pointer', 'menu keyboard'] as const)(
  'disables export, shows live progress and cancels through the real engine from the %s, then permits retry',
  async (surface) => {
    const jobs: { signal: AbortSignal; reject(error: Error): void }[] = [];
    harness.execute.mockImplementation((_plan, _fileName, deps) => {
      deps.onStage?.('serializing');
      return new Promise<void>((_resolve, reject) => {
        jobs.push({ signal: deps.signal!, reject });
      });
    });
    const { engine, openMenu } = await renderActions();
    const exportButton = page.getByRole('button', { exact: true, name: 'Export to PSD' });
    const cancelButton = page.getByRole('button', { exact: true, name: 'Cancel PSD export' });
    await expect.element(exportButton).toBeEnabled();
    await expect.element(cancelButton).not.toBeInTheDocument();

    await act(() => exportButton.click());

    await vi.waitFor(() => expect(jobs).toHaveLength(1));
    expect(engine.interaction.get('psdExportStatus')).toBe('serializing');
    await expect.element(exportButton).toBeDisabled();
    await expect.element(page.getByRole('status')).toHaveTextContent('Writing PSD…');
    await expect.element(cancelButton).toBeEnabled();
    if (surface === 'menu keyboard') {
      await openMenu();
      await expect
        .element(page.getByRole('menuitem', { name: 'Writing PSD…' }))
        .toHaveAttribute('aria-disabled', 'true');
      const cancelItem = document.querySelector<HTMLElement>('[role="menuitem"][data-value="cancelExportPsd"]')!;
      for (let step = 0; step < 4 && !cancelItem.hasAttribute('data-highlighted'); step++) {
        await act(() => userEvent.keyboard('{ArrowDown}'));
      }
      expect(cancelItem).toHaveAttribute('data-highlighted');
      await act(() => userEvent.keyboard('{Enter}'));
    } else {
      await act(() => cancelButton.click());
    }

    await vi.waitFor(() => expect(jobs[0]!.signal.aborted).toBe(true));
    expect(engine.interaction.get('psdExportStatus')).toBe('cancelling');
    await expect.element(page.getByRole('status')).toHaveTextContent('Cancelling PSD…');
    await expect.element(cancelButton).toBeDisabled();
    await expect.element(exportButton).toBeDisabled();
    await act(() => jobs[0]!.reject(new DOMException('Aborted', 'AbortError')));

    await expect.element(exportButton).toBeEnabled();
    await expect.element(cancelButton).not.toBeInTheDocument();
    await expect.element(page.getByRole('status')).not.toBeInTheDocument();
    expect(engine.interaction.get('psdExportStatus')).toBe('idle');
    await act(() => exportButton.click());
    await vi.waitFor(() => expect(jobs).toHaveLength(2));
    expect(jobs[1]!.signal.aborted).toBe(false);
    await act(() => engine.exports.cancelPsdExport());
    await act(() => jobs[1]!.reject(new DOMException('Aborted', 'AbortError')));
    await expect.element(exportButton).toBeEnabled();
  }
);
