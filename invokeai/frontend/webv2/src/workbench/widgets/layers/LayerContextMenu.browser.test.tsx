/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { CanvasLayerContract, CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { getDocumentIndex, getDocumentLeaves } from '@workbench/canvas-engine/api';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { clearLayerPanelStates, publishLayerPanelSelection } from '@workbench/layerPanelState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

/** The menu reads the real store and drives a real engine over real decoded pixels. */
const harness = vi.hoisted(() => ({
  engine: null as CanvasEngine | null,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));
const activeProject = () => harness.store!.getState().projects[0]!;

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => activeProject().id,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject()),
  useOptionalWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchCommands: () => harness.store!.commands,
}));

import { CanvasLayerContextMenu } from './LayerContextMenu';

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
  Object.assign(harness, { engine: null, store: null });
  clearLayerPanelStates();
  root = null;
  host = null;
});

const image = (name: string) => ({ height: 16, imageName: `${name}.png`, width: 16 });
const imageLayer = (id: string): CanvasLayerContract =>
  layerContract(id, 'raster', {
    name: id,
    source: { image: image(id), type: 'image' },
  } as Partial<CanvasLayerContract>);
const maskLayer = (id: string, overrides: Partial<CanvasLayerContract> = {}): CanvasLayerContract =>
  layerContract(id, 'inpaint_mask', {
    mask: { bitmap: image(id), fill: { color: '#e45050', style: 'solid' }, offset: { x: 0, y: 0 } },
    name: id,
    ...overrides,
  } as Partial<CanvasLayerContract>);

/**
 * Renders the Layers panel's layer menu over a real engine for `nodes`, with `selectedIds` selected in the panel (the
 * first as primary), returning a way to open it on a layer. Images named in `unreadable` fail to load.
 */
const renderMenu = async (
  nodes: CanvasNodeContract[],
  selectedIds: readonly string[] = [],
  unreadable: readonly string[] = []
) => {
  const initial = createInitialWorkbenchState();
  const base = initial.projects[0]!;
  const store = createWorkbenchStore({
    ...initial,
    projects: [
      {
        ...base,
        canvas: {
          ...base.canvas,
          document: { ...base.canvas.document, selectedLayerId: selectedIds[0] ?? null, stacks: stacksFrom(nodes) },
        },
      },
    ],
  });
  const projectId = store.getState().activeProjectId;
  harness.store = store;
  publishLayerPanelSelection({ primaryId: selectedIds[0] ?? null, projectId, selectedIds });
  const engine = createCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: (imageName) =>
      unreadable.includes(imageName) ? Promise.reject(new Error('image deleted')) : Promise.resolve(pixels),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  harness.engine = engine;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const queryClient = new QueryClient();
  const render = (layerId: string | null) =>
    act(() =>
      root!.render(
        <QueryClientProvider client={queryClient}>
          <I18nextProvider i18n={i18n}>
            <ChakraProvider value={system}>
              <CanvasLayerContextMenu
                engine={engine}
                target={layerId ? { layerId, x: 40, y: 40 } : null}
                onClose={() => void render(null)}
              />
            </ChakraProvider>
          </I18nextProvider>
        </QueryClientProvider>
      )
    );
  await render(null);
  const openMenu = async (layerId: string) => {
    await render(layerId);
    await expect.element(page.getByRole('menu')).toBeVisible();
  };
  const document_ = () => store.getState().projects[0]!.canvas.document;
  const notices = () => store.getState().notifications.map(({ message, title }) => [title, message]);
  return { document: document_, engine, notices, openMenu };
};

const item = (name: string) => page.getByRole('menuitem', { exact: true, name });

describe('Layers context menu over a real engine', () => {
  it('opens a transform session on the chosen layer while the Transform tool is already active', async () => {
    const { engine, openMenu } = await renderMenu([imageLayer('top'), imageLayer('art')]);
    engine.tools.setTool('transform');
    expect(engine.stores.transformSession.get()).toBeNull();

    await openMenu('art');
    await act(() => item('Transform').click());

    await vi.waitFor(() => expect(engine.stores.transformSession.get()?.layerId).toBe('art'));
    expect(engine.document.getDocument()?.selectedLayerId).toBe('art');
  });

  it('merges two selected masks into one', async () => {
    const { document, engine, openMenu } = await renderMenu([maskLayer('m1'), maskLayer('m2')], ['m1', 'm2']);
    await vi.waitFor(() => expect(['m1', 'm2'].every((id) => engine.exports.hasExportableLayerContent(id))).toBe(true));

    await openMenu('m1');
    await act(() => item('Merge masks').click());

    await vi.waitFor(() =>
      expect(getDocumentLeaves(document()).map((layer) => [layer.type, layer.name])).toEqual([
        ['inpaint_mask', 'm1 merged'],
      ])
    );
    expect(engine.history.getEntries().past).toEqual(['Merge masks']);
  });

  it('removes a layer from its group as one undo step', async () => {
    const { document, engine, openMenu } = await renderMenu([
      groupContract('G', [imageLayer('child'), imageLayer('sibling')]),
    ]);
    const parentOf = (id: string) => getDocumentIndex(document()).byId.get(id)?.parentId;

    await openMenu('child');
    await act(() => item('Remove from group').click());

    await vi.waitFor(() => expect(parentOf('child')).toBeNull());
    expect(engine.history.getEntries().past).toHaveLength(1);
    await act(() => engine.history.undo());
    expect(parentOf('child')).toBe('G');
    expect(engine.history.getEntries().past).toHaveLength(0);
  });

  it.each([
    ['Crop layer to frame', ['gone'], []],
    ['Merge selected layers', ['gone', 'kept'], ['kept']],
  ] as const)(
    'says %s could not read a missing image, never that it is still loading',
    async (label, ids, readable) => {
      const { document, notices, openMenu } = await renderMenu(
        ids.map(imageLayer),
        ids,
        ids.filter((id) => !(readable as readonly string[]).includes(id)).map((id) => `${id}.png`)
      );
      const before = document();

      await openMenu('gone');
      await act(() => item(label).click());

      await vi.waitFor(() =>
        expect(notices()).toContainEqual([
          'Layer action failed',
          "A layer's image could not be read (it may have been deleted or damaged), so nothing was changed.",
        ])
      );
      expect(notices().flat()).not.toContain('The layer content is still loading. Try again shortly.');
      expect(document().stacks).toBe(before.stacks);
    }
  );

  it('keeps an unavailable Merge masks reachable by keyboard and says why', async () => {
    const { engine, openMenu } = await renderMenu(
      [maskLayer('m1'), maskLayer('m2', { noise: { isEnabled: true, level: 0.5 } } as Partial<CanvasLayerContract>)],
      ['m1', 'm2']
    );
    const reason = 'The masks have different noise or denoise settings. Match them to merge.';

    await openMenu('m1');
    const merge = document.querySelector<HTMLElement>('[role="menuitem"][data-value="merge-masks"]')!;
    expect(merge).toHaveAttribute('aria-disabled', 'true');
    for (let step = 0; step < 30 && !merge.hasAttribute('data-highlighted'); step += 1) {
      await userEvent.keyboard('{ArrowDown}');
    }
    expect(merge).toHaveAttribute('data-highlighted');
    await expect.element(page.getByRole('tooltip')).toHaveTextContent(reason);
    expect(document.getElementById(merge.getAttribute('aria-describedby')!)).toHaveTextContent(reason);

    await userEvent.keyboard('{Enter}');
    expect(getDocumentLeaves(engine.document.getDocument()!).map((layer) => layer.id)).toEqual(['m1', 'm2']);
  });
});
