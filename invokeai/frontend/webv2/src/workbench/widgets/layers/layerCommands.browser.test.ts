/**
 * The layer commands over a real engine wired as the app wires it: the labels they give reach the engine's undo
 * history, and a duplicate leaves its copies selected in the Layers panel.
 */

import { getDocumentIndex } from '@workbench/canvas-engine/api';
import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { clearLayerPanelStates, publishLayerPanelSelection, readLayerPanelState } from '@workbench/layerPanelState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import { deleteLayers, duplicateLayers, groupLayers, type LayerCommandDeps } from './layerCommands';

const i18n = createInstance();
const cleanups: (() => void)[] = [];

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) {
    cleanup();
  }
  clearLayerPanelStates();
});

const setup = (ids: readonly string[]) => {
  const store = createWorkbenchStore();
  const projectId = store.getState().activeProjectId;
  store.commands.canvas.apply(projectId, {
    document: {
      ...createEmptyCanvasDocument(64, 64),
      selectedLayerId: ids[0]!,
      stacks: stacksFrom(ids.map((id) => layerContract(id))),
    },
    type: 'replaceCanvasDocument',
  });
  const engine = createCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.reject(new Error('no images')),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
    setSelectedLayerIds: (primaryId, selectedIds) => publishLayerPanelSelection({ primaryId, projectId, selectedIds }),
  });
  cleanups.push(() => engine.lifecycle.dispose());
  const errors: [string, string][] = [];
  const deps: LayerCommandDeps = { engine, reportError: (title, message) => errors.push([title, message]), t: i18n.t };
  const document = () => store.getState().projects.find((project) => project.id === projectId)!.canvas.document;
  const panel = () => readLayerPanelState(projectId, document().selectedLayerId);
  return { deps, document, engine, errors, panel };
};

describe('layer commands on a real engine', () => {
  it('records each command under its count label, and selects the copies a duplicate makes', async () => {
    const h = setup(['a', 'b', 'c']);

    expect(await duplicateLayers(h.deps, ['a', 'b'])).toBe('committed');
    const copies = h.panel().selectedIds;
    const byId = getDocumentIndex(h.document()).byId;
    expect(copies.map((id) => byId.get(id)?.node.name).sort()).toEqual(['a copy', 'b copy']);
    expect(h.document().selectedLayerId).toBe(h.panel().primaryId);

    expect(groupLayers(h.deps, copies)).toBe('committed');
    expect(deleteLayers(h.deps, ['c'])).toBe('committed');

    expect(h.engine.history.getEntries().past).toEqual(['Duplicate 2 layers', 'Group 2 layers', 'Delete layer']);
    expect(h.errors).toEqual([]);
  });
});
