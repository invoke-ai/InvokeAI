import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project } from '@workbench/projectContracts';

import { createEmptyCanvasState } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { describe, expect, it, vi } from 'vitest';

import type { CanvasDocumentContractV3, CanvasLayerContract, CanvasTextFontRef } from './contracts';
import type { BitmapStore } from './document/bitmapStore';
import type { CanvasFontRuntime } from './render/fontLoader';

import { documentFrom, groupContract, layerContract } from './document-model/documentFixtures.testStub';
import { createCanvasEngine } from './engine';
import { createTestStubRasterBackend } from './render/raster.testStub';

const from: CanvasTextFontRef = {
  contentHash: 'a'.repeat(64),
  family: 'Catalog Sans',
  id: 'font-a',
  label: 'Catalog Sans Regular',
};
const to: CanvasTextFontRef = {
  contentHash: 'b'.repeat(64),
  family: 'Replacement Sans',
  id: 'font-b',
  label: 'Replacement Sans Variable',
};

const textLayer = (id: string, fontRef: CanvasTextFontRef): CanvasLayerContract =>
  layerContract(id, 'raster', {
    source: {
      align: 'left',
      color: '#ffffff',
      content: id,
      fontFamily: fontRef.family,
      fontRef,
      fontSize: 24,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    },
  });

const createBitmapStoreStub = (): BitmapStore => ({
  discardLayer: () => undefined,
  dispose: () => undefined,
  flushPendingUploads: () => Promise.resolve(),
  hasPendingClear: () => false,
  hasPendingWork: () => false,
  isSelfEcho: () => false,
  markLayerDirty: () => undefined,
  reset: () => undefined,
  suspendLayer: () => () => undefined,
});

const createHarness = (document: CanvasDocumentContractV3, fonts: CanvasFontRuntime | null = null) => {
  const initial = createInitialWorkbenchState().projects[0]!;
  let project: Project = { ...initial, canvas: { ...createEmptyCanvasState(), document } };
  const listeners = new Set<() => void>();
  const mutationPort: CanvasProjectMutationPort = {
    commitEdit: () => undefined,
    dispatch: (mutation) => {
      const next = applyCanvasProjectMutation(project, mutation);
      if (next.canvas === project.canvas) {
        return false;
      }
      project = next;
      for (const listener of listeners) {
        listener();
      }
      return true;
    },
    getCanvasState: () => project.canvas,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  const composition = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    bitmapStore: createBitmapStoreStub(),
    fonts,
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort,
    projectId: project.id,
    reportError: () => undefined,
    uploadImage: () => Promise.resolve({ height: 1, imageName: 'image', width: 1 }),
    uploadIntermediateImage: () => Promise.resolve({ height: 1, imageName: 'intermediate', width: 1 }),
  });
  return {
    composition,
    getDocument: () => project.canvas.document,
    notifyUnchanged: () => listeners.forEach((listener) => listener()),
  };
};

describe('Canvas font replacement capability', () => {
  it('does not rescan text sources for unrelated workbench notifications', () => {
    const layer = textLayer('one', from);
    const source = layer.type === 'raster' ? layer.source : undefined;
    const readSource = vi.fn(() => source);
    Object.defineProperty(layer, 'source', { get: readSource });
    const harness = createHarness(documentFrom([layer]));
    readSource.mockClear();
    for (let index = 0; index < 100; index += 1) {
      harness.notifyUnchanged();
    }
    expect(readSource).not.toHaveBeenCalled();
    harness.composition.engine.lifecycle.dispose();
  });

  it('commits one replace-all history step and restores it with undo/redo', () => {
    const document = documentFrom([groupContract('group', [textLayer('one', from), textLayer('two', from)])]);
    const harness = createHarness(document);
    const { engine } = harness.composition;

    expect(engine.fonts.collectReferences(document)).toEqual([{ count: 2, fontRef: from, layerIds: ['one', 'two'] }]);
    expect(
      engine.fonts.replaceAllReferences(from, {
        fontRef: to,
        style: 'normal',
        weight: 400,
        axes: [{ default: 400, maximum: 700, minimum: 300, tag: 'wght' }],
      })
    ).toEqual(expect.objectContaining({ replacedCount: 2, status: 'committed' }));
    expect(harness.getDocument().version).toBe(4);
    const replaced = harness.getDocument().stacks.raster[0];
    expect(replaced?.type === 'group' ? replaced.children[0] : null).toMatchObject({
      source: { fontFamily: to.family, fontRef: to },
    });
    expect(engine.stores.canUndo.get()).toBe(true);

    engine.history.undo();
    expect(harness.getDocument().stacks.raster[0]).toEqual(document.stacks.raster[0]);
    engine.history.redo();
    expect(harness.getDocument().version).toBe(4);
    expect(engine.stores.canUndo.get()).toBe(true);
    engine.lifecycle.dispose();
  });

  it('returns unchanged without adding history when no layer uses the reference', () => {
    const document = documentFrom([textLayer('other', { ...from, label: 'Other' })]);
    const harness = createHarness(document);
    const { engine } = harness.composition;

    expect(engine.fonts.replaceAllReferences(from, { fontRef: to, style: 'italic', weight: 700, axes: [] })).toEqual(
      expect.objectContaining({ replacedCount: 0, status: 'unchanged' })
    );
    expect(engine.stores.canUndo.get()).toBe(false);
    engine.lifecycle.dispose();
  });
});

describe('Canvas active font sources', () => {
  const pendingRuntime = () => {
    const signals: AbortSignal[] = [];
    const runtime: CanvasFontRuntime = {
      ensure: (_reference, signal) => {
        signals.push(signal!);
        return new Promise<string>(() => {
          // Never settles: the test observes the signal, not the face.
        });
      },
      ensureForOutput: () => Promise.reject(new Error('unused')),
      resolveFamily: (reference) => reference.family ?? 'sans-serif',
      subscribe: () => () => undefined,
    };
    return { runtime, signals };
  };

  it('keeps the text defaults loading through a session open, while an unrelated preview is dropped', () => {
    const { runtime, signals } = pendingRuntime();
    const harness = createHarness(documentFrom([]), runtime);
    const { engine } = harness.composition;
    engine.interaction.set('textOptions', {
      ...engine.interaction.get('textOptions'),
      fontFamily: from.family,
      fontRef: from,
    });
    const defaults = { ...engine.interaction.get('textOptions'), color: '#000000', content: '', type: 'text' as const };
    void engine.fonts.ensurePreview(defaults).catch(() => undefined);
    void engine.fonts.ensurePreview({ ...defaults, fontFamily: to.family, fontRef: to }).catch(() => undefined);
    expect(signals).toHaveLength(2);

    // The session's draft diverges from the defaults; only the defaults keep the first preview alive.
    engine.layers.openTextCreate({ x: 0, y: 0 });
    engine.layers.updateTextEditStyle({ fontWeight: 700 });
    expect(signals[0]!.aborted).toBe(false);
    expect(signals[1]!.aborted).toBe(true);
    engine.lifecycle.dispose();
  });
});
