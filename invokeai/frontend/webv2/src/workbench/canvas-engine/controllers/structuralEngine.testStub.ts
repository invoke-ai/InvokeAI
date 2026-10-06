import type { CanvasDocumentContractV3, CanvasNodeContract } from '@workbench/canvas-engine/contracts';
import type { PreparedDocumentEdit } from '@workbench/canvas-engine/document-model/documentCommands';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Project } from '@workbench/projectContracts';

import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createDocumentModel } from '@workbench/canvas-engine/document-model/documentModel';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { vi } from 'vitest';

import { HistoryController } from './historyController';
import { createCanvasMutationContext, type EditStepError } from './mutationContext';
import { StructuralLayerController } from './structuralLayerController';

export interface StructuralEngineStubOptions {
  /** Top-first nodes of the document; one paint layer `layer` named `Layer` by default. */
  layers?: readonly CanvasNodeContract[];
  /** Defaults to the first layer. */
  selectedLayerId?: string | null;
  /** A live flag lets a test lock edits mid-gesture. */
  locked?: boolean | { value: boolean };
  gestureActive?: boolean;
  schedulePreview?: (flush: () => void) => () => void;
  /** Mirror refreshes only when asked, as when a store observer threw mid-notification. */
  mirrorLag?: boolean;
  /** Makes mirror refresh fail so an accepted mutation can never be mirrored. */
  mirrorBroken?: boolean;
  now?: () => number;
}

/**
 * The structural half of a canvas engine over the real reducer: a project document, the mutation context, engine
 * history with its replay guards, and the structural controller wired the way the engine wires them. Widgets that
 * preview and commit structural edits can run against `engine` and observe the document they change.
 */
export const createStructuralEngineStub = (options: StructuralEngineStubOptions = {}) => {
  const base = createInitialWorkbenchState().projects[0]!;
  const layers = options.layers ?? [layerContract('layer', 'raster', { name: 'Layer' })];
  const selectedLayerId = options.selectedLayerId === undefined ? (layers[0]?.id ?? null) : options.selectedLayerId;
  let project: Project = { ...base, canvas: { ...base.canvas, document: documentFrom(layers, selectedLayerId) } };
  let mirrorDocument = project.canvas.document;
  const listeners = new Set<() => void>();
  const dispatched: CanvasProjectMutation[] = [];
  const report = vi.fn();
  const refreshMirror = (): void => {
    if (options.mirrorBroken) {
      throw new Error('mirror broken');
    }
    mirrorDocument = project.canvas.document;
  };
  const dispatch = (action: CanvasProjectMutation): boolean => {
    dispatched.push(action);
    const next = applyCanvasProjectMutation(project, action);
    const changed = next.canvas !== project.canvas;
    project = next;
    if (!options.mirrorLag && !options.mirrorBroken) {
      mirrorDocument = project.canvas.document;
    }
    listeners.forEach((listener) => listener());
    return changed;
  };
  const isLocked = (): boolean =>
    typeof options.locked === 'object' ? options.locked.value : (options.locked ?? false);
  const isGestureActive = (): boolean => options.gestureActive ?? false;
  const historyController = new HistoryController({
    beforeReplay: () => controller.endPreview(),
    canEdit: () => ctx.canEdit(),
    isGestureActive,
    reportFailure: (label, error) => report('History replay failed', label, error),
  });
  const history = historyController.history;
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'new',
    projectId: base.id,
    dispatch,
    editOwner: Symbol('owner'),
    editingLocked: { get: isLocked, subscribe: () => () => undefined },
    getDocument: () => mirrorDocument,
    getReducerDocument: () => project.canvas.document,
    history,
    installPrepared: () => undefined,
    isGestureActive,
    isGuardCurrent: () => true,
    preparePixels: () => ({}) as never,
    refreshMirror,
    // As the engine wires it: unrecoverable step failures are reported with the edit's label.
    report: (error: EditStepError, label: string) =>
      report(
        error.outcome === 'reverted-unmirrored'
          ? 'Structural edit could not be mirrored'
          : 'Structural edit could not be reverted',
        label,
        error
      ),
    reserveRaster: () => ({ lease: { release: () => undefined }, status: 'ok' }),
    subscribeReducer: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  });
  const controller = new StructuralLayerController({
    ctx,
    now: options.now ?? (() => 0),
    report,
    schedulePreview: options.schedulePreview,
  });
  /** Every prepared edit that landed, through a session or directly, in order. */
  const commits: PreparedDocumentEdit[] = [];
  const engine = {
    document: {
      model: () => createDocumentModel(mirrorDocument, { editRevision: ctx.getEditRevision(), projectId: base.id }),
    },
    history: {
      canRedo: () => history.canRedo(),
      canUndo: () => history.canUndo(),
      redo: async () => (await historyController.redo()).status,
      undo: async () => (await historyController.undo()).status,
    },
    layers: {
      beginStructuralPreview: () => {
        const session = controller.beginPreview();
        if (!session) {
          return null;
        }
        return {
          ...session,
          commit: (label: string, edit: PreparedDocumentEdit) => {
            const result = session.commit(label, edit);
            if (result.status === 'committed') {
              commits.push(edit);
            }
            return result;
          },
        };
      },
      canCommitStructural: () => controller.canCommit(),
      commitPrepared: (label: string, edit: PreparedDocumentEdit) => {
        const result = controller.commitPrepared(label, edit);
        if (result.status === 'committed') {
          commits.push(edit);
        }
        return result;
      },
      commitStructural: controller.commit.bind(controller),
      endStructuralPreview: () => controller.endPreview(),
    },
    projectId: base.id,
  };
  return {
    commits,
    controller,
    ctx,
    dispatched,
    document: (): CanvasDocumentContractV3 => project.canvas.document,
    engine,
    history,
    historyController,
    mirror: (): CanvasDocumentContractV3 => mirrorDocument,
    project: (): Project => project,
    projectId: base.id,
    report,
    subscribe: (listener: () => void): (() => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

export type StructuralEngineStub = ReturnType<typeof createStructuralEngineStub>;
