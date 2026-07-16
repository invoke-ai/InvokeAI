import type {
  CommitGeneratedImageOptions,
  CommitGeneratedImageResult,
  ExportBakedLayerBlobResult,
  LayerExportGuard,
} from '@workbench/canvas-engine/engine';
import type { RunUtilityGraphOptions, UtilityGraphResult } from '@workbench/canvas-operations/backend/utilityQueue';
import type { GalleryImage } from '@workbench/gallery/api';
import type { CanvasStagingCandidateContract } from '@workbench/types';
import type {
  BuildLayerWorkflowGraphOptions,
  BuiltLayerWorkflowGraph,
  LayerWorkflowDestination,
  WorkflowImageBinding,
} from '@workbench/workflows/layerWorkflow';
import type { InvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import type { ProjectGraphState } from '@workbench/workflows/types';

export type { LayerWorkflowDestination } from '@workbench/workflows/layerWorkflow';

export type LayerWorkflowFailureStage =
  | 'export'
  | 'upload'
  | 'graph'
  | 'hydrate'
  | 'durability'
  | 'gallery'
  | 'staging'
  | 'commit';

export type RunLayerWorkflowResult =
  | { status: 'completed'; imageName: string }
  | {
      status:
        | 'aborted'
        | 'missing'
        | 'disabled'
        | 'unsupported'
        | 'empty'
        | 'not-ready'
        | 'over-budget'
        | 'locked'
        | 'stale'
        | 'busy';
    }
  | { status: 'failed'; message: string; stage: LayerWorkflowFailureStage };

export interface RunLayerWorkflowDeps {
  exportLayer(layerId: string): Promise<ExportBakedLayerBlobResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
  buildGraph(options: BuildLayerWorkflowGraphOptions): BuiltLayerWorkflowGraph;
  runGraph(
    options: Pick<RunUtilityGraphOptions, 'graph' | 'outputNodeId' | 'signal'>
  ): Promise<Pick<UtilityGraphResult, 'imageName' | 'origin'>>;
  getImage(imageName: string, signal?: AbortSignal): Promise<GalleryImage>;
  makeDurable(imageName: string): Promise<void>;
  saveToGallery(imageName: string): Promise<GalleryImage>;
  touchGallery(projectId: string): void;
  appendStaging(projectId: string, candidate: CanvasStagingCandidateContract): void;
  commitGenerated(options: CommitGeneratedImageOptions): Promise<CommitGeneratedImageResult>;
  isGuardCurrent(guard: LayerExportGuard): boolean;
  createRequestId(): string;
}

export interface RunLayerWorkflowOptions {
  layerId: string;
  projectId: string;
  document: ProjectGraphState;
  templatesSnapshot: InvocationTemplatesSnapshot;
  input: WorkflowImageBinding;
  output: WorkflowImageBinding;
  destination: LayerWorkflowDestination;
  signal?: AbortSignal;
  deps: RunLayerWorkflowDeps;
}

const isAbortError = (error: unknown): boolean =>
  typeof error === 'object' && error !== null && 'name' in error && error.name === 'AbortError';

const messageOf = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const stopped = (
  signal: AbortSignal | undefined,
  guard: LayerExportGuard,
  isGuardCurrent: (guard: LayerExportGuard) => boolean
): { status: 'aborted' | 'stale' } | null => {
  if (signal?.aborted) {
    return { status: 'aborted' };
  }
  if (!isGuardCurrent(guard)) {
    return { status: 'stale' };
  }
  return null;
};

export const runLayerWorkflow = async (options: RunLayerWorkflowOptions): Promise<RunLayerWorkflowResult> => {
  const { deps, signal } = options;
  let failureStage: LayerWorkflowFailureStage = 'export';
  if (signal?.aborted) {
    return { status: 'aborted' };
  }

  try {
    const exported = await deps.exportLayer(options.layerId);
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    if (exported.status !== 'ok') {
      return exported;
    }

    let canceled = stopped(signal, exported.guard, deps.isGuardCurrent);
    if (canceled) {
      return canceled;
    }

    failureStage = 'upload';
    const uploaded = await deps.uploadIntermediate(exported.blob, signal);
    canceled = stopped(signal, exported.guard, deps.isGuardCurrent);
    if (canceled) {
      return canceled;
    }

    failureStage = 'graph';
    const built = deps.buildGraph({
      document: options.document,
      imageName: uploaded.imageName,
      input: options.input,
      output: options.output,
      templatesSnapshot: options.templatesSnapshot,
    });
    canceled = stopped(signal, exported.guard, deps.isGuardCurrent);
    if (canceled) {
      return canceled;
    }

    const output = await deps.runGraph({ graph: built.graph, outputNodeId: built.outputNodeId, signal });
    canceled = stopped(signal, exported.guard, deps.isGuardCurrent);
    if (canceled) {
      return canceled;
    }

    failureStage = 'hydrate';
    const image = await deps.getImage(output.imageName, signal);
    canceled = stopped(signal, exported.guard, deps.isGuardCurrent);
    if (canceled) {
      return canceled;
    }

    if (options.destination === 'gallery') {
      // Saving is itself the authoritative external mutation. Once it succeeds,
      // refresh the captured project and report success even if the dialog closes
      // in the same tick.
      failureStage = 'gallery';
      await deps.saveToGallery(image.imageName);
      try {
        deps.touchGallery(options.projectId);
      } catch {
        // Refresh is ancillary once the backend has authoritatively saved the image.
      }
      return { imageName: image.imageName, status: 'completed' };
    }

    failureStage = 'durability';
    // Promotion must precede any canvas reference. If cancellation/staleness
    // wins after this non-abortable PATCH, leave the output durable but
    // unreferenced: arbitrary workflow outputs may echo/cache an existing image,
    // and the image API has no ownership token or compare-and-swap with which to
    // safely delete or mark it intermediate again.
    await deps.makeDurable(image.imageName);
    canceled = stopped(signal, exported.guard, deps.isGuardCurrent);
    if (canceled) {
      return canceled;
    }

    if (options.destination === 'staging') {
      failureStage = 'staging';
      deps.appendStaging(options.projectId, {
        ...image,
        placement: {
          height: image.height,
          opacity: 1,
          width: image.width,
          x: exported.rect.x,
          y: exported.rect.y,
        },
        sourceQueueItemId: `layer-workflow:${deps.createRequestId()}`,
      });
      return { imageName: image.imageName, status: 'completed' };
    }

    failureStage = 'commit';
    const committed = await deps.commitGenerated({
      guard: exported.guard,
      image: { height: image.height, imageName: image.imageName, width: image.width },
      origin: { x: exported.rect.x, y: exported.rect.y },
      signal,
      target: options.destination,
    });
    // The engine performs the last abort/guard check immediately before its
    // structural mutation. Preserve that authoritative result instead of
    // relabeling a successful commit after a late dialog close.
    if (committed.status === 'committed') {
      return { imageName: image.imageName, status: 'completed' };
    }
    if (committed.status === 'failed') {
      return { ...committed, stage: 'commit' };
    }
    return committed;
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) {
      return { status: 'aborted' };
    }
    return { message: messageOf(error), stage: failureStage, status: 'failed' };
  }
};
