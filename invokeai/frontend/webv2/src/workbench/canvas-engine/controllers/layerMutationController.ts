import type { DuplicateLayersResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterMemoryReservationResult } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { History } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

export type CapturedLayerCache = { pixels: RasterSurface; rect: Rect } | null | 'not-ready';

export type DuplicateLayerRasterPlan =
  | {
      readonly captureBytes: number;
      readonly initialReserveBytes: number;
      readonly replayReserveBytes: number;
      readonly retainForHistory: boolean;
      readonly type: 'capture';
    }
  | { readonly type: 'empty' }
  | { readonly type: 'reference' }
  | { readonly type: 'not-ready' };

type DuplicateRasterPreparationResult =
  | { readonly status: 'ready'; readonly layer: CanvasLayerContract }
  | { readonly status: 'not-ready' | 'over-budget' };

export interface LayerMutationControllerOptions<Permit> {
  readonly canEdit: () => boolean;
  readonly capturePermit: () => Permit | null;
  readonly captureCache: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => CapturedLayerCache;
  readonly createLayerId: () => string;
  readonly discardPersisted: (layerId: string) => void;
  readonly dispatchPrepared: (
    action: CanvasProjectMutation,
    reducerAccepted: () => boolean,
    mirrorAccepted: () => boolean
  ) => void;
  readonly endBurst: () => void;
  readonly getDuplicateRasterPlan: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV2
  ) => DuplicateLayerRasterPlan;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getReducerDocument: () => CanvasDocumentContractV2 | null;
  readonly getSelectedLayerIds: (document: CanvasDocumentContractV2) => readonly string[];
  readonly history: History;
  readonly hasPendingPixelWork: (layerId: string) => boolean;
  readonly installPrepared: (prepared: PreparedLayerCacheReplacement, persist?: boolean) => void;
  readonly isGestureActive: () => boolean;
  readonly isPermitCurrent: (permit: Permit) => boolean;
  readonly needsPixelPersistence: (layer: CanvasLayerContract) => boolean;
  readonly preparePixels: (layerId: string, rect: Rect, pixels: RasterSurface) => PreparedLayerCacheReplacement;
  readonly publishSelectedLayerIds: (primaryId: string | null, selectedIds: readonly string[]) => void;
  readonly prepareDuplicateRasterSource: (layerId: string) => Promise<DuplicateRasterPreparationResult>;
  readonly pinDuplicateRasterSources: (layerIds: readonly string[]) => { release(): void };
  readonly reserve: (bytes: number) => RasterMemoryReservationResult;
  readonly scheduleDuplicateRasterization: (layerIds: readonly string[]) => void;
  readonly sameContract: (document: CanvasDocumentContractV2 | null, layer: CanvasLayerContract) => boolean;
  readonly trackDetached: (bytes: number) => { release(): void };
}

/** Owns failure-atomic copy and cross-type conversion mutations. */
export class LayerMutationController<Permit> {
  private duplicateInFlight = false;

  constructor(private readonly options: LayerMutationControllerOptions<Permit>) {}

  async duplicate(layerIds: readonly string[]): Promise<DuplicateLayersResult> {
    const o = this.options;
    const permit = o.capturePermit();
    if (this.duplicateInFlight || !permit || !o.canEdit() || o.isGestureActive()) {
      return { status: 'busy' };
    }
    const document = o.getDocument();
    if (!document) {
      return { status: 'nothing' };
    }
    const requested = new Set(layerIds);
    const sources = document.layers.filter((layer) => requested.has(layer.id));
    if (sources.length === 0 || sources.length !== requested.size) {
      return { status: 'nothing' };
    }
    const plans = sources.map((source) => o.getDuplicateRasterPlan(source, document));
    const notReadySources = sources.filter((_source, index) => plans[index]?.type === 'not-ready');
    if (notReadySources.length === 0) {
      return this.commitDuplicate(layerIds);
    }

    this.duplicateInFlight = true;
    const pinLease = o.pinDuplicateRasterSources(sources.map((source) => source.id));
    try {
      for (const source of notReadySources) {
        if (o.hasPendingPixelWork(source.id)) {
          return { status: 'not-ready' };
        }
        const prepared = await o.prepareDuplicateRasterSource(source.id);
        if (prepared.status !== 'ready') {
          return { status: prepared.status };
        }
        if (prepared.layer !== source) {
          return { status: 'stale' };
        }
      }
      if (!o.isPermitCurrent(permit) || o.isGestureActive() || o.getDocument() !== document) {
        return { status: 'stale' };
      }
      return this.commitDuplicate(layerIds);
    } finally {
      pinLease.release();
      this.duplicateInFlight = false;
    }
  }

  private commitDuplicate(layerIds: readonly string[]): DuplicateLayersResult {
    const o = this.options;
    if (!o.canEdit() || o.isGestureActive()) {
      return { status: 'busy' };
    }
    o.endBurst();
    const document = o.getDocument();
    if (!document) {
      return { status: 'nothing' };
    }
    const requested = new Set(layerIds);
    const sources = document.layers.filter((layer) => requested.has(layer.id));
    if (sources.length === 0 || sources.length !== requested.size) {
      return { status: 'nothing' };
    }
    const plans = sources.map((source) => o.getDuplicateRasterPlan(source, document));
    let retainedBytes = 0;
    let reserveBytes = 0;
    for (const plan of plans) {
      if (plan.type === 'not-ready') {
        return { status: 'not-ready' };
      }
      if (plan.type === 'capture') {
        if (plan.retainForHistory) {
          retainedBytes += plan.captureBytes;
        }
        reserveBytes += plan.initialReserveBytes;
      }
    }
    const historyBytes = retainedBytes + sources.length * 256;
    if (!o.history.canRetain(historyBytes)) {
      return { status: 'over-budget' };
    }
    // Durable layer sources are immutable, so their one prepared cache is used
    // only for the initial insertion and can be reconstructed from the source
    // on redo. Live paint/mask pixels that have not reached a durable source
    // retain a separate immutable history capture and therefore need a second
    // live cache. This makes the common path one full-size copy while keeping
    // dirty pixels exact and every allocation inside the raster reservation.
    const reservation = o.reserve(reserveBytes);
    if (reservation.status === 'over-budget') {
      return { status: 'over-budget' };
    }
    try {
      const captures = sources.map((source, index) =>
        plans[index]?.type === 'capture' ? o.captureCache(source, document) : null
      );
      if (captures.some((capture) => capture === 'not-ready')) {
        return { status: 'not-ready' };
      }
      const existingIds = new Set(document.layers.map((layer) => layer.id));
      const duplicates = sources.map((source) => {
        const duplicate = structuredClone(source);
        duplicate.id = o.createLayerId();
        duplicate.name = `${source.name} copy`;
        return duplicate;
      });
      for (const [index, duplicate] of duplicates.entries()) {
        if (plans[index]?.type !== 'empty') {
          continue;
        }
        if (duplicate.type === 'raster' || duplicate.type === 'control') {
          if (duplicate.source.type === 'paint') {
            duplicate.source = { bitmap: null, type: 'paint' };
          }
        } else if (duplicate.type === 'regional_guidance' || duplicate.type === 'inpaint_mask') {
          duplicate.mask = { ...duplicate.mask, bitmap: null, offset: { x: 0, y: 0 } };
        }
      }
      if (
        duplicates.some((layer) => existingIds.has(layer.id)) ||
        new Set(duplicates.map((layer) => layer.id)).size !== duplicates.length
      ) {
        return { status: 'stale' };
      }
      const duplicateBySource = new Map(sources.map((source, index) => [source.id, duplicates[index]!]));
      const orderedIds = document.layers.flatMap((layer) => {
        const duplicate = duplicateBySource.get(layer.id);
        return duplicate ? [duplicate.id, layer.id] : [layer.id];
      });
      const selectedLayerId =
        (document.selectedLayerId ? duplicateBySource.get(document.selectedLayerId)?.id : null) ?? duplicates[0]!.id;
      const previousSelectedLayerId = document.selectedLayerId;
      const previousSelectedIds = [...o.getSelectedLayerIds(document)];
      const originalIds = document.layers.map((layer) => layer.id);
      const duplicateIds = duplicates.map((layer) => layer.id);
      const hasDuplicates = (candidate: CanvasDocumentContractV2 | null): boolean =>
        candidate?.selectedLayerId === selectedLayerId &&
        candidate.layers.length === orderedIds.length &&
        candidate.layers.every((layer, index) => layer.id === orderedIds[index]);
      const hasOriginals = (candidate: CanvasDocumentContractV2 | null): boolean =>
        candidate?.selectedLayerId === previousSelectedLayerId &&
        candidate.layers.length === originalIds.length &&
        candidate.layers.every((layer, index) => layer.id === originalIds[index]);
      const applyPrepared = (
        prepared: readonly { duplicate: CanvasLayerContract; replacement: PreparedLayerCacheReplacement }[]
      ): void => {
        o.dispatchPrepared(
          {
            add: { index: 0, layers: duplicates },
            enabledUpdates: [],
            orderedIds,
            selectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          () => hasDuplicates(o.getReducerDocument()),
          () => hasDuplicates(o.getDocument())
        );
        prepared.forEach(({ duplicate, replacement }) => {
          o.installPrepared(replacement, o.needsPixelPersistence(duplicate));
        });
        o.publishSelectedLayerIds(selectedLayerId, duplicateIds);
      };
      const retainedCaptures = captures.map((capture, index) =>
        plans[index]?.type === 'capture' && plans[index].retainForHistory ? capture : null
      );
      const initialPrepared = captures.flatMap((capture, index) => {
        if (!capture || capture === 'not-ready') {
          return [];
        }
        const duplicate = duplicates[index]!;
        const plan = plans[index]!;
        return [
          {
            duplicate,
            replacement:
              plan.type === 'capture' && plan.retainForHistory
                ? o.preparePixels(duplicate.id, capture.rect, capture.pixels)
                : { layerId: duplicate.id, rect: capture.rect, surface: capture.pixels },
          },
        ];
      });
      applyPrepared(initialPrepared);
      initialPrepared.length = 0;
      captures.forEach((_capture, index) => {
        const plan = plans[index];
        if (plan?.type !== 'capture' || !plan.retainForHistory) {
          captures[index] = null;
        }
      });
      const detachedLease = retainedBytes > 0 ? o.trackDetached(retainedBytes) : null;
      const redo = (): void => {
        const current = o.getDocument();
        if (!current) {
          throw new Error('Canvas document is not ready to restore duplicated layers');
        }
        const replayPlans = sources.map((_source, index) => {
          const originalPlan = plans[index]!;
          if (originalPlan.type === 'capture' && originalPlan.retainForHistory) {
            return originalPlan;
          }
          return originalPlan.type === 'capture' ? ({ type: 'reference' } as const) : originalPlan;
        });
        const replayBytes = replayPlans.reduce(
          (total, plan) => total + (plan.type === 'capture' ? plan.replayReserveBytes : 0),
          0
        );
        const replayReservation = o.reserve(replayBytes);
        if (replayReservation.status === 'over-budget') {
          throw new Error('Not enough raster memory to restore duplicated layers');
        }
        try {
          const prepared = replayPlans.flatMap((plan, index) => {
            if (plan.type !== 'capture') {
              return [];
            }
            const duplicate = duplicates[index]!;
            const retained = retainedCaptures[index];
            if (!retained || retained === 'not-ready') {
              throw new Error('Layer pixels are not ready to restore duplicated layers');
            }
            return [
              {
                duplicate,
                replacement: o.preparePixels(duplicate.id, retained.rect, retained.pixels),
              },
            ];
          });
          applyPrepared(prepared);
          o.scheduleDuplicateRasterization(
            replayPlans.flatMap((plan, index) =>
              plan.type === 'reference' && plans[index]?.type === 'capture' ? [duplicates[index]!.id] : []
            )
          );
        } finally {
          replayReservation.lease.release();
        }
      };
      o.history.push({
        bytes: historyBytes,
        dispose: () => detachedLease?.release(),
        label: duplicates.length === 1 ? 'Duplicate layer' : 'Duplicate layers',
        redo,
        replayFailureAtomic: true,
        undo: () => {
          o.dispatchPrepared(
            {
              enabledUpdates: [],
              removeIds: duplicateIds,
              selectedLayerId: previousSelectedLayerId,
              type: 'applyCanvasLayerStackMutation',
            },
            () => hasOriginals(o.getReducerDocument()),
            () => hasOriginals(o.getDocument())
          );
          o.publishSelectedLayerIds(previousSelectedLayerId, previousSelectedIds);
        },
      });
      return { duplicateIds, selectedLayerId, status: 'duplicated' };
    } finally {
      reservation.lease.release();
    }
  }

  copy(label: string, sourceLayerId: string, layer: CanvasLayerContract, index: number): boolean {
    const o = this.options;
    if (!o.canEdit() || o.isGestureActive()) {
      return false;
    }
    o.endBurst();
    const document = o.getDocument();
    const source = document?.layers.find((candidate) => candidate.id === sourceLayerId);
    if (!document || !source || document.layers.some((candidate) => candidate.id === layer.id)) {
      return false;
    }
    const captured = o.captureCache(source, document);
    if (captured === 'not-ready') {
      return false;
    }
    const selectedLayerId = document.selectedLayerId;
    const apply = (): void => {
      const prepared = captured ? o.preparePixels(layer.id, captured.rect, captured.pixels) : null;
      o.dispatchPrepared(
        {
          add: { index, layers: [layer] },
          enabledUpdates: [],
          selectedLayerId: layer.id,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          o.getReducerDocument()?.selectedLayerId === layer.id &&
          o.getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () =>
          o.getDocument()?.selectedLayerId === layer.id &&
          o.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
      if (prepared) {
        o.installPrepared(prepared, o.needsPixelPersistence(layer));
      }
    };
    apply();
    o.history.push({
      bytes: captured ? captured.rect.width * captured.rect.height * 4 + 256 : 256,
      label,
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        o.dispatchPrepared(
          { enabledUpdates: [], removeIds: [layer.id], selectedLayerId, type: 'applyCanvasLayerStackMutation' },
          () =>
            o.getReducerDocument()?.selectedLayerId === selectedLayerId &&
            o.getReducerDocument()?.layers.some((candidate) => candidate.id === layer.id) === false,
          () =>
            o.getDocument()?.selectedLayerId === selectedLayerId &&
            o.getDocument()?.layers.some((candidate) => candidate.id === layer.id) === false
        ),
    });
    return true;
  }

  convert(label: string, expected: CanvasLayerContract, after: CanvasLayerContract): boolean {
    const o = this.options;
    if (!o.canEdit() || o.isGestureActive() || expected.id !== after.id || expected.type === after.type) {
      return false;
    }
    o.endBurst();
    const document = o.getDocument();
    const current = document?.layers.find((candidate) => candidate.id === expected.id);
    if (!document || !current || current !== expected || current.isLocked || current.type !== expected.type) {
      return false;
    }
    const captured = o.captureCache(current, document);
    if (captured === 'not-ready') {
      return false;
    }
    const apply = (layer: CanvasLayerContract): void => {
      const prepared = captured ? o.preparePixels(layer.id, captured.rect, captured.pixels) : null;
      o.dispatchPrepared(
        { id: layer.id, layer, targetType: layer.type, type: 'convertCanvasLayer' },
        () => o.sameContract(o.getReducerDocument(), layer),
        () => o.sameContract(o.getDocument(), layer)
      );
      try {
        o.discardPersisted(layer.id);
      } catch {
        /* Ancillary after reducer acceptance. */
      }
      if (prepared) {
        o.installPrepared(prepared, o.needsPixelPersistence(layer));
      }
    };
    const before = structuredClone(current);
    apply(after);
    o.history.push({
      bytes: captured ? captured.rect.width * captured.rect.height * 4 + 256 : 256,
      label,
      redo: () => apply(after),
      replayFailureAtomic: true,
      undo: () => apply(before),
    });
    return true;
  }

  dispose(): void {}
}
