import type {
  CanvasControlDerivationCapability,
  ControlDetectionGuard,
  ControlLayerPatchResult,
  ControlMapCommitResult,
} from '@workbench/canvas-engine/capabilities';
import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasImageRef,
} from '@workbench/canvas-engine/contracts';
import type { LayerMutationControllerOptions } from '@workbench/canvas-engine/controllers/layerMutationController';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { HistoryEntryToken } from '@workbench/canvas-engine/history/history';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import {
  controlDetectionInputKey,
  getControlDetectionInput,
  withDetectedMap,
} from '@workbench/canvas-engine/document/controlDerivation';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';

import type { CanvasMutationContext, EditStep, EditTransaction } from './mutationContext';

import {
  guardedResultRefusal,
  layerEditRefusal,
  type LayerPixels,
  replaceLayerStep,
  replayWithPixels,
  rgbaBytes,
} from './editSteps';

export interface ControlDerivationControllerOptions {
  readonly ctx: Pick<
    CanvasMutationContext,
    | 'applyStep'
    | 'begin'
    | 'capturePermit'
    | 'getDocument'
    | 'getReducerDocument'
    | 'historyTop'
    | 'installPrepared'
    | 'isPermitCurrent'
    | 'preparePixels'
    | 'projectId'
    | 'reserveRaster'
  >;
  readonly captureCache: LayerMutationControllerOptions['captureCache'];
  readonly decodeImage: (
    image: CanvasImageRef,
    options: { signal?: AbortSignal; isCurrent?: () => boolean; scaleToImage?: boolean }
  ) => Promise<DecodeImageResult>;
  readonly discardPersisted: (layerId: string) => void;
  /** The reducer state's wholesale-replacement counter; null without canvas state. */
  readonly getDocumentRevision: () => number | null;
  /** Whether the layer's live pixels are not yet represented by its persisted source. */
  readonly hasPendingPixelWork: (layerId: string) => boolean;
  /** Whether a session (transform, text, float, pixel edit, operation) owns the layer's live pixels. */
  readonly isLayerSessionHeld: (layerId: string) => boolean;
}

/** The newest entry this controller recorded, so a later patch or map can amend it while it is still on top. */
interface RecordedEntry {
  readonly token: HistoryEntryToken;
  readonly layerId: string;
  readonly label: string;
  readonly coalesceKey: string | undefined;
  /** `controlDetectionInputKey` of the entry's after-state. */
  readonly inputKey: string;
  /**
   * Whether a map for `inputKey` belongs to this entry: it changed what detection runs on (an intent that requested
   * the map), or it is a map itself. A weight or name patch is not, so undoing it never also drops a later map.
   */
  readonly requestsMap: boolean;
  /** The layer the entry's undo restores, with its unpersisted pixels when its source cannot reproduce them. */
  readonly before: CanvasControlLayerContract;
  readonly beforePixels: LayerPixels | null;
}

type BeforeState =
  | { readonly before: CanvasControlLayerContract; readonly pixels: LayerPixels | null }
  | { readonly refusal: 'not-ready' | 'over-budget' };

const asControlLayer = (
  document: CanvasDocumentContractV3 | null,
  layerId: string
): CanvasControlLayerContract | null => {
  const layer = getDocumentLayer(document, layerId);
  return layer?.type === 'control' ? layer : null;
};

const sameSource = (a: CanvasControlLayerContract, b: CanvasControlLayerContract): boolean =>
  areJsonValuesStructurallyEqual(a.source, b.source);

/**
 * Commits control-layer derivation edits: intent patches (type, detector, settings, image) and background detection
 * maps. Neither takes the operation edit owner, so both yield to a modal operation, gesture, replay or layer session
 * and the caller retries. A map lands in the same undo step as the intent that requested it while that intent is
 * still the newest step.
 */
export class ControlDerivationController implements CanvasControlDerivationCapability {
  private last: RecordedEntry | null = null;

  constructor(private readonly options: ControlDerivationControllerOptions) {}

  commitControlLayerPatch({
    coalesceKey,
    expected,
    label,
    next,
  }: Parameters<CanvasControlDerivationCapability['commitControlLayerPatch']>[0]): ControlLayerPatchResult {
    const { ctx, ...o } = this.options;
    if (next.id !== expected.id || next.type !== 'control') {
      return { message: 'A control layer patch must keep the layer id and type.', status: 'failed' };
    }
    const layerId = expected.id;
    const document = ctx.getDocument();
    const live = asControlLayer(document, layerId);
    if (!document || !live) {
      return { status: 'missing' };
    }
    if (live !== expected && !areJsonValuesStructurallyEqual(live, expected)) {
      return { status: 'stale' };
    }
    if (lookupDocumentLeaf(document, layerId)?.effectiveLocked !== false) {
      return { status: 'locked' };
    }
    if (o.isLayerSessionHeld(layerId)) {
      return { status: 'busy' };
    }
    const txn = ctx.begin({ historyBytes: HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return { status: layerEditRefusal(txn.status) };
    }
    try {
      const amended =
        coalesceKey !== undefined ? this.amendable(layerId, (entry) => entry.coalesceKey === coalesceKey) : null;
      const resolved = this.resolveBefore(txn, document, live, next, amended);
      if ('refusal' in resolved) {
        return { status: resolved.refusal };
      }
      const { before, pixels: beforePixels } = resolved;
      const after = structuredClone(next);
      const recorded = this.recordOnPublish();
      const result = txn.publish(
        amended?.label ?? label,
        this.replaceStep(after, null, live),
        {
          bytes: (beforePixels ? rgbaBytes(beforePixels.rect) : 0) + HISTORY_ENTRY_OVERHEAD_BYTES,
          dispose: recorded.dispose,
          heldAssetRefs: collectHistoryMediaRefs(before, after),
          redo: () => ctx.applyStep(this.replaceStep(after, null, before)),
          undo: () => this.restoreBefore(before, beforePixels, after),
        },
        { origin: 'user', replacing: amended?.token }
      );
      if (result.status !== 'committed') {
        return result.status === 'dispatch-rejected' || result.status === 'postcondition-failed'
          ? { status: 'failed' }
          : { status: layerEditRefusal(result.status) };
      }
      const inputKey = controlDetectionInputKey(after);
      recorded.record({
        before,
        beforePixels,
        coalesceKey,
        inputKey,
        label: amended?.label ?? label,
        layerId,
        requestsMap: controlDetectionInputKey(before) !== inputKey,
        token: result.token,
      });
      return { status: 'committed' };
    } finally {
      txn.end();
    }
  }

  captureControlDetectionGuard(layerId: string): ControlDetectionGuard | null {
    // A background read: the reducer document, so an open structural preview stays open.
    const layer = asControlLayer(this.options.ctx.getReducerDocument(), layerId);
    const documentRevision = this.options.getDocumentRevision();
    if (!layer || documentRevision === null) {
      return null;
    }
    return {
      documentRevision,
      inputKey: controlDetectionInputKey(layer),
      layerId,
      projectId: this.options.ctx.projectId,
    };
  }

  async commitControlMap({
    detection,
    guard,
    image,
    makeDurable,
    rect,
    signal,
  }: Parameters<CanvasControlDerivationCapability['commitControlMap']>[0]): Promise<ControlMapCommitResult> {
    const { ctx, ...o } = this.options;
    const permit = ctx.capturePermit();
    if (!permit) {
      return { status: 'busy' };
    }
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    /**
     * Everything the map was computed against must still hold. It reads the reducer document, so a refusal leaves an
     * open structural preview alone; only an admitted commit ends it.
     */
    const check = ():
      | { document: CanvasDocumentContractV3; live: CanvasControlLayerContract }
      | ControlMapCommitResult => {
      if (signal?.aborted) {
        return { status: 'aborted' };
      }
      if (!ctx.isPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      const document = ctx.getReducerDocument();
      const live = asControlLayer(document, guard.layerId);
      if (!document || !live) {
        return { status: 'missing' };
      }
      if (o.getDocumentRevision() !== guard.documentRevision || controlDetectionInputKey(live) !== guard.inputKey) {
        return { status: 'stale' };
      }
      if (lookupDocumentLeaf(document, live.id)?.effectiveLocked !== false) {
        return { status: 'locked' };
      }
      // Pixels painted on the input itself after the run started are not in the original it names yet.
      const input = getControlDetectionInput(live);
      const inputIsPendingSource = live.original === undefined && input !== null && o.hasPendingPixelWork(live.id);
      if (o.isLayerSessionHeld(live.id) || inputIsPendingSource) {
        return { status: 'busy' };
      }
      return { document, live };
    };
    // Refusals the caller retries are cheap to find; find them before decoding.
    const early = check();
    if ('status' in early) {
      return early;
    }
    const mapRect = { ...rect };
    const map = { image: structuredClone(image), rect: mapRect };
    if (!withDetectedMap(early.live, map, detection)) {
      return { status: 'unrepresentable' };
    }
    try {
      // Promoted only once nothing refuses it, so a result that is dropped stays collectible.
      await makeDurable();
      const decoded = await o.decodeImage(image, {
        isCurrent: () => ctx.isPermitCurrent(permit),
        scaleToImage: false,
        signal,
      });
      if (decoded.status !== 'ok') {
        return { status: decoded.status === 'aborted' ? 'aborted' : 'busy' };
      }
      const decodedCheck = check();
      if ('status' in decodedCheck) {
        return decodedCheck;
      }
      const afterPixels: LayerPixels = { pixels: decoded.surface, rect: mapRect };
      const txn = ctx.begin({ historyBytes: rgbaBytes(mapRect) + HISTORY_ENTRY_OVERHEAD_BYTES });
      if (!('publish' in txn)) {
        return { status: layerEditRefusal(txn.status) };
      }
      try {
        // Admission ended any structural preview; the map lands on the committed layer, never on previewed values.
        const checked = check();
        if ('status' in checked) {
          return checked;
        }
        const { document, live } = checked;
        const after = withDetectedMap(live, map, structuredClone(detection));
        if (!after) {
          return { status: 'unrepresentable' };
        }
        if (!txn.reserveRaster(rgbaBytes(mapRect))) {
          return { status: 'over-budget' };
        }
        const paired = this.amendable(live.id, (entry) => entry.requestsMap && entry.inputKey === guard.inputKey);
        const resolved = this.resolveBefore(txn, document, live, after, paired);
        if ('refusal' in resolved) {
          return { status: resolved.refusal };
        }
        const { before, pixels: beforePixels } = resolved;
        const prepared = ctx.preparePixels(live.id, mapRect, afterPixels.pixels);
        const label = paired?.label ?? `Detect ${after.controlType ?? 'control'}`;
        const recorded = this.recordOnPublish();
        const result = txn.publish(
          label,
          this.replaceStep(after, prepared, live),
          {
            bytes:
              (beforePixels ? rgbaBytes(beforePixels.rect) : 0) + rgbaBytes(mapRect) + HISTORY_ENTRY_OVERHEAD_BYTES,
            dispose: recorded.dispose,
            heldAssetRefs: collectHistoryMediaRefs(before, after),
            redo: () =>
              replayWithPixels(ctx, after.id, afterPixels, (replayed) => this.replaceStep(after, replayed, before)),
            undo: () => this.restoreBefore(before, beforePixels, after),
          },
          // A background result is not a user edit: it must never change the invocation route.
          { origin: 'system', replacing: paired?.token }
        );
        if (result.status !== 'committed') {
          return { status: guardedResultRefusal(result) };
        }
        recorded.record({
          before,
          beforePixels,
          coalesceKey: undefined,
          inputKey: controlDetectionInputKey(after),
          label,
          layerId: after.id,
          requestsMap: true,
          token: result.token,
        });
        return { status: 'committed' };
      } finally {
        txn.end();
      }
    } catch (error) {
      if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        return { status: 'aborted' };
      }
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  }

  dispose(): void {
    this.last = null;
  }

  /**
   * Ties the record of an entry about to publish to that entry's lifetime: its `dispose` (eviction, a cleared redo
   * stack, replacement by an amending commit) drops the record, so its before-pixels never outlive the entry.
   */
  private recordOnPublish(): { dispose: () => void; record: (entry: RecordedEntry) => void } {
    let recorded: RecordedEntry | null = null;
    return {
      dispose: () => {
        if (recorded && this.last === recorded) {
          this.last = null;
        }
        recorded = null;
      },
      record: (entry) => {
        recorded = entry;
        this.last = entry;
      },
    };
  }

  /**
   * The recorded entry a new commit may replace: still the newest step, for this layer, and matching. A record whose
   * entry is not on top is dropped with the pixels it holds, trading pairing for memory: if a redo later puts that
   * entry back on top, its map records its own entry instead of joining it.
   */
  private amendable(layerId: string, matches: (entry: RecordedEntry) => boolean): RecordedEntry | null {
    const entry = this.last;
    if (!entry) {
      return null;
    }
    if (entry.token !== this.options.ctx.historyTop()) {
      this.last = null;
      return null;
    }
    return entry.layerId === layerId && matches(entry) ? entry : null;
  }

  /**
   * The layer one undo returns to, and the pixels it needs: an amended entry's own before-state, else the live layer.
   * Pixels are kept only when the step replaces a source whose pixels are not persisted yet; otherwise the restored
   * contract's persisted source reproduces them exactly.
   */
  private resolveBefore(
    txn: EditTransaction,
    document: CanvasDocumentContractV3,
    live: CanvasControlLayerContract,
    after: CanvasControlLayerContract,
    amended: RecordedEntry | null
  ): BeforeState {
    const before = amended?.before ?? structuredClone(live);
    if (amended?.beforePixels) {
      return { before, pixels: amended.beforePixels };
    }
    if (sameSource(before, after) || !sameSource(before, live) || !this.options.hasPendingPixelWork(live.id)) {
      return { before, pixels: null };
    }
    const pixels = this.options.captureCache(
      live,
      document,
      (rect) => txn.growHistory(rgbaBytes(rect)) && txn.reserveRaster(rgbaBytes(rect))
    );
    if (pixels === 'not-ready' || pixels === 'over-budget') {
      return { refusal: pixels };
    }
    // Unpersisted pixels without a cache cannot be restored by undo.
    return pixels ? { before, pixels } : { refusal: 'not-ready' };
  }

  /**
   * Swaps `layer` in. A source change drops the outgoing source's pending persistence, which would otherwise land
   * its old pixels over the new source.
   */
  private replaceStep(
    layer: CanvasControlLayerContract,
    prepared: Parameters<typeof replaceLayerStep>[2],
    restore: CanvasControlLayerContract,
    persist = false
  ): EditStep {
    return replaceLayerStep(this.options.ctx, layer, prepared, {
      beforeInstall: sameSource(layer, restore) ? undefined : () => this.options.discardPersisted(layer.id),
      persist,
      restore,
    });
  }

  private restoreBefore(
    before: CanvasControlLayerContract,
    pixels: LayerPixels | null,
    after: CanvasControlLayerContract
  ): void {
    const { ctx } = this.options;
    if (!pixels) {
      ctx.applyStep(this.replaceStep(before, null, after));
      return;
    }
    // Restored pixels were never persisted, so they persist again.
    replayWithPixels(ctx, before.id, pixels, (prepared) => this.replaceStep(before, prepared, after, true));
  }
}
