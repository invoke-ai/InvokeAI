import type { StructuralCommitResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { LayerStackKind } from '@workbench/canvas-engine/document/layerStacks';
import type {
  ActiveColorPairState,
  TextEditSession,
  TextSource,
  TextStylePatch,
  TextToolOptions,
} from '@workbench/canvas-engine/engineStores';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';

export interface TextEditingControllerOptions {
  readonly session: {
    get(): TextEditSession | null;
    set(value: TextEditSession | null): void;
  };
  readonly options: { get(): TextToolOptions };
  /** The active pair; a new session's color is the foreground at open. */
  readonly colors: { get(): ActiveColorPairState };
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly createLayerId: () => string;
  readonly captureInsertionAnchor: (stack: LayerStackKind, aboveId: string | null) => CanvasNodeInsertionAnchor;
  readonly commitStructural: (
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation
  ) => StructuralCommitResult;
  readonly invalidate: (payload: { layers?: string[]; overlay?: true }) => void;
}

const fontRefEqual = (left: TextSource['fontRef'], right: TextSource['fontRef']): boolean =>
  left === right ||
  (left !== undefined &&
    right !== undefined &&
    left.id === right.id &&
    left.contentHash === right.contentHash &&
    left.family === right.family &&
    left.label === right.label);

const fontVariationsEqual = (left: TextSource['fontVariations'], right: TextSource['fontVariations']): boolean => {
  const leftEntries = Object.entries(left ?? {});
  const rightEntries = Object.entries(right ?? {});
  return (
    leftEntries.length === rightEntries.length &&
    leftEntries.every(([tag, value]) => (right?.[tag] ?? undefined) === value)
  );
};

const sourcesEqual = (left: TextSource, right: TextSource): boolean =>
  left.content === right.content &&
  left.fontFamily === right.fontFamily &&
  left.fontSize === right.fontSize &&
  left.fontWeight === right.fontWeight &&
  fontRefEqual(left.fontRef, right.fontRef) &&
  (left.fontStyle ?? 'normal') === (right.fontStyle ?? 'normal') &&
  fontVariationsEqual(left.fontVariations, right.fontVariations) &&
  left.lineHeight === right.lineHeight &&
  left.align === right.align &&
  left.color === right.color;

/** Keeps default typography out of new documents while accepting older explicit defaults. */
const canonicalizeSource = (source: TextSource): TextSource => {
  const next: TextSource = { ...source };
  if (next.fontRef) {
    next.fontRef = { ...next.fontRef };
  }
  if (next.fontStyle === 'normal') {
    delete next.fontStyle;
  }
  if (!next.fontVariations || Object.keys(next.fontVariations).length === 0) {
    delete next.fontVariations;
  } else {
    next.fontVariations = { ...next.fontVariations };
  }
  return next;
};

/** Owns create/edit text session state and its structural commits. */
export class TextEditingController {
  private sessionId = 0;
  private contentReader: (() => string) | null = null;
  private disposed = false;

  constructor(private readonly deps: TextEditingControllerOptions) {}

  private sourceFromOptions(content: string): TextSource {
    const options = this.deps.options.get();
    const source: TextSource = {
      align: options.align,
      color: this.deps.colors.get().foreground,
      content,
      fontFamily: options.fontFamily,
      fontSize: options.fontSize,
      fontWeight: options.fontWeight,
      lineHeight: options.lineHeight,
      type: 'text',
    };
    if (options.fontRef) {
      source.fontRef = { ...options.fontRef };
    }
    if (options.fontStyle && options.fontStyle !== 'normal') {
      source.fontStyle = options.fontStyle;
    }
    if (options.fontVariations && Object.keys(options.fontVariations).length > 0) {
      source.fontVariations = { ...options.fontVariations };
    }
    return canonicalizeSource(source);
  }

  setContentReader(reader: (() => string) | null): void {
    this.contentReader = reader;
  }

  openCreate(point: Vec2): void {
    if (this.disposed || !this.deps.getDocument()) {
      return;
    }
    this.deps.session.set({
      id: ++this.sessionId,
      layerId: null,
      mode: 'create',
      source: this.sourceFromOptions(''),
      startSource: null,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: Math.round(point.x), y: Math.round(point.y) },
    });
    this.deps.invalidate({ overlay: true });
  }

  openEdit(layerId: string): void {
    if (this.disposed) {
      return;
    }
    const document = this.deps.getDocument();
    const leaf = lookupDocumentLeaf(document, layerId);
    const layer = leaf?.layer;
    if (!document || !layer || layer.type !== 'raster' || layer.source.type !== 'text' || !isLeafEditable(leaf)) {
      return;
    }
    this.deps.session.set({
      id: ++this.sessionId,
      layerId,
      mode: 'edit',
      source: canonicalizeSource(layer.source),
      startSource: canonicalizeSource(layer.source),
      transform: { ...layer.transform },
    });
    this.deps.invalidate({ layers: [layerId] });
  }

  updateStyle(patch: TextStylePatch): void {
    const session = this.deps.session.get();
    if (this.disposed || !session) {
      return;
    }
    const source = canonicalizeSource({ ...session.source, ...patch });
    if ('fontRef' in patch) {
      source.fontRef = patch.fontRef ? { ...patch.fontRef } : undefined;
    }
    this.deps.session.set({ ...session, source });
  }

  cancel(): void {
    const session = this.deps.session.get();
    if (!session) {
      return;
    }
    this.deps.session.set(null);
    this.deps.invalidate(session.layerId ? { layers: [session.layerId] } : { overlay: true });
  }

  commit(content: string, styleChanges?: Partial<TextToolOptions>): StructuralCommitResult | null {
    if (this.disposed) {
      return { status: 'not-ready' };
    }
    if (!this.deps.canEdit()) {
      return { status: 'busy' };
    }
    if (this.deps.isGestureActive()) {
      return { status: 'gesture-active' };
    }
    const session = this.deps.session.get();
    if (!session) {
      return null;
    }
    const finalSource = canonicalizeSource({ ...session.source, ...styleChanges, content });
    if (session.mode === 'create') {
      if (content.trim() === '') {
        this.cancel();
        return null;
      }
      const layerId = this.deps.createLayerId();
      const layer: CanvasLayerContract = {
        blendMode: 'normal',
        id: layerId,
        isEnabled: true,
        isLocked: false,
        name: `Text ${(getDocumentLeaves(this.deps.getDocument() ?? null).length ?? 0) + 1}`,
        opacity: 1,
        source: finalSource,
        transform: session.transform,
        type: 'raster',
      };
      const added = this.deps.commitStructural(
        'Add text',
        {
          anchor: this.deps.captureInsertionAnchor('raster', this.deps.getDocument()?.selectedLayerId ?? null),
          layer,
          type: 'addCanvasLayer',
        },
        { ids: [layerId], type: 'removeCanvasLayers' }
      );
      this.settle(added);
      this.deps.invalidate({ overlay: true });
      return added;
    }
    const { layerId, startSource } = session;
    if (!layerId || !startSource) {
      this.cancel();
      return null;
    }
    if (sourcesEqual(startSource, finalSource)) {
      this.deps.session.set(null);
      this.deps.invalidate({ layers: [layerId] });
      return null;
    }
    const edited = this.deps.commitStructural(
      'Edit text',
      { id: layerId, source: finalSource, type: 'updateCanvasLayerSource' },
      { id: layerId, source: startSource, type: 'updateCanvasLayerSource' }
    );
    this.settle(edited);
    return edited;
  }

  /** A landed edit or a target that is gone ends the session; a transient refusal keeps the text for retry. */
  private settle(result: StructuralCommitResult): void {
    if (result.status === 'committed' || result.status === 'dispatch-rejected' || result.status === 'not-ready') {
      this.deps.session.set(null);
    }
  }

  commitOpen(): boolean {
    if (this.disposed || !this.deps.canEdit()) {
      return false;
    }
    const session = this.deps.session.get();
    if (!session) {
      return false;
    }
    this.commit(this.contentReader ? this.contentReader() : session.source.content);
    return true;
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.contentReader = null;
    this.deps.session.set(null);
  }
}
