import type { TextEditSession, TextSource, TextToolOptions } from '@workbench/canvas-engine/engineStores';
import type { Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

export interface TextEditingControllerOptions {
  readonly session: {
    get(): TextEditSession | null;
    set(value: TextEditSession | null): void;
  };
  readonly options: { get(): TextToolOptions };
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly createLayerId: () => string;
  readonly commitStructural: (label: string, forward: CanvasProjectMutation, inverse: CanvasProjectMutation) => void;
  readonly invalidate: (payload: { layers?: string[]; overlay?: true }) => void;
}

const sourcesEqual = (left: TextSource, right: TextSource): boolean =>
  left.content === right.content &&
  left.fontFamily === right.fontFamily &&
  left.fontSize === right.fontSize &&
  left.fontWeight === right.fontWeight &&
  left.lineHeight === right.lineHeight &&
  left.align === right.align &&
  left.color === right.color;

/** Owns create/edit text session state and its structural commits. */
export class TextEditingController {
  private sessionId = 0;
  private contentReader: (() => string) | null = null;
  private disposed = false;

  constructor(private readonly deps: TextEditingControllerOptions) {}

  private sourceFromOptions(content: string): TextSource {
    const options = this.deps.options.get();
    return {
      align: options.align,
      color: options.color,
      content,
      fontFamily: options.fontFamily,
      fontSize: options.fontSize,
      fontWeight: options.fontWeight,
      lineHeight: options.lineHeight,
      type: 'text',
    };
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
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (
      !document ||
      !layer ||
      layer.type !== 'raster' ||
      layer.source.type !== 'text' ||
      !layer.isEnabled ||
      layer.isLocked
    ) {
      return;
    }
    this.deps.session.set({
      id: ++this.sessionId,
      layerId,
      mode: 'edit',
      source: { ...layer.source },
      startSource: { ...layer.source },
      transform: { ...layer.transform },
    });
    this.deps.invalidate({ layers: [layerId] });
  }

  updateStyle(patch: Partial<TextToolOptions>): void {
    const session = this.deps.session.get();
    if (this.disposed || !session) {
      return;
    }
    this.deps.session.set({ ...session, source: { ...session.source, ...patch } });
  }

  cancel(): void {
    const session = this.deps.session.get();
    if (!session) {
      return;
    }
    this.deps.session.set(null);
    this.deps.invalidate(session.layerId ? { layers: [session.layerId] } : { overlay: true });
  }

  commit(content: string, styleChanges?: Partial<TextToolOptions>): void {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return;
    }
    const session = this.deps.session.get();
    if (!session) {
      return;
    }
    const finalSource: TextSource = { ...session.source, ...styleChanges, content };
    if (session.mode === 'create') {
      if (content.trim() === '') {
        this.cancel();
        return;
      }
      const layerId = this.deps.createLayerId();
      const layer: CanvasLayerContract = {
        blendMode: 'normal',
        id: layerId,
        isEnabled: true,
        isLocked: false,
        name: `Text ${(this.deps.getDocument()?.layers.length ?? 0) + 1}`,
        opacity: 1,
        source: finalSource,
        transform: session.transform,
        type: 'raster',
      };
      this.deps.session.set(null);
      this.deps.commitStructural(
        'Add text',
        { index: 0, layer, type: 'addCanvasLayer' },
        { ids: [layerId], type: 'removeCanvasLayers' }
      );
      this.deps.invalidate({ overlay: true });
      return;
    }
    const { layerId, startSource } = session;
    if (!layerId || !startSource) {
      this.cancel();
      return;
    }
    this.deps.session.set(null);
    if (sourcesEqual(startSource, finalSource)) {
      this.deps.invalidate({ layers: [layerId] });
      return;
    }
    this.deps.commitStructural(
      'Edit text',
      { id: layerId, source: finalSource, type: 'updateCanvasLayerSource' },
      { id: layerId, source: startSource, type: 'updateCanvasLayerSource' }
    );
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
