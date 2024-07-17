import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { type ControlAdapterEntity, isDrawingTool } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasControlAdapter {
  static NAME_PREFIX = 'control-adapter';
  static LAYER_NAME = `${CanvasControlAdapter.NAME_PREFIX}_layer`;
  static TRANSFORMER_NAME = `${CanvasControlAdapter.NAME_PREFIX}_transformer`;
  static GROUP_NAME = `${CanvasControlAdapter.NAME_PREFIX}_group`;
  static OBJECT_GROUP_NAME = `${CanvasControlAdapter.NAME_PREFIX}_object-group`;

  private controlAdapterState: ControlAdapterEntity;

  id: string;
  manager: CanvasManager;

  konva: {
    layer: Konva.Layer;
    group: Konva.Group;
    objectGroup: Konva.Group;
    transformer: Konva.Transformer;
  };

  image: CanvasImage | null;

  constructor(controlAdapterState: ControlAdapterEntity, manager: CanvasManager) {
    const { id } = controlAdapterState;
    this.id = id;
    this.manager = manager;
    this.konva = {
      layer: new Konva.Layer({
        name: CanvasControlAdapter.LAYER_NAME,
        imageSmoothingEnabled: false,
        listening: false,
      }),
      group: new Konva.Group({
        name: CanvasControlAdapter.GROUP_NAME,
        listening: false,
      }),
      objectGroup: new Konva.Group({ name: CanvasControlAdapter.GROUP_NAME, listening: false }),
      transformer: new Konva.Transformer({
        name: CanvasControlAdapter.TRANSFORMER_NAME,
        shouldOverdrawWholeArea: true,
        draggable: true,
        dragDistance: 0,
        enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        rotateEnabled: false,
        flipEnabled: false,
      }),
    };
    this.konva.transformer.on('transformend', () => {
      this.manager.stateApi.onScaleChanged(
        {
          id: this.id,
          scale: this.konva.group.scaleX(),
          position: { x: this.konva.group.x(), y: this.konva.group.y() },
        },
        'control_adapter'
      );
    });
    this.konva.transformer.on('dragend', () => {
      this.manager.stateApi.onPosChanged(
        { id: this.id, position: { x: this.konva.group.x(), y: this.konva.group.y() } },
        'control_adapter'
      );
    });
    this.konva.group.add(this.konva.objectGroup);
    this.konva.layer.add(this.konva.group);
    this.konva.layer.add(this.konva.transformer);

    this.image = null;
    this.controlAdapterState = controlAdapterState;
  }

  async render(controlAdapterState: ControlAdapterEntity) {
    this.controlAdapterState = controlAdapterState;

    // Update the layer's position and listening state
    this.konva.group.setAttrs({
      x: controlAdapterState.position.x,
      y: controlAdapterState.position.y,
      scaleX: 1,
      scaleY: 1,
    });

    const imageObject = controlAdapterState.processedImageObject ?? controlAdapterState.imageObject;

    let didDraw = false;

    if (!imageObject) {
      if (this.image) {
        this.image.konva.group.visible(false);
        didDraw = true;
      }
    } else if (!this.image) {
      this.image = new CanvasImage(imageObject);
      this.updateGroup(true);
      this.konva.objectGroup.add(this.image.konva.group);
      await this.image.updateImageSource(imageObject.image.name);
    } else if (!this.image.isLoading && !this.image.isError) {
      if (await this.image.update(imageObject)) {
        didDraw = true;
      }
    }

    this.updateGroup(didDraw);
  }

  updateGroup(didDraw: boolean) {
    this.konva.layer.visible(this.controlAdapterState.isEnabled);

    this.konva.group.opacity(this.controlAdapterState.opacity);
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;

    if (!this.image?.image) {
      // If the layer is totally empty, reset the cache and bail out.
      this.konva.layer.listening(false);
      this.konva.transformer.nodes([]);
      if (this.konva.group.isCached()) {
        this.konva.group.clearCache();
      }
      return;
    }

    if (isSelected && selectedTool === 'move') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.konva.group.isCached() || didDraw) {
        this.konva.group.cache();
      }
      // Activate the transformer
      this.konva.layer.listening(true);
      this.konva.transformer.nodes([this.konva.group]);
      this.konva.transformer.forceUpdate();
      return;
    }

    if (isSelected && selectedTool !== 'move') {
      // If the layer is selected but not using the move tool, we don't want the layer to be listening.
      this.konva.layer.listening(false);
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      if (isDrawingTool(selectedTool)) {
        // We are using a drawing tool (brush, eraser, rect). These tools change the layer's rendered appearance, so we
        // should never be cached.
        if (this.konva.group.isCached()) {
          this.konva.group.clearCache();
        }
      } else {
        // We are using a non-drawing tool (move, view, bbox), so we should cache the layer.
        // We should update the cache if we drew to the layer.
        if (!this.konva.group.isCached() || didDraw) {
          this.konva.group.cache();
        }
      }
      return;
    }

    if (!isSelected) {
      // Unselected layers should not be listening
      this.konva.layer.listening(false);
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      // Update the layer's cache if it's not already cached or we drew to it.
      if (!this.konva.group.isCached() || didDraw) {
        this.konva.group.cache();
      }

      return;
    }
  }

  destroy(): void {
    this.konva.layer.destroy();
  }
}
