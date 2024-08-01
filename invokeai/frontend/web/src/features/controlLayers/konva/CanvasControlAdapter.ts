import { CanvasEntity } from 'features/controlLayers/konva/CanvasEntity';
import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasTransformer } from 'features/controlLayers/konva/CanvasTransformer';
import { type ControlAdapterEntity, isDrawingTool } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasControlAdapter extends CanvasEntity {
  static NAME_PREFIX = 'control-adapter';
  static LAYER_NAME = `${CanvasControlAdapter.NAME_PREFIX}_layer`;
  static TRANSFORMER_NAME = `${CanvasControlAdapter.NAME_PREFIX}_transformer`;
  static GROUP_NAME = `${CanvasControlAdapter.NAME_PREFIX}_group`;
  static OBJECT_GROUP_NAME = `${CanvasControlAdapter.NAME_PREFIX}_object-group`;

  type = 'control_adapter';
  _state: ControlAdapterEntity;

  konva: {
    layer: Konva.Layer;
    group: Konva.Group;
    objectGroup: Konva.Group;
  };

  image: CanvasImage | null;
  transformer: CanvasTransformer;

  constructor(state: ControlAdapterEntity, manager: CanvasManager) {
    super(state.id, manager);
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
    };
    this.transformer = new CanvasTransformer(this);
    this.konva.group.add(this.konva.objectGroup);
    this.konva.layer.add(this.konva.group);
    this.konva.layer.add(this.konva.transformer);

    this.image = null;
    this._state = state;
  }

  async render(state: ControlAdapterEntity) {
    this._state = state;

    // Update the layer's position and listening state
    this.konva.group.setAttrs({
      x: state.position.x,
      y: state.position.y,
      scaleX: 1,
      scaleY: 1,
    });

    const imageObject = state.processedImageObject ?? state.imageObject;

    let didDraw = false;

    if (!imageObject) {
      if (this.image) {
        this.image.konva.group.visible(false);
        didDraw = true;
      }
    } else if (!this.image) {
      this.image = new CanvasImage(imageObject, this);
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
    this.konva.layer.visible(this._state.isEnabled);

    this.konva.group.opacity(this._state.opacity);
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;

    if (!this.image?.konva.image) {
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

  repr() {
    return {
      id: this.id,
      type: this.type,
      state: this._state,
    };
  }
}
