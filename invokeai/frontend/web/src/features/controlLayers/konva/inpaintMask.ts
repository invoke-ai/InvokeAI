import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { getNodeBboxFast } from 'features/controlLayers/konva/entityBbox';
import { getObjectGroupId,INPAINT_MASK_LAYER_ID } from 'features/controlLayers/konva/naming';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { KonvaBrushLine, KonvaEraserLine, KonvaRect } from 'features/controlLayers/konva/objects';
import { mapId } from 'features/controlLayers/konva/util';
import { type InpaintMaskEntity, isDrawingTool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export class CanvasInpaintMask {
  id: string;
  manager: KonvaNodeManager;
  layer: Konva.Layer;
  group: Konva.Group;
  objectsGroup: Konva.Group;
  compositingRect: Konva.Rect;
  transformer: Konva.Transformer;
  objects: Map<string, KonvaBrushLine | KonvaEraserLine | KonvaRect>;

  constructor(manager: KonvaNodeManager) {
    this.id = INPAINT_MASK_LAYER_ID;
    this.manager = manager;
    this.layer = new Konva.Layer({ id: INPAINT_MASK_LAYER_ID });

    this.group = new Konva.Group({
      id: getObjectGroupId(this.layer.id(), uuidv4()),
      listening: false,
    });
    this.objectsGroup = new Konva.Group({});
    this.group.add(this.objectsGroup);
    this.layer.add(this.group);

    this.transformer = new Konva.Transformer({
      shouldOverdrawWholeArea: true,
      draggable: true,
      dragDistance: 0,
      enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
      rotateEnabled: false,
      flipEnabled: false,
    });
    this.transformer.on('transformend', () => {
      this.manager.stateApi.onScaleChanged(
        { id: this.id, scale: this.group.scaleX(), x: this.group.x(), y: this.group.y() },
        'inpaint_mask'
      );
    });
    this.transformer.on('dragend', () => {
      this.manager.stateApi.onPosChanged({ id: this.id, x: this.group.x(), y: this.group.y() }, 'inpaint_mask');
    });
    this.layer.add(this.transformer);

    this.compositingRect = new Konva.Rect({ listening: false });
    this.group.add(this.compositingRect);
    this.objects = new Map();
  }

  destroy(): void {
    this.layer.destroy();
  }

  async render(inpaintMaskState: InpaintMaskEntity) {
    // Update the layer's position and listening state
    this.group.setAttrs({
      x: inpaintMaskState.x,
      y: inpaintMaskState.y,
      scaleX: 1,
      scaleY: 1,
    });

    let didDraw = false;

    const objectIds = inpaintMaskState.objects.map(mapId);
    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id)) {
        this.objects.delete(object.id);
        object.destroy();
        didDraw = true;
      }
    }

    for (const obj of inpaintMaskState.objects) {
      if (obj.type === 'brush_line') {
        let brushLine = this.objects.get(obj.id);
        assert(brushLine instanceof KonvaBrushLine || brushLine === undefined);

        if (!brushLine) {
          brushLine = new KonvaBrushLine(obj);
          this.objects.set(brushLine.id, brushLine);
          this.objectsGroup.add(brushLine.konvaLineGroup);
          didDraw = true;
        } else {
          if (brushLine.update(obj)) {
            didDraw = true;
          }
        }
      } else if (obj.type === 'eraser_line') {
        let eraserLine = this.objects.get(obj.id);
        assert(eraserLine instanceof KonvaEraserLine || eraserLine === undefined);

        if (!eraserLine) {
          eraserLine = new KonvaEraserLine(obj);
          this.objects.set(eraserLine.id, eraserLine);
          this.objectsGroup.add(eraserLine.konvaLineGroup);
          didDraw = true;
        } else {
          if (eraserLine.update(obj)) {
            didDraw = true;
          }
        }
      } else if (obj.type === 'rect_shape') {
        let rect = this.objects.get(obj.id);
        assert(rect instanceof KonvaRect || rect === undefined);

        if (!rect) {
          rect = new KonvaRect(obj);
          this.objects.set(rect.id, rect);
          this.objectsGroup.add(rect.konvaRect);
          didDraw = true;
        } else {
          if (rect.update(obj)) {
            didDraw = true;
          }
        }
      }
    }

    // Only update layer visibility if it has changed.
    if (this.layer.visible() !== inpaintMaskState.isEnabled) {
      this.layer.visible(inpaintMaskState.isEnabled);
    }

    // The user is allowed to reduce mask opacity to 0, but we need the opacity for the compositing rect to work
    this.group.opacity(1);

    if (didDraw) {
      // Convert the color to a string, stripping the alpha - the object group will handle opacity.
      const rgbColor = rgbColorToString(inpaintMaskState.fill);
      const maskOpacity = this.manager.stateApi.getMaskOpacity();

      this.compositingRect.setAttrs({
        // The rect should be the size of the layer - use the fast method if we don't have a pixel-perfect bbox already
        ...getNodeBboxFast(this.objectsGroup),
        fill: rgbColor,
        opacity: maskOpacity,
        // Draw this rect only where there are non-transparent pixels under it (e.g. the mask shapes)
        globalCompositeOperation: 'source-in',
        visible: true,
        // This rect must always be on top of all other shapes
        zIndex: this.objects.size + 1,
      });
    }

    this.updateGroup(didDraw);
  }

  updateGroup(didDraw: boolean) {
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;

    if (this.objects.size === 0) {
      // If the layer is totally empty, reset the cache and bail out.
      this.layer.listening(false);
      this.transformer.nodes([]);
      if (this.group.isCached()) {
        this.group.clearCache();
      }
      return;
    }

    if (isSelected && selectedTool === 'move') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.group.isCached() || didDraw) {
        this.group.cache();
      }
      // Activate the transformer
      this.layer.listening(true);
      this.transformer.nodes([this.group]);
      this.transformer.forceUpdate();
      return;
    }

    if (isSelected && selectedTool !== 'move') {
      // If the layer is selected but not using the move tool, we don't want the layer to be listening.
      this.layer.listening(false);
      // The transformer also does not need to be active.
      this.transformer.nodes([]);
      if (isDrawingTool(selectedTool)) {
        // We are using a drawing tool (brush, eraser, rect). These tools change the layer's rendered appearance, so we
        // should never be cached.
        if (this.group.isCached()) {
          this.group.clearCache();
        }
      } else {
        // We are using a non-drawing tool (move, view, bbox), so we should cache the layer.
        // We should update the cache if we drew to the layer.
        if (!this.group.isCached() || didDraw) {
          this.group.cache();
        }
      }
      return;
    }

    if (!isSelected) {
      // Unselected layers should not be listening
      this.layer.listening(false);
      // The transformer also does not need to be active.
      this.transformer.nodes([]);
      // Update the layer's cache if it's not already cached or we drew to it.
      if (!this.group.isCached() || didDraw) {
        this.group.cache();
      }

      return;
    }
  }
}
