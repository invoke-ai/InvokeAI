import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import type { StateApi } from 'features/controlLayers/konva/nodeManager';
import { getLayerBboxFast } from 'features/controlLayers/konva/renderers/entityBbox';
import { KonvaBrushLine, KonvaEraserLine, KonvaRect } from 'features/controlLayers/konva/renderers/objects';
import { mapId } from 'features/controlLayers/konva/util';
import type { CanvasEntityIdentifier, InpaintMaskEntity, Tool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export class CanvasInpaintMask {
  id: string;
  layer: Konva.Layer;
  group: Konva.Group;
  compositingRect: Konva.Rect;
  objects: Map<string, KonvaBrushLine | KonvaEraserLine | KonvaRect>;

  constructor(entity: InpaintMaskEntity, onPosChanged: StateApi['onPosChanged']) {
    this.id = entity.id;

    this.layer = new Konva.Layer({
      id: entity.id,
      draggable: true,
      dragDistance: 0,
    });

    // When a drag on the layer finishes, update the layer's position in state. During the drag, konva handles changing
    // the position - we do not need to call this on the `dragmove` event.
    this.layer.on('dragend', function (e) {
      onPosChanged({ id: entity.id, x: Math.floor(e.target.x()), y: Math.floor(e.target.y()) }, 'inpaint_mask');
    });
    this.group = new Konva.Group({
      id: getObjectGroupId(this.layer.id(), uuidv4()),
      listening: false,
    });
    this.layer.add(this.group);
    this.compositingRect = new Konva.Rect({ listening: false });
    this.layer.add(this.compositingRect);
    this.objects = new Map();
  }

  destroy(): void {
    this.layer.destroy();
  }

  async render(
    inpaintMaskState: InpaintMaskEntity,
    selectedTool: Tool,
    selectedEntityIdentifier: CanvasEntityIdentifier | null,
    maskOpacity: number
  ) {
    // Update the layer's position and listening state
    this.layer.setAttrs({
      listening: selectedTool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
      x: Math.floor(inpaintMaskState.x),
      y: Math.floor(inpaintMaskState.y),
    });

    // Convert the color to a string, stripping the alpha - the object group will handle opacity.
    const rgbColor = rgbColorToString(inpaintMaskState.fill);

    // We use caching to handle "global" layer opacity, but caching is expensive and we should only do it when required.
    let groupNeedsCache = false;

    const objectIds = inpaintMaskState.objects.map(mapId);
    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id)) {
        this.objects.delete(object.id);
        object.destroy();
        groupNeedsCache = true;
      }
    }

    for (const obj of inpaintMaskState.objects) {
      if (obj.type === 'brush_line') {
        let brushLine = this.objects.get(obj.id);
        assert(brushLine instanceof KonvaBrushLine || brushLine === undefined);

        if (!brushLine) {
          brushLine = new KonvaBrushLine({ brushLine: obj });
          this.objects.set(brushLine.id, brushLine);
          this.group.add(brushLine.konvaLineGroup);
          groupNeedsCache = true;
        }

        if (obj.points.length !== brushLine.konvaLine.points().length) {
          brushLine.konvaLine.points(obj.points);
          groupNeedsCache = true;
        }
      } else if (obj.type === 'eraser_line') {
        let eraserLine = this.objects.get(obj.id);
        assert(eraserLine instanceof KonvaEraserLine || eraserLine === undefined);

        if (!eraserLine) {
          eraserLine = new KonvaEraserLine({ eraserLine: obj });
          this.objects.set(eraserLine.id, eraserLine);
          this.group.add(eraserLine.konvaLineGroup);
          groupNeedsCache = true;
        }

        if (obj.points.length !== eraserLine.konvaLine.points().length) {
          eraserLine.konvaLine.points(obj.points);
          groupNeedsCache = true;
        }
      } else if (obj.type === 'rect_shape') {
        let rect = this.objects.get(obj.id);
        assert(rect instanceof KonvaRect || rect === undefined);

        if (!rect) {
          rect = new KonvaRect({ rectShape: obj });
          this.objects.set(rect.id, rect);
          this.group.add(rect.konvaRect);
          groupNeedsCache = true;
        }
      }
    }

    // Only update layer visibility if it has changed.
    if (this.layer.visible() !== inpaintMaskState.isEnabled) {
      this.layer.visible(inpaintMaskState.isEnabled);
      groupNeedsCache = true;
    }

    if (this.objects.size === 0) {
      // No objects - clear the cache to reset the previous pixel data
      this.group.clearCache();
      return;
    }


    // We must clear the cache first so Konva will re-draw the group with the new compositing rect
    if (this.group.isCached()) {
      this.group.clearCache();
    }
    // The user is allowed to reduce mask opacity to 0, but we need the opacity for the compositing rect to work
    this.group.opacity(1);

    this.compositingRect.setAttrs({
      // The rect should be the size of the layer - use the fast method if we don't have a pixel-perfect bbox already
      ...(!inpaintMaskState.bboxNeedsUpdate && inpaintMaskState.bbox
        ? inpaintMaskState.bbox
        : getLayerBboxFast(this.layer)),
      fill: rgbColor,
      opacity: maskOpacity,
      // Draw this rect only where there are non-transparent pixels under it (e.g. the mask shapes)
      globalCompositeOperation: 'source-in',
      visible: true,
      // This rect must always be on top of all other shapes
      zIndex: this.objects.size + 1,
    });

    // const isSelected = selectedEntityIdentifier?.id === inpaintMaskState.id;

    // /**
    //  * When the group is selected, we use a rect of the selected preview color, composited over the shapes. This allows
    //  * shapes to render as a "raster" layer with all pixels drawn at the same color and opacity.
    //  *
    //  * Without this special handling, each shape is drawn individually with the given opacity, atop the other shapes. The
    //  * effect is like if you have a Photoshop Group consisting of many shapes, each of which has the given opacity.
    //  * Overlapping shapes will have their colors blended together, and the final color is the result of all the shapes.
    //  *
    //  * Instead, with the special handling, the effect is as if you drew all the shapes at 100% opacity, flattened them to
    //  * a single raster image, and _then_ applied the 50% opacity.
    //  */
    // if (isSelected && selectedTool !== 'move') {
    //   // We must clear the cache first so Konva will re-draw the group with the new compositing rect
    //   if (this.konvaObjectGroup.isCached()) {
    //     this.konvaObjectGroup.clearCache();
    //   }
    //   // The user is allowed to reduce mask opacity to 0, but we need the opacity for the compositing rect to work
    //   this.konvaObjectGroup.opacity(1);

    //   this.compositingRect.setAttrs({
    //     // The rect should be the size of the layer - use the fast method if we don't have a pixel-perfect bbox already
    //     ...(!inpaintMaskState.bboxNeedsUpdate && inpaintMaskState.bbox
    //       ? inpaintMaskState.bbox
    //       : getLayerBboxFast(this.konvaLayer)),
    //     fill: rgbColor,
    //     opacity: maskOpacity,
    //     // Draw this rect only where there are non-transparent pixels under it (e.g. the mask shapes)
    //     globalCompositeOperation: 'source-in',
    //     visible: true,
    //     // This rect must always be on top of all other shapes
    //     zIndex: this.objects.size + 1,
    //   });
    // } else {
    //   // The compositing rect should only be shown when the layer is selected.
    //   this.compositingRect.visible(false);
    //   // Cache only if needed - or if we are on this code path and _don't_ have a cache
    //   if (groupNeedsCache || !this.konvaObjectGroup.isCached()) {
    //     this.konvaObjectGroup.cache();
    //   }
    //   // Updating group opacity does not require re-caching
    //   this.konvaObjectGroup.opacity(maskOpacity);
    // }

    // const bboxRect =
    //   regionMap.konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(rg, regionMap.konvaLayer);
    // if (rg.bbox) {
    //   const active = !rg.bboxNeedsUpdate && isSelected && tool === 'move';
    //   bboxRect.setAttrs({
    //     visible: active,
    //     listening: active,
    //     x: rg.bbox.x,
    //     y: rg.bbox.y,
    //     width: rg.bbox.width,
    //     height: rg.bbox.height,
    //     stroke: isSelected ? BBOX_SELECTED_STROKE : '',
    //   });
    // } else {
    //   bboxRect.visible(false);
    // }
  }
}
