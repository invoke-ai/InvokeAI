import { deepClone } from 'common/util/deepClone';
import { CanvasBrushLine } from 'features/controlLayers/konva/CanvasBrushLine';
import { CanvasEraserLine } from 'features/controlLayers/konva/CanvasEraserLine';
import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasRect } from 'features/controlLayers/konva/CanvasRect';
import { mapId } from 'features/controlLayers/konva/util';
import type { BrushLine, EraserLine, LayerEntity, Rect, RectShape } from 'features/controlLayers/store/types';
import { isDrawingTool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { debounce } from 'lodash-es';
import { assert } from 'tsafe';

const MIN_LAYER_SIZE_PX = 10;

export class CanvasLayer {
  static NAME_PREFIX = 'layer';
  static LAYER_NAME = `${CanvasLayer.NAME_PREFIX}_layer`;
  static TRANSFORMER_NAME = `${CanvasLayer.NAME_PREFIX}_transformer`;
  static INTERACTION_RECT_NAME = `${CanvasLayer.NAME_PREFIX}_interaction-rect`;
  static GROUP_NAME = `${CanvasLayer.NAME_PREFIX}_group`;
  static OBJECT_GROUP_NAME = `${CanvasLayer.NAME_PREFIX}_object-group`;
  static BBOX_NAME = `${CanvasLayer.NAME_PREFIX}_bbox`;

  private static BBOX_PADDING_PX = 5;

  private drawingBuffer: BrushLine | EraserLine | RectShape | null;
  private state: LayerEntity;

  id: string;
  manager: CanvasManager;

  konva: {
    layer: Konva.Layer;
    bbox: Konva.Rect;
    objectGroup: Konva.Group;
    objectGroupBbox: Konva.Rect;
    positionXLine: Konva.Line;
    positionYLine: Konva.Line;
    transformer: Konva.Transformer;
    interactionRect: Konva.Rect;
  };
  objects: Map<string, CanvasBrushLine | CanvasEraserLine | CanvasRect | CanvasImage>;
  bbox: Rect;

  getBbox = debounce(this._getBbox, 300);

  constructor(state: LayerEntity, manager: CanvasManager) {
    this.id = state.id;
    this.manager = manager;
    this.konva = {
      layer: new Konva.Layer({ id: this.id, name: CanvasLayer.LAYER_NAME, listening: false }),
      bbox: new Konva.Rect({
        listening: false,
        draggable: false,
        name: CanvasLayer.BBOX_NAME,
        stroke: 'hsl(200deg 76% 59%)', // invokeBlue.400
        perfectDrawEnabled: false,
        strokeHitEnabled: false,
      }),
      objectGroup: new Konva.Group({ name: CanvasLayer.OBJECT_GROUP_NAME, listening: false }),
      objectGroupBbox: new Konva.Rect({ fill: 'green', opacity: 0.5, listening: false }),
      positionXLine: new Konva.Line({ stroke: 'white', strokeWidth: 1 }),
      positionYLine: new Konva.Line({ stroke: 'white', strokeWidth: 1 }),
      transformer: new Konva.Transformer({
        name: CanvasLayer.TRANSFORMER_NAME,
        draggable: false,
        // enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        rotateEnabled: false,
        flipEnabled: false,
        listening: false,
        padding: CanvasLayer.BBOX_PADDING_PX,
        stroke: 'hsl(200deg 76% 59%)', // invokeBlue.400
      }),
      interactionRect: new Konva.Rect({
        name: CanvasLayer.INTERACTION_RECT_NAME,
        listening: false,
        draggable: true,
        fill: 'rgba(255,0,0,0.5)',
      }),
    };

    this.konva.layer.add(this.konva.objectGroup);
    this.konva.layer.add(this.konva.transformer);
    this.konva.layer.add(this.konva.interactionRect);
    this.konva.layer.add(this.konva.bbox);
    this.konva.layer.add(this.konva.objectGroupBbox);
    this.konva.layer.add(this.konva.positionXLine);
    this.konva.layer.add(this.konva.positionYLine);

    this.konva.transformer.on('transformstart', () => {
      console.log('>>> transformstart');
      console.log('interactionRect', {
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
        scaleX: this.konva.interactionRect.scaleX(),
        scaleY: this.konva.interactionRect.scaleY(),
        width: this.konva.interactionRect.width(),
        height: this.konva.interactionRect.height(),
      });
      console.log('this.bbox', deepClone(this.bbox));
      console.log('this.state.position', this.state.position);
    });

    this.konva.transformer.on('transform', () => {
      // Always snap the interaction rect to the nearest pixel when transforming

      const x = Math.round(this.konva.interactionRect.x());
      const y = Math.round(this.konva.interactionRect.y());
      // Snap its position
      this.konva.interactionRect.x(x);
      this.konva.interactionRect.y(y);

      // Calculate the new scale of the interaction rect such that its width and height snap to the nearest pixel
      const targetWidth = Math.max(
        Math.round(this.konva.interactionRect.width() * Math.abs(this.konva.interactionRect.scaleX())),
        MIN_LAYER_SIZE_PX
      );
      const scaleX = targetWidth / this.konva.interactionRect.width();
      const targetHeight = Math.max(
        Math.round(this.konva.interactionRect.height() * Math.abs(this.konva.interactionRect.scaleY())),
        MIN_LAYER_SIZE_PX
      );
      const scaleY = targetHeight / this.konva.interactionRect.height();

      // Snap the width and height (via scale) of the interaction rect
      this.konva.interactionRect.scaleX(scaleX);
      this.konva.interactionRect.scaleY(scaleY);
      this.konva.interactionRect.rotation(0);

      console.log('>>> transform');
      console.log('activeAnchor', this.konva.transformer.getActiveAnchor());
      console.log('interactionRect', {
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
        scaleX: this.konva.interactionRect.scaleX(),
        scaleY: this.konva.interactionRect.scaleY(),
        width: this.konva.interactionRect.width(),
        height: this.konva.interactionRect.height(),
        rotation: this.konva.interactionRect.rotation(),
      });

      // Handle anchor-specific transformations of the layer's objects
      const anchor = this.konva.transformer.getActiveAnchor();
      // 'top-left'
      // 'top-center'
      // 'top-right'
      // 'middle-right'
      // 'middle-left'
      // 'bottom-left'
      // 'bottom-center'
      // 'bottom-right'
      if (anchor === 'middle-right') {
        // Dragging the anchor to the right
        this.konva.objectGroup.setAttrs({
          scaleX,
          x: x - (x - this.state.position.x) * scaleX,
        });
      } else if (anchor === 'middle-left') {
        // Dragging the anchor to the right
        this.konva.objectGroup.setAttrs({
          scaleX,
          x: x - (x - this.state.position.x) * scaleX,
        });
      } else if (anchor === 'bottom-center') {
        // Resize the interaction rect downwards
        this.konva.objectGroup.setAttrs({
          scaleY,
          y: y - (y - this.state.position.y) * scaleY,
        });
      } else if (anchor === 'bottom-right') {
        // Resize the interaction rect to the right and downwards via scale
        this.konva.objectGroup.setAttrs({
          scaleX,
          scaleY,
          x: x - (x - this.state.position.x) * scaleX,
          y: y - (y - this.state.position.y) * scaleY,
        });
      } else if (anchor === 'top-center') {
        // Resize the interaction rect to the upwards via scale & y position
        this.konva.objectGroup.setAttrs({
          y,
          scaleY,
        });
      }
      this.konva.objectGroupBbox.setAttrs({
        x: this.konva.objectGroup.x(),
        y: this.konva.objectGroup.y(),
        rotation: this.konva.objectGroup.rotation(),
        scaleX: this.konva.objectGroup.scaleX(),
        scaleY: this.konva.objectGroup.scaleY(),
      });
    });

    // this.konva.transformer.on('transform', () => {
    //   // We need to snap the transform to the nearest pixel - both the position and the scale

    //   // Snap the interaction rect to the nearest pixel
    //   this.konva.interactionRect.x(Math.round(this.konva.interactionRect.x()));
    //   this.konva.interactionRect.y(Math.round(this.konva.interactionRect.y()));

    //   // Calculate the new scale of the interaction rect such that its width and height snap to the nearest pixel
    //   const roundedScaledWidth = Math.round(this.konva.interactionRect.width() * this.konva.interactionRect.scaleX());
    //   const correctedScaleX = roundedScaledWidth / this.konva.interactionRect.width();
    //   const roundedScaledHeight = Math.round(this.konva.interactionRect.height() * this.konva.interactionRect.scaleY());
    //   const correctedScaleY = roundedScaledHeight / this.konva.interactionRect.height();

    //   // Update the interaction rect's scale to the corrected scale
    //   this.konva.interactionRect.scaleX(correctedScaleX);
    //   this.konva.interactionRect.scaleY(correctedScaleY);

    //   console.log('>>> transform');
    //   console.log('activeAnchor', this.konva.transformer.getActiveAnchor());
    //   console.log('interactionRect', {
    //     x: this.konva.interactionRect.x(),
    //     y: this.konva.interactionRect.y(),
    //     scaleX: this.konva.interactionRect.scaleX(),
    //     scaleY: this.konva.interactionRect.scaleY(),
    //     width: this.konva.interactionRect.width(),
    //     height: this.konva.interactionRect.height(),
    //     rotation: this.konva.interactionRect.rotation(),
    //   });

    //   // Update the object group to reflect the new scale and position of the interaction rect
    //   this.konva.objectGroup.setAttrs({
    //     // The scale is the same as the interaction rect
    //     scaleX: this.konva.interactionRect.scaleX(),
    //     scaleY: this.konva.interactionRect.scaleY(),
    //     rotation: this.konva.interactionRect.rotation(),
    //     // We need to do some compensation for the new position. The bounds of the object group may be different from the
    //     // interaction rect/bbox, because the object group may have eraser strokes that are not included in the bbox.
    //     x:
    //       this.konva.interactionRect.x() -
    //       Math.abs(this.konva.interactionRect.x() - this.state.position.x) * this.konva.interactionRect.scaleX(),
    //     y:
    //       this.konva.interactionRect.y() -
    //       Math.abs(this.konva.interactionRect.y() - this.state.position.y) * this.konva.interactionRect.scaleY(),
    //     // x: this.konva.interactionRect.x() + (this.konva.interactionRect.x() - this.state.position.x) * this.konva.interactionRect.scaleX(),
    //     // y: this.konva.interactionRect.y() + (this.konva.interactionRect.y() - this.state.position.y) * this.konva.interactionRect.scaleY(),
    //   });
    //   this.konva.objectGroupBbox.setAttrs({
    //     x: this.konva.objectGroup.x(),
    //     y: this.konva.objectGroup.y(),
    //     scaleX: this.konva.objectGroup.scaleX(),
    //     scaleY: this.konva.objectGroup.scaleY(),
    //   });
    // });
    this.konva.transformer.on('transformend', () => {
      this.bbox = {
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
        width: Math.round(this.konva.interactionRect.width() * this.konva.interactionRect.scaleX()),
        height: Math.round(this.konva.interactionRect.height() * this.konva.interactionRect.scaleY()),
      };

      // this.manager.stateApi.onPosChanged(
      //   {
      //     id: this.id,
      //     position: { x: this.konva.objectGroup.x(), y: this.konva.objectGroup.y() },
      //   },
      //   'layer'
      // );
    });

    this.konva.interactionRect.on('dragmove', () => {
      // Snap the interaction rect to the nearest pixel
      this.konva.interactionRect.x(Math.round(this.konva.interactionRect.x()));
      this.konva.interactionRect.y(Math.round(this.konva.interactionRect.y()));

      // The bbox should be updated to reflect the new position of the interaction rect, taking into account its padding
      // and border
      this.konva.bbox.setAttrs({
        x: this.konva.interactionRect.x() - CanvasLayer.BBOX_PADDING_PX / this.manager.stage.scaleX(),
        y: this.konva.interactionRect.y() - CanvasLayer.BBOX_PADDING_PX / this.manager.stage.scaleX(),
      });

      // The object group is translated by the difference between the interaction rect's new and old positions (which is
      // stored as this.bbox)
      this.konva.objectGroup.setAttrs({
        x: this.state.position.x + this.konva.interactionRect.x() - this.bbox.x,
        y: this.state.position.y + this.konva.interactionRect.y() - this.bbox.y,
      });

      const rect = this.konva.objectGroup.getClientRect({ skipTransform: true });
      this.konva.objectGroupBbox.setAttrs({ ...rect, x: this.konva.objectGroup.x(), y: this.konva.objectGroup.y() });
    });
    this.konva.interactionRect.on('dragend', () => {
      // Update the bbox
      this.bbox.x = this.konva.interactionRect.x();
      this.bbox.y = this.konva.interactionRect.y();

      // Update internal state
      this.manager.stateApi.onPosChanged(
        {
          id: this.id,
          position: { x: this.konva.objectGroup.x(), y: this.konva.objectGroup.y() },
        },
        'layer'
      );
    });

    this.objects = new Map();
    this.drawingBuffer = null;
    this.state = state;
    this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;

    console.log(this);
  }

  private static get DEFAULT_BBOX_RECT() {
    return { x: 0, y: 0, width: 0, height: 0 };
  }

  destroy(): void {
    this.konva.layer.destroy();
  }

  getDrawingBuffer() {
    return this.drawingBuffer;
  }

  async setDrawingBuffer(obj: BrushLine | EraserLine | RectShape | null) {
    if (obj) {
      this.drawingBuffer = obj;
      await this.renderObject(this.drawingBuffer, true);
      this.updateGroup(true);
    } else {
      this.drawingBuffer = null;
    }
  }

  finalizeDrawingBuffer() {
    if (!this.drawingBuffer) {
      return;
    }
    if (this.drawingBuffer.type === 'brush_line') {
      this.manager.stateApi.onBrushLineAdded({ id: this.id, brushLine: this.drawingBuffer }, 'layer');
    } else if (this.drawingBuffer.type === 'eraser_line') {
      this.manager.stateApi.onEraserLineAdded({ id: this.id, eraserLine: this.drawingBuffer }, 'layer');
    } else if (this.drawingBuffer.type === 'rect_shape') {
      this.manager.stateApi.onRectShapeAdded({ id: this.id, rectShape: this.drawingBuffer }, 'layer');
    }
    this.setDrawingBuffer(null);
  }

  async render(state: LayerEntity) {
    this.state = state;

    // Update the layer's position and listening state
    this.konva.objectGroup.setAttrs({
      x: state.position.x,
      y: state.position.y,
      scaleX: 1,
      scaleY: 1,
    });
    this.konva.positionXLine.points([
      state.position.x,
      -this.manager.stage.y(),
      state.position.x,
      this.manager.stage.y() + this.manager.stage.height() / this.manager.stage.scaleY(),
    ]);
    this.konva.positionYLine.points([
      -this.manager.stage.x(),
      state.position.y,
      this.manager.stage.x() + this.manager.stage.width() / this.manager.stage.scaleX(),
      state.position.y,
    ]);

    let didDraw = false;

    const objectIds = state.objects.map(mapId);
    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id) && object.id !== this.drawingBuffer?.id) {
        this.objects.delete(object.id);
        object.destroy();
        didDraw = true;
      }
    }

    for (const obj of state.objects) {
      if (await this.renderObject(obj)) {
        didDraw = true;
      }
    }

    if (this.drawingBuffer) {
      if (await this.renderObject(this.drawingBuffer)) {
        didDraw = true;
      }
    }

    this.renderBbox();
    this.updateGroup(didDraw);
  }

  private async renderObject(obj: LayerEntity['objects'][number], force = false): Promise<boolean> {
    if (obj.type === 'brush_line') {
      let brushLine = this.objects.get(obj.id);
      assert(brushLine instanceof CanvasBrushLine || brushLine === undefined);

      if (!brushLine) {
        brushLine = new CanvasBrushLine(obj);
        this.objects.set(brushLine.id, brushLine);
        this.konva.objectGroup.add(brushLine.konva.group);
        return true;
      } else {
        if (brushLine.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'eraser_line') {
      let eraserLine = this.objects.get(obj.id);
      assert(eraserLine instanceof CanvasEraserLine || eraserLine === undefined);

      if (!eraserLine) {
        eraserLine = new CanvasEraserLine(obj);
        this.objects.set(eraserLine.id, eraserLine);
        this.konva.objectGroup.add(eraserLine.konva.group);
        return true;
      } else {
        if (eraserLine.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'rect_shape') {
      let rect = this.objects.get(obj.id);
      assert(rect instanceof CanvasRect || rect === undefined);

      if (!rect) {
        rect = new CanvasRect(obj);
        this.objects.set(rect.id, rect);
        this.konva.objectGroup.add(rect.konva.group);
        return true;
      } else {
        if (rect.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'image') {
      let image = this.objects.get(obj.id);
      assert(image instanceof CanvasImage || image === undefined);

      if (!image) {
        image = new CanvasImage(obj);
        this.objects.set(image.id, image);
        this.konva.objectGroup.add(image.konva.group);
        await image.updateImageSource(obj.image.name);
        return true;
      } else {
        if (await image.update(obj, force)) {
          return true;
        }
      }
    }

    return false;
  }

  updateGroup(didDraw: boolean) {
    if (!this.state.isEnabled) {
      this.konva.layer.visible(false);
      return;
    }

    if (didDraw) {
      if (this.objects.size > 0) {
        // this.getBbox();
      } else {
        this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
        this.renderBbox();
      }
    }

    this.konva.layer.visible(true);
    this.konva.objectGroup.opacity(this.state.opacity);
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;

    const isTransforming = selectedTool === 'transform' && isSelected;
    const isMoving = selectedTool === 'move' && isSelected;

    this.konva.layer.listening(isTransforming || isMoving);
    this.konva.transformer.listening(isTransforming);
    this.konva.bbox.visible(isMoving);
    this.konva.interactionRect.listening(isMoving);

    if (this.objects.size === 0) {
      // If the layer is totally empty, reset the cache and bail out.
      this.konva.transformer.nodes([]);
      if (this.konva.objectGroup.isCached()) {
        this.konva.objectGroup.clearCache();
      }
    } else if (isSelected && selectedTool === 'transform') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.konva.objectGroup.isCached() || didDraw) {
        // this.konva.objectGroup.cache();
      }
      // Activate the transformer - it *must* be transforming the interactionRect, not the group!
      this.konva.transformer.nodes([this.konva.interactionRect]);
      this.konva.transformer.forceUpdate();
      this.konva.transformer.visible(true);
    } else if (selectedTool === 'move') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.konva.objectGroup.isCached() || didDraw) {
        // this.konva.objectGroup.cache();
      }
      // Activate the transformer
      this.konva.transformer.nodes([]);
      this.konva.transformer.forceUpdate();
      this.konva.transformer.visible(false);
    } else if (isSelected) {
      // If the layer is selected but not using the move tool, we don't want the layer to be listening.
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      if (isDrawingTool(selectedTool)) {
        // We are using a drawing tool (brush, eraser, rect). These tools change the layer's rendered appearance, so we
        // should never be cached.
        if (this.konva.objectGroup.isCached()) {
          this.konva.objectGroup.clearCache();
        }
      } else {
        // We are using a non-drawing tool (move, view, bbox), so we should cache the layer.
        // We should update the cache if we drew to the layer.
        if (!this.konva.objectGroup.isCached() || didDraw) {
          // this.konva.objectGroup.cache();
        }
      }
    } else if (!isSelected) {
      // Unselected layers should not be listening
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      // Update the layer's cache if it's not already cached or we drew to it.
      if (!this.konva.objectGroup.isCached() || didDraw) {
        // this.konva.objectGroup.cache();
      }
    }
  }

  renderBbox() {
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;
    const scale = this.manager.stage.scaleX();
    const hasBbox = this.bbox.width !== 0 && this.bbox.height !== 0;

    this.konva.bbox.visible(hasBbox && isSelected && selectedTool === 'move');
    this.konva.interactionRect.visible(hasBbox);
    const rect = this.konva.objectGroup.getClientRect({ skipTransform: true });
    this.konva.objectGroupBbox.setAttrs({
      ...rect,
      x: this.konva.objectGroup.x(),
      y: this.konva.objectGroup.y(),
      scaleX: 1,
      scaleY: 1,
    });
    this.konva.bbox.setAttrs({
      x: this.bbox.x - CanvasLayer.BBOX_PADDING_PX / scale,
      y: this.bbox.y - CanvasLayer.BBOX_PADDING_PX / scale,
      width: this.bbox.width + (CanvasLayer.BBOX_PADDING_PX / scale) * 2,
      height: this.bbox.height + (CanvasLayer.BBOX_PADDING_PX / scale) * 2,
      scaleX: 1,
      scaleY: 1,
      strokeWidth: 1 / this.manager.stage.scaleX(),
    });
    this.konva.interactionRect.setAttrs({
      x: this.bbox.x,
      y: this.bbox.y,
      width: this.bbox.width,
      height: this.bbox.height,
      scaleX: 1,
      scaleY: 1,
    });
  }

  private _getBbox() {
    if (this.objects.size === 0) {
      this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
      this.renderBbox();
      return;
    }

    let needsPixelBbox = false;
    const rect = this.konva.objectGroup.getClientRect({ skipTransform: true });

    console.log('getBbox rect', rect);

    // If there are no eraser strokes, we can use the client rect directly
    for (const obj of this.objects.values()) {
      if (obj instanceof CanvasEraserLine) {
        needsPixelBbox = true;
        break;
      }
    }

    if (!needsPixelBbox) {
      if (rect.width === 0 || rect.height === 0) {
        this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
      } else {
        this.bbox = {
          x: this.konva.objectGroup.x() + rect.x,
          y: this.konva.objectGroup.y() + rect.y,
          width: rect.width,
          height: rect.height,
        };
      }
      this.renderBbox();
      return;
    }

    // We have eraser strokes - we must calculate the bbox using pixel data

    const clone = this.konva.objectGroup.clone();
    const canvas = clone.toCanvas();
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const imageData = ctx.getImageData(0, 0, rect.width, rect.height);
    this.manager.requestBbox(
      { buffer: imageData.data.buffer, width: imageData.width, height: imageData.height },
      (extents) => {
        if (extents) {
          const { minX, minY, maxX, maxY } = extents;
          this.bbox = {
            x: this.konva.objectGroup.x() + rect.x + minX,
            y: this.konva.objectGroup.y() + rect.y + minY,
            width: maxX - minX,
            height: maxY - minY,
          };
        } else {
          this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
        }
        console.log('new bbox', deepClone(this.bbox));
        this.renderBbox();
        clone.destroy();
      }
    );
  }
}
