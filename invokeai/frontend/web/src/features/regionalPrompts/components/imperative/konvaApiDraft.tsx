import { chakra } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import {
  $cursorPosition,
  BRUSH_PREVIEW_BORDER_INNER_ID,
  BRUSH_PREVIEW_BORDER_OUTER_ID,
  BRUSH_PREVIEW_FILL_ID,
  BRUSH_PREVIEW_LAYER_ID,
  getPromptRegionLayerObjectGroupId,
  layerBboxChanged,
  layerSelected,
  layerTranslated,
  REGIONAL_PROMPT_LAYER_NAME,
  REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getKonvaLayerBbox } from 'features/regionalPrompts/util/bbox';
import Konva from 'konva';
import type { Node, NodeConfig } from 'konva/lib/Node';
import { atom } from 'nanostores';
import { useLayoutEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

import { useMouseDown, useMouseEnter, useMouseLeave, useMouseMove, useMouseUp } from './mouseEventHooks';

export const $stage = atom<Konva.Stage | null>(null);

type Props = {
  container: HTMLDivElement | null;
};

export const selectPromptLayerObjectGroup = (item: Node<NodeConfig>) =>
  item.name() !== REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME;

const isKonvaLayer = (node: Node<NodeConfig>): node is Konva.Layer => node.nodeType === 'Layer';
const isKonvaLine = (node: Node<NodeConfig>): node is Konva.Line => node.nodeType === 'Line';
const isKonvaGroup = (node: Node<NodeConfig>): node is Konva.Group => node.nodeType === 'Group';
const isKonvaRect = (node: Node<NodeConfig>): node is Konva.Rect => node.nodeType === 'Rect';

const $brushPreviewNodes = atom<{
  layer: Konva.Layer;
  fill: Konva.Circle;
  borderInner: Konva.Circle;
  borderOuter: Konva.Circle;
} | null>(null);

export const LogicalStage = (props: Props) => {
  const dispatch = useAppDispatch();
  const width = useAppSelector((s) => s.generation.width);
  const height = useAppSelector((s) => s.generation.height);
  const state = useAppSelector((s) => s.regionalPrompts);
  const stage = useStore($stage);
  const onMouseDown = useMouseDown();
  const onMouseUp = useMouseUp();
  const onMouseMove = useMouseMove();
  const onMouseEnter = useMouseEnter();
  const onMouseLeave = useMouseLeave();
  const cursorPosition = useStore($cursorPosition);

  useLayoutEffect(() => {
    console.log('init effect');
    if (!props.container) {
      return;
    }

    const stage = new Konva.Stage({
      container: props.container,
    });

    const brushPreviewLayer = new Konva.Layer({ id: BRUSH_PREVIEW_LAYER_ID });
    stage.add(brushPreviewLayer);
    const fill = new Konva.Circle({
      id: BRUSH_PREVIEW_FILL_ID,
      listening: false,
      strokeEnabled: false,
      strokeHitEnabled: false,
    });
    const borderInner = new Konva.Circle({
      id: BRUSH_PREVIEW_BORDER_INNER_ID,
      listening: false,
      stroke: 'rgba(0,0,0,1)',
      strokeWidth: 1,
      strokeEnabled: true,
    });
    const borderOuter = new Konva.Circle({
      id: BRUSH_PREVIEW_BORDER_OUTER_ID,
      listening: false,
      stroke: 'rgba(255,255,255,0.8)',
      strokeWidth: 1,
      strokeEnabled: true,
    });
    brushPreviewLayer.add(fill);
    brushPreviewLayer.add(borderInner);
    brushPreviewLayer.add(borderOuter);
    $brushPreviewNodes.set({ layer: brushPreviewLayer, fill, borderInner: borderInner, borderOuter: borderOuter });

    $stage.set(stage);

    return () => {
      const stage = $stage.get();
      if (!stage) {
        return;
      }
      stage.destroy();
    };
  }, [props.container]);

  useLayoutEffect(() => {
    console.log('event effect');
    if (!stage) {
      return;
    }
    stage.on('mousedown', onMouseDown);
    stage.on('mouseup', onMouseUp);
    stage.on('mousemove', onMouseMove);
    stage.on('mouseenter', onMouseEnter);
    stage.on('mouseleave', onMouseLeave);

    return () => {
      stage.off('mousedown', onMouseDown);
      stage.off('mouseup', onMouseUp);
      stage.off('mousemove', onMouseMove);
      stage.off('mouseenter', onMouseEnter);
      stage.off('mouseleave', onMouseLeave);
    };
  }, [stage, onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave]);

  useLayoutEffect(() => {
    console.log('stage dims effect');
    if (!stage || !props.container) {
      return;
    }
    stage.width(width);
    stage.height(height);
  }, [stage, width, height, props.container]);

  useLayoutEffect(() => {
    console.log('brush preview effect');
    const brushPreviewNodes = $brushPreviewNodes.get();
    brushPreviewNodes?.layer.visible(state.tool !== 'move');
    if (!stage || !cursorPosition || !brushPreviewNodes) {
      return;
    }
    const color = getStore()
      .getState()
      .regionalPrompts.layers.find((l) => l.id === state.selectedLayer)?.color;
    if (!color) {
      return;
    }
    const fill = rgbColorToString(color);
    brushPreviewNodes.fill.setAttrs({
      x: cursorPosition.x,
      y: cursorPosition.y,
      radius: state.brushSize / 2,
      fill,
      globalCompositeOperation: state.tool === 'brush' ? 'source-over' : 'destination-out',
    });
    brushPreviewNodes.borderInner.setAttrs({ x: cursorPosition.x, y: cursorPosition.y, radius: state.brushSize / 2 });
    brushPreviewNodes.borderOuter.setAttrs({
      x: cursorPosition.x,
      y: cursorPosition.y,
      radius: state.brushSize / 2 + 1,
    });
  }, [cursorPosition, stage, state.brushSize, state.selectedLayer, state.tool]);

  useLayoutEffect(() => {
    console.log('obj effect');
    if (!stage) {
      return;
    }

    const reduxLayerIds = state.layers.map((l) => l.id);

    // Remove deleted layers - we know these are of type Layer
    for (const konvaLayer of stage.find<Konva.Layer>(`.${REGIONAL_PROMPT_LAYER_NAME}`)) {
      if (!reduxLayerIds.includes(konvaLayer.id())) {
        konvaLayer.destroy();
      }
    }

    for (const reduxLayer of state.layers) {
      let konvaLayer = stage.findOne<Konva.Layer>(`#${reduxLayer.id}`);

      // New layer - create a new Konva layer
      if (!konvaLayer) {
        konvaLayer = new Konva.Layer({
          id: reduxLayer.id,
          name: REGIONAL_PROMPT_LAYER_NAME,
          draggable: true,
          listening: reduxLayer.id === state.selectedLayer,
          x: reduxLayer.x,
          y: reduxLayer.y,
        });
        konvaLayer.on('dragmove', function (e) {
          dispatch(
            layerTranslated({
              layerId: reduxLayer.id,
              x: e.target.x(),
              y: e.target.y(),
            })
          );
        });
        konvaLayer.dragBoundFunc(function (pos) {
          const cursorPos = getScaledCursorPosition(stage);
          if (!cursorPos) {
            return this.getAbsolutePosition();
          }
          // This prevents the user from dragging the object out of the stage.
          if (cursorPos.x < 0 || cursorPos.x > stage.width() || cursorPos.y < 0 || cursorPos.y > stage.height()) {
            return this.getAbsolutePosition();
          }

          return pos;
        });
        stage.add(konvaLayer);
        konvaLayer.add(
          new Konva.Group({
            id: getPromptRegionLayerObjectGroupId(reduxLayer.id, uuidv4()),
            name: REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
            listening: false,
          })
        );
        $brushPreviewNodes.get()?.layer.moveToTop();
      } else {
        konvaLayer.listening(reduxLayer.id === state.selectedLayer);
        konvaLayer.x(reduxLayer.x);
        konvaLayer.y(reduxLayer.y);
      }

      const color = rgbColorToString(reduxLayer.color);
      const konvaObjectGroup = konvaLayer.findOne<Konva.Group>(`.${REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME}`);

      // Remove deleted objects
      const objectIds = reduxLayer.objects.map((o) => o.id);
      for (const objectNode of stage.find(`.${reduxLayer.id}-object`)) {
        if (!objectIds.includes(objectNode.id())) {
          objectNode.destroy();
        }
      }

      for (const reduxObject of reduxLayer.objects) {
        // TODO: Handle rects, images, etc
        if (reduxObject.kind !== 'line') {
          return;
        }
        const konvaObject = stage.findOne<Konva.Line>(`#${reduxObject.id}`);

        if (!konvaObject) {
          // This object hasn't been added to the konva state yet.
          konvaObjectGroup?.add(
            new Konva.Line({
              id: reduxObject.id,
              key: reduxObject.id,
              name: `${reduxLayer.id}-object`,
              points: reduxObject.points,
              strokeWidth: reduxObject.strokeWidth,
              stroke: color,
              tension: 0,
              lineCap: 'round',
              lineJoin: 'round',
              shadowForStrokeEnabled: false,
              globalCompositeOperation: reduxObject.tool === 'brush' ? 'source-over' : 'destination-out',
              listening: false,
              visible: reduxLayer.isVisible,
            })
          );
        } else {
          // Only update the points if they have changed. The point values are never mutated, they are only added to the array.
          if (konvaObject.points().length !== reduxObject.points.length) {
            konvaObject.points(reduxObject.points);
          }
          // Only update the color if it has changed.
          if (konvaObject.stroke() !== color) {
            konvaObject.stroke(color);
          }
          // Only update layer visibility if it has changed.
          if (konvaObject.visible() !== reduxLayer.isVisible) {
            konvaObject.visible(reduxLayer.isVisible);
          }
        }
      }
    }
  }, [dispatch, stage, state.tool, state.layers, state.selectedLayer]);

  useLayoutEffect(() => {
    if (!stage) {
      return;
    }
    stage.container().style.cursor = state.tool === 'move' ? 'default' : 'none';
  }, [stage, state.tool]);

  useLayoutEffect(() => {
    console.log('bbox effect');
    if (!stage) {
      return;
    }

    if (state.tool !== 'move') {
      // Tool was just changed to something other than move - hide all layer bounding boxes
      for (const n of stage.find('.layer-bbox')) {
        n.visible(false);
      }
      return;
    }

    for (const konvaLayer of stage.find<Konva.Layer>(`.${REGIONAL_PROMPT_LAYER_NAME}`)) {
      const bbox = getKonvaLayerBbox(konvaLayer);
      dispatch(layerBboxChanged({ layerId: konvaLayer.id(), bbox }));
      let rect = konvaLayer.findOne<Konva.Rect>('.layer-bbox');
      if (!rect) {
        rect = new Konva.Rect({
          id: `${konvaLayer.id()}-bbox`,
          name: 'layer-bbox',
          strokeWidth: 1,
        });
        konvaLayer.add(rect);
        konvaLayer.on('mousedown', () => {
          dispatch(layerSelected(konvaLayer.id()));
        });
      }
      rect.visible(true);
      rect.x(bbox.x);
      rect.y(bbox.y);
      rect.width(bbox.width);
      rect.height(bbox.height);
      rect.stroke(state.selectedLayer === konvaLayer.id() ? 'rgba(153, 187, 189, 1)' : 'rgba(255, 255, 255, 0.149)');
    }
  }, [dispatch, stage, state.tool, state.selectedLayer]);

  return null;
};

const $container = atom<HTMLDivElement | null>(null);
const containerRef = (el: HTMLDivElement | null) => {
  $container.set(el);
};

export const StageComponent = () => {
  const container = useStore($container);
  return (
    <>
      <chakra.div ref={containerRef} tabIndex={-1} sx={{ borderWidth: 1, borderRadius: 'base' }} />
      <LogicalStage container={container} />
    </>
  );
};
