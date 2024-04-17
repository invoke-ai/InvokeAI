import { chakra } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import type { Layer, Tool } from 'features/regionalPrompts/store/regionalPromptsSlice';
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
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getKonvaLayerBbox } from 'features/regionalPrompts/util/bbox';
import Konva from 'konva';
import type { KonvaEventObject, Node, NodeConfig } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';
import { atom } from 'nanostores';
import { useCallback, useLayoutEffect } from 'react';
import type { RgbColor } from 'react-colorful';
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

const renderBrushPreview = (
  stage: Konva.Stage,
  tool: Tool,
  color: RgbColor,
  cursorPos: Vector2d,
  brushSize: number
) => {
  // Create the layer if it doesn't exist
  let layer = stage.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`);
  if (!layer) {
    layer = new Konva.Layer({ id: BRUSH_PREVIEW_LAYER_ID, visible: tool !== 'move' });
    stage.add(layer);
  }

  // The brush preview is hidden when using the move tool
  layer.visible(tool !== 'move');

  // Create and/or update the fill circle
  let fill = layer.findOne<Konva.Circle>(`#${BRUSH_PREVIEW_FILL_ID}`);
  if (!fill) {
    fill = new Konva.Circle({
      id: BRUSH_PREVIEW_FILL_ID,
      listening: false,
      strokeEnabled: false,
      strokeHitEnabled: false,
    });
    layer.add(fill);
  }
  fill.setAttrs({
    x: cursorPos.x,
    y: cursorPos.y,
    radius: brushSize / 2,
    fill: rgbColorToString(color),
    globalCompositeOperation: tool === 'brush' ? 'source-over' : 'destination-out',
  });

  // Create and/or update the inner border of the brush preview
  let borderInner = layer.findOne<Konva.Circle>(`#${BRUSH_PREVIEW_BORDER_INNER_ID}`);
  if (!borderInner) {
    borderInner = new Konva.Circle({
      id: BRUSH_PREVIEW_BORDER_INNER_ID,
      listening: false,
      stroke: 'rgba(0,0,0,1)',
      strokeWidth: 1,
      strokeEnabled: true,
    });
    layer.add(borderInner);
  }
  borderInner.setAttrs({ x: cursorPos.x, y: cursorPos.y, radius: brushSize / 2 });

  // Create and/or update the outer border of the brush preview
  let borderOuter = layer.findOne<Konva.Circle>(`#${BRUSH_PREVIEW_BORDER_OUTER_ID}`);
  if (!borderOuter) {
    borderOuter = new Konva.Circle({
      id: BRUSH_PREVIEW_BORDER_OUTER_ID,
      listening: false,
      stroke: 'rgba(255,255,255,0.8)',
      strokeWidth: 1,
      strokeEnabled: true,
    });
    layer.add(borderOuter);
  }
  borderOuter.setAttrs({
    x: cursorPos.x,
    y: cursorPos.y,
    radius: brushSize / 2 + 1,
  });
};

export const renderLayers = (
  stage: Konva.Stage,
  reduxLayers: Layer[],
  selectedLayerId: string | null,
  getOnDragMove?: (layerId: string) => (e: KonvaEventObject<MouseEvent>) => void
) => {
  const reduxLayerIds = reduxLayers.map((l) => l.id);

  // Remove deleted layers - we know these are of type Layer
  for (const konvaLayer of stage.find<Konva.Layer>(`.${REGIONAL_PROMPT_LAYER_NAME}`)) {
    if (!reduxLayerIds.includes(konvaLayer.id())) {
      konvaLayer.destroy();
    }
  }

  for (const reduxLayer of reduxLayers) {
    let konvaLayer = stage.findOne<Konva.Layer>(`#${reduxLayer.id}`);

    // New layer - create a new Konva layer
    if (!konvaLayer) {
      konvaLayer = new Konva.Layer({
        id: reduxLayer.id,
        name: REGIONAL_PROMPT_LAYER_NAME,
        draggable: true,
        listening: reduxLayer.id === selectedLayerId,
        x: reduxLayer.x,
        y: reduxLayer.y,
      });
      if (getOnDragMove) {
        konvaLayer.on('dragmove', getOnDragMove(reduxLayer.id));
      }
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
      // Brush preview should always be the top layer
      stage.findOne<Konva.Layer>(`#${BRUSH_PREVIEW_LAYER_ID}`)?.moveToTop();
    } else {
      konvaLayer.listening(reduxLayer.id === selectedLayerId);
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
};

const selectSelectedLayerColor = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  return regionalPrompts.layers.find((l) => l.id === regionalPrompts.selectedLayer)?.color;
});

export const LogicalStage = ({ container }: Props) => {
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
  const selectedLayerColor = useAppSelector(selectSelectedLayerColor);

  useLayoutEffect(() => {
    if (!stage || !cursorPosition || !selectedLayerColor) {
      return;
    }
    renderBrushPreview(stage, state.tool, selectedLayerColor, cursorPosition, state.brushSize);
  }, [stage, state.tool, cursorPosition, state.brushSize, selectedLayerColor]);

  useLayoutEffect(() => {
    console.log('init effect');
    if (!container) {
      return;
    }
    $stage.set(
      new Konva.Stage({
        container,
      })
    );
    return () => {
      $stage.get()?.destroy();
    };
  }, [container]);

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
    if (!stage) {
      return;
    }
    stage.width(width);
    stage.height(height);
  }, [stage, width, height]);

  const getOnDragMove = useCallback(
    (layerId: string) => (e: KonvaEventObject<MouseEvent>) => {
      dispatch(layerTranslated({ layerId, x: e.target.x(), y: e.target.y() }));
    },
    [dispatch]
  );

  useLayoutEffect(() => {
    console.log('obj effect');
    if (!stage) {
      return;
    }

    renderLayers(stage, state.layers, state.selectedLayer, getOnDragMove);
  }, [getOnDragMove, stage, state.layers, state.selectedLayer]);

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
      <chakra.div
        ref={containerRef}
        tabIndex={-1}
        sx={{ borderWidth: 1, borderRadius: 'base', flexGrow: 0, h: 'min-content' }}
      />
      <LogicalStage container={container} />
    </>
  );
};
