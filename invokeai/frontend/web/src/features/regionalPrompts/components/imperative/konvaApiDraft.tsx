import { chakra } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import {
  $cursorPosition,
  layerBboxChanged,
  layerSelected,
  layerTranslated,
  REGIONAL_PROMPT_LAYER_NAME,
  REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getKonvaLayerBbox } from 'features/regionalPrompts/util/bbox';
import Konva from 'konva';
import type { Node, NodeConfig } from 'konva/lib/Node';
import type { StageConfig } from 'konva/lib/Stage';
import { atom } from 'nanostores';
import { useLayoutEffect } from 'react';

import { useMouseDown, useMouseEnter, useMouseLeave, useMouseMove, useMouseUp } from './mouseEventHooks';

export const $stage = atom<Konva.Stage | null>(null);

const initStage = (container: StageConfig['container']) => {
  const stage = new Konva.Stage({
    container,
  });
  $stage.set(stage);

  const layer = new Konva.Layer();
  const circle = new Konva.Circle({ id: 'cursor', radius: 5, fill: 'red' });
  layer.add(circle);
  stage.add(layer);
};

type Props = {
  container: HTMLDivElement | null;
};

export const selectPromptLayerObjectGroup = (item: Node<NodeConfig>) =>
  item.name() !== REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME;

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
    initStage(props.container);
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
    console.log('cursor effect');
    if (!stage || !cursorPosition) {
      return;
    }
    const cursor = stage.findOne('#cursor');
    if (!cursor) {
      return;
    }
    cursor.x(cursorPosition?.x);
    cursor.y(cursorPosition?.y);
  }, [cursorPosition, stage]);

  useLayoutEffect(() => {
    console.log('obj effect');
    if (!stage) {
      return;
    }

    // TODO: Handle layer getting deleted and reset
    for (const l of state.layers) {
      let layer = stage.findOne(`#${l.id}`) as Konva.Layer | null;
      if (!layer) {
        layer = new Konva.Layer({ id: l.id, name: REGIONAL_PROMPT_LAYER_NAME, draggable: true });
        layer.on('dragmove', (e) => {
          dispatch(layerTranslated({ layerId: l.id, x: e.target.x(), y: e.target.y() }));
        });
        layer.dragBoundFunc(function (pos) {
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
        stage.add(layer);
      }

      if (state.tool === 'move') {
        layer.listening(true);
      } else {
        layer.listening(l.id === state.selectedLayer);
      }

      for (const o of l.objects) {
        if (o.kind !== 'line') {
          return;
        }
        let obj = stage.findOne(`#${o.id}`) as Konva.Line | null;
        if (!obj) {
          obj = new Konva.Line({
            id: o.id,
            key: o.id,
            strokeWidth: o.strokeWidth,
            stroke: rgbColorToString(l.color),
            tension: 0,
            lineCap: 'round',
            lineJoin: 'round',
            shadowForStrokeEnabled: false,
            globalCompositeOperation: o.tool === 'brush' ? 'source-over' : 'destination-out',
            listening: false,
            visible: l.isVisible,
          });
          layer.add(obj);
        }
        if (obj.points().length < o.points.length) {
          obj.points(o.points);
        }
        if (obj.stroke() !== rgbColorToString(l.color)) {
          obj.stroke(rgbColorToString(l.color));
        }
        if (obj.visible() !== l.isVisible) {
          obj.visible(l.isVisible);
        }
      }
    }
  }, [dispatch, stage, state.tool, state.layers, state.selectedLayer]);

  useLayoutEffect(() => {
    if (!stage) {
      return;
    }

    if (state.tool !== 'move') {
      for (const n of stage.find('.layer-bbox')) {
        n.visible(false);
      }
      return;
    }

    for (const layer of stage.find(`.${REGIONAL_PROMPT_LAYER_NAME}`) as Konva.Layer[]) {
      const bbox = getKonvaLayerBbox(layer);
      dispatch(layerBboxChanged({ layerId: layer.id(), bbox }));
      let rect = layer.findOne('.layer-bbox') as Konva.Rect | null;
      if (!rect) {
        rect = new Konva.Rect({
          id: `${layer.id()}-bbox`,
          name: 'layer-bbox',
          strokeWidth: 1,
        });
        layer.add(rect);
        layer.on('mousedown', () => {
          dispatch(layerSelected(layer.id()));
        });
      }
      rect.visible(true);
      rect.x(bbox.x);
      rect.y(bbox.y);
      rect.width(bbox.width);
      rect.height(bbox.height);
      rect.stroke(state.selectedLayer === layer.id() ? 'rgba(153, 187, 189, 1)' : 'rgba(255, 255, 255, 0.149)');
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
