import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import getScaledCursorPosition from 'features/canvas/util/getScaledCursorPosition';
import { BrushPreviewFill } from 'features/regionalPrompts/components/BrushPreview';
import { LayerBoundingBox } from 'features/regionalPrompts/components/LayerBoundingBox';
import { LineComponent } from 'features/regionalPrompts/components/LineComponent';
import { RectComponent } from 'features/regionalPrompts/components/RectComponent';
import { useLayer } from 'features/regionalPrompts/hooks/layerStateHooks';
import {
  $stage,
  layerBboxChanged,
  layerTranslated,
  REGIONAL_PROMPT_LAYER_NAME,
  REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getKonvaLayerBbox } from 'features/regionalPrompts/util/bbox';
import type { Group as KonvaGroupType } from 'konva/lib/Group';
import type { Layer as KonvaLayerType } from 'konva/lib/Layer';
import type { KonvaEventObject, Node as KonvaNodeType, NodeConfig as KonvaNodeConfigType } from 'konva/lib/Node';
import type { IRect, Vector2d } from 'konva/lib/types';
import type React from 'react';
import { useCallback, useEffect, useRef } from 'react';
import { Group as KonvaGroup, Layer as KonvaLayer } from 'react-konva';

type Props = {
  id: string;
};

export const selectPromptLayerObjectGroup = (item: KonvaNodeType<KonvaNodeConfigType>) =>
  item.name() !== REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME;

export const LayerComponent: React.FC<Props> = ({ id }) => {
  const dispatch = useAppDispatch();
  const layer = useLayer(id);
  const selectedLayer = useAppSelector((s) => s.regionalPrompts.selectedLayer);
  const promptLayerOpacity = useAppSelector((s) => s.regionalPrompts.promptLayerOpacity);
  const tool = useAppSelector((s) => s.regionalPrompts.tool);
  const layerRef = useRef<KonvaLayerType>(null);
  const groupRef = useRef<KonvaGroupType>(null);

  const onChangeBbox = useCallback(
    (bbox: IRect | null) => {
      dispatch(layerBboxChanged({ layerId: layer.id, bbox }));
    },
    [dispatch, layer.id]
  );

  const onDragMove = useCallback(
    (e: KonvaEventObject<DragEvent>) => {
      // We must track the layer's coordinates to accurately position objects. For example, when drawing a line, the
      // mouse events are handled by the stage, but the line is drawn on the layer.
      dispatch(layerTranslated({ layerId: id, x: e.target.x(), y: e.target.y() }));
    },
    [dispatch, id]
  );

  const dragBoundFunc = useCallback(function (this: KonvaNodeType<KonvaNodeConfigType>, pos: Vector2d) {
    const stage = $stage.get();
    if (!stage) {
      return this.getAbsolutePosition();
    }
    const cursorPos = getScaledCursorPosition(stage);
    if (!cursorPos) {
      return this.getAbsolutePosition();
    }
    // This prevents the user from dragging the object out of the stage.
    if (cursorPos.x < 0 || cursorPos.x > stage.width() || cursorPos.y < 0 || cursorPos.y > stage.height()) {
      return this.getAbsolutePosition();
    }

    return pos;
  }, []);

  useEffect(() => {
    if (!layerRef.current || tool !== 'move') {
      // This effectively makes the bbox only calculate as the tool is changed to 'move'.
      return;
    }
    if (layer.objects.length === 0) {
      // Reset the bbox when we have no objects.
      onChangeBbox(null);
      return;
    }
    // We could debounce this, remove the early return when the tool isn't move, and always show the bbox. For now,
    // it is only shown when we are using the move tool.
    onChangeBbox(getKonvaLayerBbox(layerRef.current, selectPromptLayerObjectGroup));
  }, [tool, layer.objects, onChangeBbox]);

  useEffect(() => {
    if (!groupRef.current) {
      return;
    }
    // Caching is not allowed for "empty" nodes. We must draw here to render the group instead. This is required
    // to clear the layer when the layer is reset and on first render, when the layer has no objects.
    if (layer.objects.length === 0) {
      groupRef.current.draw();
      return;
    }
    // Caching the group allows its opacity to apply to all shapes at once. We should cache only when the layer's
    // objects or attributes with a visual effect (e.g. color) change.
    groupRef.current.cache();
  }, [layer.objects, layer.color, layer.isVisible]);

  if (!layer.isVisible) {
    return null;
  }

  return (
    <>
      <KonvaLayer
        ref={layerRef}
        id={layer.id}
        name={REGIONAL_PROMPT_LAYER_NAME}
        onDragMove={onDragMove}
        dragBoundFunc={dragBoundFunc}
        draggable
      >
        <KonvaGroup
          id={`layer-${layer.id}-group`}
          name={REGIONAL_PROMPT_LAYER_OBJECT_GROUP_NAME}
          ref={groupRef}
          listening={false}
          opacity={promptLayerOpacity}
        >
          {layer.objects.map((obj) => {
            if (obj.kind === 'line') {
              return <LineComponent key={obj.id} line={obj} color={layer.color} layerId={layer.id} />;
            }
            if (obj.kind === 'fillRect') {
              return <RectComponent key={obj.id} rect={obj} color={layer.color} />;
            }
          })}
        </KonvaGroup>
        <LayerBoundingBox layerId={layer.id} />
      </KonvaLayer>
      <KonvaLayer name="brushPreviewFill">{layer.id === selectedLayer && <BrushPreviewFill />}</KonvaLayer>
    </>
  );
};
