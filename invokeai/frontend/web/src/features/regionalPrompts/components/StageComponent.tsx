import { Box } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useMouseEvents } from 'features/regionalPrompts/hooks/mouseEventHooks';
import {
  $cursorPosition,
  layerBboxChanged,
  layerTranslated,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { renderBbox, renderBrushPreview, renderLayers } from 'features/regionalPrompts/util/renderers';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { atom } from 'nanostores';
import { useCallback, useLayoutEffect } from 'react';

const log = logger('regionalPrompts');
const $stage = atom<Konva.Stage | null>(null);
const selectSelectedLayerColor = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  return regionalPrompts.layers.find((l) => l.id === regionalPrompts.selectedLayer)?.color ?? null;
});

const useStageRenderer = (container: HTMLDivElement | null, wrapper: HTMLDivElement | null) => {
  const dispatch = useAppDispatch();
  const width = useAppSelector((s) => s.generation.width);
  const height = useAppSelector((s) => s.generation.height);
  const state = useAppSelector((s) => s.regionalPrompts);
  const stage = useStore($stage);
  const { onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave } = useMouseEvents();
  const cursorPosition = useStore($cursorPosition);
  const selectedLayerColor = useAppSelector(selectSelectedLayerColor);

  const onLayerPosChanged = useCallback(
    (layerId: string, x: number, y: number) => {
      dispatch(layerTranslated({ layerId, x, y }));
    },
    [dispatch]
  );

  const onBboxChanged = useCallback(
    (layerId: string, bbox: IRect) => {
      dispatch(layerBboxChanged({ layerId, bbox }));
    },
    [dispatch]
  );

  useLayoutEffect(() => {
    log.trace('Initializing stage');
    if (!container) {
      return;
    }
    $stage.set(
      new Konva.Stage({
        container,
      })
    );
    return () => {
      log.trace('Cleaning up stage');
      $stage.get()?.destroy();
    };
  }, [container]);

  useLayoutEffect(() => {
    log.trace('Adding stage listeners');
    if (!stage) {
      return;
    }
    stage.on('mousedown', onMouseDown);
    stage.on('mouseup', onMouseUp);
    stage.on('mousemove', onMouseMove);
    stage.on('mouseenter', onMouseEnter);
    stage.on('mouseleave', onMouseLeave);

    return () => {
      log.trace('Cleaning up stage listeners');
      stage.off('mousedown', onMouseDown);
      stage.off('mouseup', onMouseUp);
      stage.off('mousemove', onMouseMove);
      stage.off('mouseenter', onMouseEnter);
      stage.off('mouseleave', onMouseLeave);
    };
  }, [stage, onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave]);

  useLayoutEffect(() => {
    log.trace('Updating stage dimensions');
    if (!stage || !wrapper) {
      return;
    }

    const fitStageToContainer = () => {
      const newXScale = wrapper.offsetWidth / width;
      const newYScale = wrapper.offsetHeight / height;
      const newScale = Math.min(newXScale, newYScale, 1);
      stage.width(width * newScale);
      stage.height(height * newScale);
      stage.scaleX(newScale);
      stage.scaleY(newScale);
    };

    const resizeObserver = new ResizeObserver(fitStageToContainer);
    resizeObserver.observe(wrapper);
    fitStageToContainer();

    return () => {
      resizeObserver.disconnect();
    };
  }, [stage, width, height, wrapper]);

  useLayoutEffect(() => {
    if (!stage) {
      return;
    }
    renderBrushPreview(stage, state.tool, selectedLayerColor, cursorPosition, state.brushSize);
  }, [stage, state.tool, cursorPosition, state.brushSize, selectedLayerColor]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    if (!stage) {
      return;
    }
    renderLayers(stage, state.layers, state.selectedLayer, state.promptLayerOpacity, state.tool, onLayerPosChanged);
  }, [onLayerPosChanged, stage, state.layers, state.promptLayerOpacity, state.tool, state.selectedLayer]);

  useLayoutEffect(() => {
    log.trace('Rendering bbox');
    if (!stage) {
      return;
    }
    renderBbox(stage, state.tool, state.selectedLayer, onBboxChanged);
  }, [dispatch, stage, state.tool, state.selectedLayer, onBboxChanged]);
};

const $container = atom<HTMLDivElement | null>(null);
const containerRef = (el: HTMLDivElement | null) => {
  $container.set(el);
};
const $wrapper = atom<HTMLDivElement | null>(null);
const wrapperRef = (el: HTMLDivElement | null) => {
  $wrapper.set(el);
};

export const StageComponent = () => {
  const container = useStore($container);
  const wrapper = useStore($wrapper);
  useStageRenderer(container, wrapper);
  return (
    <Box overflow="hidden" w="full" h="full">
      <Box ref={wrapperRef} w="full" h="full">
        <Box ref={containerRef} tabIndex={-1} bg="base.850" w="min-content" h="min-content" />
      </Box>
    </Box>
  );
};
