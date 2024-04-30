import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useMouseEvents } from 'features/controlLayers/hooks/mouseEventHooks';
import {
  $cursorPosition,
  $isMouseOver,
  $lastMouseDownPos,
  $tool,
  isMaskedGuidanceLayer,
  layerBboxChanged,
  layerTranslated,
  selectRegionalPromptsSlice,
} from 'features/controlLayers/store/regionalPromptsSlice';
import { debouncedRenderers, renderers as normalRenderers } from 'features/controlLayers/util/renderers';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { memo, useCallback, useLayoutEffect, useMemo, useState } from 'react';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';
import { v4 as uuidv4 } from 'uuid';

// This will log warnings when layers > 5 - maybe use `import.meta.env.MODE === 'development'` instead?
Konva.showWarnings = false;

const log = logger('regionalPrompts');

const selectSelectedLayerColor = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  const layer = regionalPrompts.present.layers
    .filter(isMaskedGuidanceLayer)
    .find((l) => l.id === regionalPrompts.present.selectedLayerId);
  return layer?.previewColor ?? null;
});

const selectSelectedLayerType = createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  const selectedLayer = regionalPrompts.present.layers.find((l) => l.id === regionalPrompts.present.selectedLayerId);
  return selectedLayer?.type ?? null;
});

const useStageRenderer = (
  stage: Konva.Stage,
  container: HTMLDivElement | null,
  wrapper: HTMLDivElement | null,
  asPreview: boolean
) => {
  const dispatch = useAppDispatch();
  const state = useAppSelector((s) => s.regionalPrompts.present);
  const tool = useStore($tool);
  const { onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave, onMouseWheel } = useMouseEvents();
  const cursorPosition = useStore($cursorPosition);
  const lastMouseDownPos = useStore($lastMouseDownPos);
  const isMouseOver = useStore($isMouseOver);
  const selectedLayerIdColor = useAppSelector(selectSelectedLayerColor);
  const selectedLayerType = useAppSelector(selectSelectedLayerType);
  const layerIds = useMemo(() => state.layers.map((l) => l.id), [state.layers]);
  const renderers = useMemo(() => (asPreview ? debouncedRenderers : normalRenderers), [asPreview]);
  const dpr = useDevicePixelRatio({ round: false });

  const onLayerPosChanged = useCallback(
    (layerId: string, x: number, y: number) => {
      dispatch(layerTranslated({ layerId, x, y }));
    },
    [dispatch]
  );

  const onBboxChanged = useCallback(
    (layerId: string, bbox: IRect | null) => {
      dispatch(layerBboxChanged({ layerId, bbox }));
    },
    [dispatch]
  );

  useLayoutEffect(() => {
    log.trace('Initializing stage');
    if (!container) {
      return;
    }
    stage.container(container);
    return () => {
      log.trace('Cleaning up stage');
      stage.destroy();
    };
  }, [container, stage]);

  useLayoutEffect(() => {
    log.trace('Adding stage listeners');
    if (asPreview) {
      return;
    }
    stage.on('mousedown', onMouseDown);
    stage.on('mouseup', onMouseUp);
    stage.on('mousemove', onMouseMove);
    stage.on('mouseenter', onMouseEnter);
    stage.on('mouseleave', onMouseLeave);
    stage.on('wheel', onMouseWheel);

    return () => {
      log.trace('Cleaning up stage listeners');
      stage.off('mousedown', onMouseDown);
      stage.off('mouseup', onMouseUp);
      stage.off('mousemove', onMouseMove);
      stage.off('mouseenter', onMouseEnter);
      stage.off('mouseleave', onMouseLeave);
      stage.off('wheel', onMouseWheel);
    };
  }, [stage, asPreview, onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave, onMouseWheel]);

  useLayoutEffect(() => {
    log.trace('Updating stage dimensions');
    if (!wrapper) {
      return;
    }

    const fitStageToContainer = () => {
      const newXScale = wrapper.offsetWidth / state.size.width;
      const newYScale = wrapper.offsetHeight / state.size.height;
      const newScale = Math.min(newXScale, newYScale, 1);
      stage.width(state.size.width * newScale);
      stage.height(state.size.height * newScale);
      stage.scaleX(newScale);
      stage.scaleY(newScale);
    };

    const resizeObserver = new ResizeObserver(fitStageToContainer);
    resizeObserver.observe(wrapper);
    fitStageToContainer();

    return () => {
      resizeObserver.disconnect();
    };
  }, [stage, state.size.width, state.size.height, wrapper]);

  useLayoutEffect(() => {
    log.trace('Rendering tool preview');
    if (asPreview) {
      // Preview should not display tool
      return;
    }
    renderers.renderToolPreview(
      stage,
      tool,
      selectedLayerIdColor,
      selectedLayerType,
      state.globalMaskLayerOpacity,
      cursorPosition,
      lastMouseDownPos,
      isMouseOver,
      state.brushSize
    );
  }, [
    asPreview,
    stage,
    tool,
    selectedLayerIdColor,
    selectedLayerType,
    state.globalMaskLayerOpacity,
    cursorPosition,
    lastMouseDownPos,
    isMouseOver,
    state.brushSize,
    renderers,
  ]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    renderers.renderLayers(stage, state.layers, state.globalMaskLayerOpacity, tool, onLayerPosChanged);
  }, [
    stage,
    state.layers,
    state.globalMaskLayerOpacity,
    tool,
    onLayerPosChanged,
    renderers,
    state.size.width,
    state.size.height,
  ]);

  useLayoutEffect(() => {
    log.trace('Rendering bbox');
    if (asPreview) {
      // Preview should not display bboxes
      return;
    }
    renderers.renderBbox(stage, state.layers, tool, onBboxChanged);
  }, [stage, asPreview, state.layers, tool, onBboxChanged, renderers]);

  useLayoutEffect(() => {
    log.trace('Rendering background');
    if (asPreview) {
      // The preview should not have a background
      return;
    }
    renderers.renderBackground(stage, state.size.width, state.size.height);
  }, [stage, asPreview, state.size.width, state.size.height, renderers]);

  useLayoutEffect(() => {
    log.trace('Arranging layers');
    renderers.arrangeLayers(stage, layerIds);
  }, [stage, layerIds, renderers]);

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);
};

type Props = {
  asPreview?: boolean;
};

export const StageComponent = memo(({ asPreview = false }: Props) => {
  const [stage] = useState(
    () => new Konva.Stage({ id: uuidv4(), container: document.createElement('div'), listening: !asPreview })
  );
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const [wrapper, setWrapper] = useState<HTMLDivElement | null>(null);

  const containerRef = useCallback((el: HTMLDivElement | null) => {
    setContainer(el);
  }, []);

  const wrapperRef = useCallback((el: HTMLDivElement | null) => {
    setWrapper(el);
  }, []);

  useStageRenderer(stage, container, wrapper, asPreview);

  return (
    <Flex overflow="hidden" w="full" h="full">
      <Flex ref={wrapperRef} w="full" h="full" alignItems="center" justifyContent="center">
        <Flex ref={containerRef} tabIndex={-1} bg="base.850" />
      </Flex>
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';
