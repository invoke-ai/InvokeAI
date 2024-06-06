import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BRUSH_SPACING_PCT, MAX_BRUSH_SPACING_PX, MIN_BRUSH_SPACING_PX } from 'features/controlLayers/konva/constants';
import { setStageEventHandlers } from 'features/controlLayers/konva/events';
import { debouncedRenderers, renderers as normalRenderers } from 'features/controlLayers/konva/renderers/layers';
import {
  $brushColor,
  $brushSize,
  $brushSpacingPx,
  $isDrawing,
  $lastAddedPoint,
  $lastCursorPos,
  $lastMouseDownPos,
  $selectedLayer,
  $shouldInvertBrushSizeScrollDirection,
  $tool,
  brushLineAdded,
  brushSizeChanged,
  eraserLineAdded,
  layerBboxChanged,
  layerTranslated,
  linePointsAdded,
  rectAdded,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import type {
  AddBrushLineArg,
  AddEraserLineArg,
  AddPointToLineArg,
  AddRectShapeArg,
} from 'features/controlLayers/store/types';
import { isRegionalGuidanceLayer } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { clamp } from 'lodash-es';
import { memo, useCallback, useLayoutEffect, useMemo, useState } from 'react';
import { getImageDTO } from 'services/api/endpoints/images';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';
import { v4 as uuidv4 } from 'uuid';

// This will log warnings when layers > 5 - maybe use `import.meta.env.MODE === 'development'` instead?
Konva.showWarnings = false;

const log = logger('controlLayers');

const selectBrushColor = createMemoizedSelector(selectControlLayersSlice, (controlLayers) => {
  const layer = controlLayers.present.layers
    .filter(isRegionalGuidanceLayer)
    .find((l) => l.id === controlLayers.present.selectedLayerId);

  if (layer) {
    return { ...layer.previewColor, a: controlLayers.present.globalMaskLayerOpacity };
  }

  return controlLayers.present.brushColor;
});

const selectSelectedLayer = createSelector(selectControlLayersSlice, (controlLayers) => {
  return controlLayers.present.layers.find((l) => l.id === controlLayers.present.selectedLayerId) ?? null;
});

const useStageRenderer = (
  stage: Konva.Stage,
  container: HTMLDivElement | null,
  wrapper: HTMLDivElement | null,
  asPreview: boolean
) => {
  const dispatch = useAppDispatch();
  const state = useAppSelector((s) => s.controlLayers.present);
  const tool = useStore($tool);
  const lastCursorPos = useStore($lastCursorPos);
  const lastMouseDownPos = useStore($lastMouseDownPos);
  const isDrawing = useStore($isDrawing);
  const brushColor = useAppSelector(selectBrushColor);
  const selectedLayer = useAppSelector(selectSelectedLayer);
  const layerIds = useMemo(() => state.layers.map((l) => l.id), [state.layers]);
  const layerCount = useMemo(() => state.layers.length, [state.layers]);
  const renderers = useMemo(() => (asPreview ? debouncedRenderers : normalRenderers), [asPreview]);
  const dpr = useDevicePixelRatio({ round: false });
  const shouldInvertBrushSizeScrollDirection = useAppSelector((s) => s.canvas.shouldInvertBrushSizeScrollDirection);
  const brushSpacingPx = useMemo(
    () => clamp(state.brushSize / BRUSH_SPACING_PCT, MIN_BRUSH_SPACING_PX, MAX_BRUSH_SPACING_PX),
    [state.brushSize]
  );

  useLayoutEffect(() => {
    $brushColor.set(brushColor);
    $brushSize.set(state.brushSize);
    $brushSpacingPx.set(brushSpacingPx);
    $selectedLayer.set(selectedLayer);
    $shouldInvertBrushSizeScrollDirection.set(shouldInvertBrushSizeScrollDirection);
  }, [
    brushSpacingPx,
    brushColor,
    selectedLayer,
    shouldInvertBrushSizeScrollDirection,
    state.brushSize,
    state.selectedLayerId,
    state.brushColor,
  ]);

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

  const onBrushLineAdded = useCallback(
    (arg: AddBrushLineArg) => {
      dispatch(brushLineAdded(arg));
    },
    [dispatch]
  );
  const onEraserLineAdded = useCallback(
    (arg: AddEraserLineArg) => {
      dispatch(eraserLineAdded(arg));
    },
    [dispatch]
  );
  const onPointAddedToLine = useCallback(
    (arg: AddPointToLineArg) => {
      dispatch(linePointsAdded(arg));
    },
    [dispatch]
  );
  const onRectShapeAdded = useCallback(
    (arg: AddRectShapeArg) => {
      dispatch(rectAdded(arg));
    },
    [dispatch]
  );
  const onBrushSizeChanged = useCallback(
    (size: number) => {
      dispatch(brushSizeChanged(size));
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
    if (asPreview || !container) {
      return;
    }

    const cancelShape = (e: KeyboardEvent) => {
      // Cancel shape drawing on escape
      if (e.key === 'Escape') {
        $isDrawing.set(false);
        $lastMouseDownPos.set(null);
      }
    };

    container.addEventListener('keydown', cancelShape);

    const cleanup = setStageEventHandlers({
      stage,
      $tool,
      $isDrawing,
      $lastMouseDownPos,
      $lastCursorPos,
      $lastAddedPoint,
      $brushSize,
      $brushColor,
      $brushSpacingPx,
      $selectedLayer,
      $shouldInvertBrushSizeScrollDirection,
      onBrushSizeChanged,
      onBrushLineAdded,
      onEraserLineAdded,
      onPointAddedToLine,
      onRectShapeAdded,
    });

    return () => {
      log.trace('Removing stage listeners');
      cleanup();
      container.removeEventListener('keydown', cancelShape);
    };
  }, [
    asPreview,
    onBrushLineAdded,
    onBrushSizeChanged,
    onEraserLineAdded,
    onPointAddedToLine,
    onRectShapeAdded,
    stage,
    container,
  ]);

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
    if (asPreview) {
      // Preview should not display tool
      return;
    }
    log.trace('Rendering tool preview');
    renderers.renderToolPreview(
      stage,
      tool,
      brushColor,
      selectedLayer?.type ?? null,
      state.globalMaskLayerOpacity,
      lastCursorPos,
      lastMouseDownPos,
      state.brushSize,
      isDrawing
    );
  }, [
    asPreview,
    stage,
    tool,
    brushColor,
    selectedLayer,
    state.globalMaskLayerOpacity,
    lastCursorPos,
    lastMouseDownPos,
    state.brushSize,
    renderers,
    isDrawing,
  ]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    renderers.renderLayers(stage, state.layers, state.globalMaskLayerOpacity, tool, getImageDTO, onLayerPosChanged);
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
    if (asPreview) {
      // Preview should not check for transparency
      return;
    }
    log.trace('Updating bboxes');
    debouncedRenderers.updateBboxes(stage, state.layers, onBboxChanged);
  }, [stage, asPreview, state.layers, onBboxChanged]);

  useLayoutEffect(() => {
    if (asPreview) {
      // The preview should not have a background
      return;
    }
    log.trace('Rendering background');
    renderers.renderBackground(stage, state.size.width, state.size.height);
  }, [stage, asPreview, state.size.width, state.size.height, renderers]);

  useLayoutEffect(() => {
    log.trace('Arranging layers');
    renderers.arrangeLayers(stage, layerIds);
  }, [stage, layerIds, renderers]);

  useLayoutEffect(() => {
    if (asPreview) {
      // The preview should not display the no layers message
      return;
    }
    log.trace('Rendering no layers message');
    renderers.renderNoLayersMessage(stage, layerCount, state.size.width, state.size.height);
  }, [stage, layerCount, renderers, asPreview, state.size.width, state.size.height]);

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
        <Flex
          ref={containerRef}
          tabIndex={-1}
          bg="base.850"
          borderRadius="base"
          overflow="hidden"
          data-testid="control-layers-canvas"
        />
      </Flex>
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';
