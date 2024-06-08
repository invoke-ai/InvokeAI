import { Box, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  BRUSH_SPACING_PCT,
  MAX_BRUSH_SPACING_PX,
  MIN_BRUSH_SPACING_PX,
  TRANSPARENCY_CHECKER_PATTERN,
} from 'features/controlLayers/konva/constants';
import { setStageEventHandlers } from 'features/controlLayers/konva/events';
import { debouncedRenderers, renderers as normalRenderers } from 'features/controlLayers/konva/renderers/layers';
import { renderImageDimsPreview } from 'features/controlLayers/konva/renderers/previewLayer';
import {
  $brushColor,
  $brushSize,
  $brushSpacingPx,
  $genBbox,
  $isDrawing,
  $isMouseDown,
  $isSpaceDown,
  $lastAddedPoint,
  $lastCursorPos,
  $lastMouseDownPos,
  $selectedLayer,
  $shouldInvertBrushSizeScrollDirection,
  $stagePos,
  $stageScale,
  $tool,
  $toolBuffer,
  bboxChanged,
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
import { useTranslation } from 'react-i18next';
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

const selectLayerCount = createSelector(
  selectControlLayersSlice,
  (controlLayers) => controlLayers.present.layers.length
);

const useStageRenderer = (stage: Konva.Stage, container: HTMLDivElement | null, asPreview: boolean) => {
  const dispatch = useAppDispatch();
  const state = useAppSelector((s) => s.controlLayers.present);
  const tool = useStore($tool);
  const lastCursorPos = useStore($lastCursorPos);
  const lastMouseDownPos = useStore($lastMouseDownPos);
  const isMouseDown = useStore($isMouseDown);
  const stageScale = useStore($stageScale);
  const isDrawing = useStore($isDrawing);
  const brushColor = useAppSelector(selectBrushColor);
  const selectedLayer = useAppSelector(selectSelectedLayer);
  const renderers = useMemo(() => (asPreview ? debouncedRenderers : normalRenderers), [asPreview]);
  const dpr = useDevicePixelRatio({ round: false });
  const shouldInvertBrushSizeScrollDirection = useAppSelector((s) => s.canvas.shouldInvertBrushSizeScrollDirection);
  const brushSpacingPx = useMemo(
    () => clamp(state.brushSize / BRUSH_SPACING_PCT, MIN_BRUSH_SPACING_PX, MAX_BRUSH_SPACING_PX),
    [state.brushSize]
  );
  const bbox = useMemo(
    () => ({ x: state.x, y: state.y, width: state.size.width, height: state.size.height }),
    [state.x, state.y, state.size.width, state.size.height]
  );

  useLayoutEffect(() => {
    $brushColor.set(brushColor);
    $brushSize.set(state.brushSize);
    $brushSpacingPx.set(brushSpacingPx);
    $selectedLayer.set(selectedLayer);
    $shouldInvertBrushSizeScrollDirection.set(shouldInvertBrushSizeScrollDirection);
    $genBbox.set(bbox);
  }, [
    brushSpacingPx,
    brushColor,
    selectedLayer,
    shouldInvertBrushSizeScrollDirection,
    state.brushSize,
    state.selectedLayerId,
    state.brushColor,
    bbox,
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
  const onBboxTransformed = useCallback(
    (bbox: IRect) => {
      dispatch(bboxChanged(bbox));
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

    const cleanup = setStageEventHandlers({
      stage,
      $tool,
      $toolBuffer,
      $isDrawing,
      $isMouseDown,
      $lastMouseDownPos,
      $lastCursorPos,
      $lastAddedPoint,
      $stageScale,
      $stagePos,
      $brushSize,
      $brushColor,
      $brushSpacingPx,
      $selectedLayer,
      $shouldInvertBrushSizeScrollDirection,
      $isSpaceDown,
      onBrushSizeChanged,
      onBrushLineAdded,
      onEraserLineAdded,
      onPointAddedToLine,
      onRectShapeAdded,
    });

    return () => {
      log.trace('Removing stage listeners');
      cleanup();
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
    if (!container) {
      return;
    }

    const fitStageToContainer = () => {
      stage.width(container.offsetWidth);
      stage.height(container.offsetHeight);
    };

    const resizeObserver = new ResizeObserver(fitStageToContainer);
    resizeObserver.observe(container);
    fitStageToContainer();

    return () => {
      resizeObserver.disconnect();
    };
  }, [stage, container]);

  useLayoutEffect(() => {
    if (asPreview) {
      // Preview should not display tool
      return;
    }
    log.trace('Rendering tool preview');
    renderers.renderPreviewLayer(
      stage,
      tool,
      brushColor,
      selectedLayer?.type ?? null,
      state.globalMaskLayerOpacity,
      lastCursorPos,
      lastMouseDownPos,
      state.brushSize,
      isDrawing,
      isMouseDown,
      $genBbox,
      onBboxTransformed
    );
    renderImageDimsPreview(stage, bbox, tool);
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
    isMouseDown,
    bbox,
    stageScale,
    onBboxTransformed,
  ]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    renderers.renderLayers(
      stage,
      bbox,
      state.layers,
      state.globalMaskLayerOpacity,
      tool,
      getImageDTO,
      onLayerPosChanged
    );
  }, [stage, state.layers, state.globalMaskLayerOpacity, tool, onLayerPosChanged, renderers, bbox]);

  useLayoutEffect(() => {
    if (asPreview) {
      // Preview should not check for transparency
      return;
    }
    log.trace('Updating bboxes');
    debouncedRenderers.updateBboxes(stage, state.layers, onBboxChanged);
  }, [stage, asPreview, state.layers, onBboxChanged]);

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);
};

type Props = {
  asPreview?: boolean;
};

export const StageComponent = memo(({ asPreview = false }: Props) => {
  const { t } = useTranslation();
  const layerCount = useAppSelector(selectLayerCount);
  const [stage] = useState(
    () =>
      new Konva.Stage({
        id: uuidv4(),
        container: document.createElement('div'),
        listening: !asPreview,
      })
  );
  const [container, setContainer] = useState<HTMLDivElement | null>(null);

  const containerRef = useCallback((el: HTMLDivElement | null) => {
    setContainer(el);
  }, []);

  useStageRenderer(stage, container, asPreview);

  return (
    <Flex position="relative" w="full" h="full">
      <Box
        position="absolute"
        w="full"
        h="full"
        borderRadius="base"
        backgroundImage={TRANSPARENCY_CHECKER_PATTERN}
        backgroundRepeat="repeat"
        opacity={0.2}
      />
      {layerCount === 0 && !asPreview && (
        <Flex position="absolute" w="full" h="full" alignItems="center" justifyContent="center">
          <Heading color="base.200">{t('controlLayers.noLayersAdded')}</Heading>
        </Flex>
      )}

      <Flex
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        ref={containerRef}
        tabIndex={-1}
        borderRadius="base"
        overflow="hidden"
        data-testid="control-layers-canvas"
      />
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';
