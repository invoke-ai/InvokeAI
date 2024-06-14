import { $alt, $ctrl, $meta, $shift, Box, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { HeadsUpDisplay } from 'features/controlLayers/components/HeadsUpDisplay';
import { TRANSPARENCY_CHECKER_PATTERN } from 'features/controlLayers/konva/constants';
import { setStageEventHandlers } from 'features/controlLayers/konva/events';
import { debouncedRenderers, renderers as normalRenderers } from 'features/controlLayers/konva/renderers/layers';
import { caBboxChanged, caTranslated } from 'features/controlLayers/store/controlAdaptersSlice';
import {
  $bbox,
  $currentFill,
  $isDrawing,
  $isMouseDown,
  $lastAddedPoint,
  $lastCursorPos,
  $lastMouseDownPos,
  $selectedEntity,
  $spaceKey,
  $stageAttrs,
  $toolState,
  bboxChanged,
  brushWidthChanged,
  eraserWidthChanged,
  selectCanvasV2Slice,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/controlLayersSlice';
import {
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerLinePointAdded,
  layerRectAdded,
  layerTranslated,
  selectLayersSlice,
} from 'features/controlLayers/store/layersSlice';
import {
  rgBboxChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgLinePointAdded,
  rgRectAdded,
  rgTranslated,
  selectRegionalGuidanceSlice,
} from 'features/controlLayers/store/regionalGuidanceSlice';
import type {
  BboxChangedArg,
  BrushLineAddedArg,
  CanvasEntity,
  EraserLineAddedArg,
  PointAddedToLineArg,
  PosChangedArg,
  RectShapeAddedArg,
  Tool,
} from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { getImageDTO } from 'services/api/endpoints/images';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';
import { v4 as uuidv4 } from 'uuid';

// This will log warnings when layers > 5 - maybe use `import.meta.env.MODE === 'development'` instead?
Konva.showWarnings = false;

const log = logger('controlLayers');

const selectBrushFill = createSelector(
  selectCanvasV2Slice,
  selectLayersSlice,
  selectRegionalGuidanceSlice,
  (canvas, layers, regionalGuidance) => {
    const rg = regionalGuidance.regions.find((i) => i.id === canvas.selectedEntityIdentifier?.id);

    if (rg) {
      return rgbaColorToString({ ...rg.fill, a: regionalGuidance.opacity });
    }

    return rgbaColorToString(canvas.tool.fill);
  }
);

const useStageRenderer = (stage: Konva.Stage, container: HTMLDivElement | null, asPreview: boolean) => {
  const dispatch = useAppDispatch();
  const canvasV2State = useAppSelector(selectCanvasV2Slice);
  const layersState = useAppSelector((s) => s.layers);
  const controlAdaptersState = useAppSelector((s) => s.controlAdaptersV2);
  const ipAdaptersState = useAppSelector((s) => s.ipAdapters);
  const regionalGuidanceState = useAppSelector((s) => s.regionalGuidance);
  const lastCursorPos = useStore($lastCursorPos);
  const lastMouseDownPos = useStore($lastMouseDownPos);
  const isMouseDown = useStore($isMouseDown);
  const isDrawing = useStore($isDrawing);
  const brushColor = useAppSelector(selectBrushFill);
  const selectedEntity = useMemo(() => {
    const identifier = canvasV2State.selectedEntityIdentifier;
    if (!identifier) {
      return null;
    } else if (identifier.type === 'layer') {
      return layersState.layers.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'control_adapter') {
      return controlAdaptersState.controlAdapters.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'ip_adapter') {
      return ipAdaptersState.ipAdapters.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      return regionalGuidanceState.regions.find((i) => i.id === identifier.id) ?? null;
    } else {
      return null;
    }
  }, [
    canvasV2State.selectedEntityIdentifier,
    controlAdaptersState.controlAdapters,
    ipAdaptersState.ipAdapters,
    layersState.layers,
    regionalGuidanceState.regions,
  ]);

  const currentFill = useMemo(() => {
    if (selectedEntity && selectedEntity.type === 'regional_guidance') {
      return { ...selectedEntity.fill, a: regionalGuidanceState.opacity };
    }
    return canvasV2State.tool.fill;
  }, [canvasV2State.tool.fill, regionalGuidanceState.opacity, selectedEntity]);

  const renderers = useMemo(() => (asPreview ? debouncedRenderers : normalRenderers), [asPreview]);
  const dpr = useDevicePixelRatio({ round: false });

  useLayoutEffect(() => {
    $toolState.set(canvasV2State.tool);
    $selectedEntity.set(selectedEntity);
    $bbox.set(canvasV2State.bbox);
    $currentFill.set(currentFill);
  }, [selectedEntity, canvasV2State.tool, canvasV2State.bbox, currentFill]);

  const onPosChanged = useCallback(
    (arg: PosChangedArg, entityType: CanvasEntity['type']) => {
      if (entityType === 'layer') {
        dispatch(layerTranslated(arg));
      } else if (entityType === 'control_adapter') {
        dispatch(caTranslated(arg));
      } else if (entityType === 'regional_guidance') {
        dispatch(rgTranslated(arg));
      }
    },
    [dispatch]
  );

  const onBboxChanged = useCallback(
    (arg: BboxChangedArg, entityType: CanvasEntity['type']) => {
      if (entityType === 'layer') {
        dispatch(layerBboxChanged(arg));
      } else if (entityType === 'control_adapter') {
        dispatch(caBboxChanged(arg));
      } else if (entityType === 'regional_guidance') {
        dispatch(rgBboxChanged(arg));
      }
    },
    [dispatch]
  );

  const onBrushLineAdded = useCallback(
    (arg: BrushLineAddedArg, entityType: CanvasEntity['type']) => {
      if (entityType === 'layer') {
        dispatch(layerBrushLineAdded(arg));
      } else if (entityType === 'regional_guidance') {
        dispatch(rgBrushLineAdded(arg));
      }
    },
    [dispatch]
  );
  const onEraserLineAdded = useCallback(
    (arg: EraserLineAddedArg, entityType: CanvasEntity['type']) => {
      if (entityType === 'layer') {
        dispatch(layerEraserLineAdded(arg));
      } else if (entityType === 'regional_guidance') {
        dispatch(rgEraserLineAdded(arg));
      }
    },
    [dispatch]
  );
  const onPointAddedToLine = useCallback(
    (arg: PointAddedToLineArg, entityType: CanvasEntity['type']) => {
      if (entityType === 'layer') {
        dispatch(layerLinePointAdded(arg));
      } else if (entityType === 'regional_guidance') {
        dispatch(rgLinePointAdded(arg));
      }
    },
    [dispatch]
  );
  const onRectShapeAdded = useCallback(
    (arg: RectShapeAddedArg, entityType: CanvasEntity['type']) => {
      if (entityType === 'layer') {
        dispatch(layerRectAdded(arg));
      } else if (entityType === 'regional_guidance') {
        dispatch(rgRectAdded(arg));
      }
    },
    [dispatch]
  );
  const onBboxTransformed = useCallback(
    (bbox: IRect) => {
      dispatch(bboxChanged(bbox));
    },
    [dispatch]
  );
  const onBrushWidthChanged = useCallback(
    (width: number) => {
      dispatch(brushWidthChanged(width));
    },
    [dispatch]
  );
  const onEraserWidthChanged = useCallback(
    (width: number) => {
      dispatch(eraserWidthChanged(width));
    },
    [dispatch]
  );
  const setTool = useCallback(
    (tool: Tool) => {
      dispatch(toolChanged(tool));
    },
    [dispatch]
  );
  const setToolBuffer = useCallback(
    (toolBuffer: Tool | null) => {
      dispatch(toolBufferChanged(toolBuffer));
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
      getToolState: $toolState.get,
      setTool,
      setToolBuffer,
      getIsDrawing: $isDrawing.get,
      setIsDrawing: $isDrawing.set,
      getIsMouseDown: $isMouseDown.get,
      setIsMouseDown: $isMouseDown.set,
      getSelectedEntity: $selectedEntity.get,
      getLastAddedPoint: $lastAddedPoint.get,
      setLastAddedPoint: $lastAddedPoint.set,
      getLastCursorPos: $lastCursorPos.get,
      setLastCursorPos: $lastCursorPos.set,
      getLastMouseDownPos: $lastMouseDownPos.get,
      setLastMouseDownPos: $lastMouseDownPos.set,
      getSpaceKey: $spaceKey.get,
      setStageAttrs: $stageAttrs.set,
      onBrushLineAdded,
      onEraserLineAdded,
      onPointAddedToLine,
      onRectShapeAdded,
      onBrushWidthChanged,
      onEraserWidthChanged,
      getCurrentFill: $currentFill.get,
    });

    return () => {
      log.trace('Removing stage listeners');
      cleanup();
    };
  }, [
    asPreview,
    onBrushLineAdded,
    onBrushWidthChanged,
    onEraserLineAdded,
    onPointAddedToLine,
    onRectShapeAdded,
    stage,
    container,
    onEraserWidthChanged,
    setTool,
    setToolBuffer,
  ]);

  useLayoutEffect(() => {
    log.trace('Updating stage dimensions');
    if (!container) {
      return;
    }

    const fitStageToContainer = () => {
      stage.width(container.offsetWidth);
      stage.height(container.offsetHeight);
      $stageAttrs.set({
        x: stage.x(),
        y: stage.y(),
        width: stage.width(),
        height: stage.height(),
        scale: stage.scaleX(),
      });
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
    renderers.renderToolPreview(
      stage,
      canvasV2State.tool,
      currentFill,
      selectedEntity,
      lastCursorPos,
      lastMouseDownPos,
      isDrawing,
      isMouseDown
    );
  }, [
    asPreview,
    brushColor,
    canvasV2State.tool,
    currentFill,
    isDrawing,
    isMouseDown,
    lastCursorPos,
    lastMouseDownPos,
    renderers,
    selectedEntity,
    stage,
  ]);

  useLayoutEffect(() => {
    if (asPreview) {
      // Preview should not display tool
      return;
    }
    log.trace('Rendering bbox preview');
    renderers.renderBboxPreview(
      stage,
      canvasV2State.bbox,
      canvasV2State.tool.selected,
      $bbox.get,
      onBboxTransformed,
      $shift.get,
      $ctrl.get,
      $meta.get,
      $alt.get
    );
  }, [asPreview, canvasV2State.bbox, canvasV2State.tool.selected, onBboxTransformed, renderers, stage]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    renderers.renderLayers(
      stage,
      layersState.layers,
      controlAdaptersState.controlAdapters,
      regionalGuidanceState.regions,
      regionalGuidanceState.opacity,
      canvasV2State.tool.selected,
      selectedEntity,
      getImageDTO,
      onPosChanged
    );
  }, [
    stage,
    renderers,
    layersState.layers,
    controlAdaptersState.controlAdapters,
    regionalGuidanceState.regions,
    regionalGuidanceState.opacity,
    onPosChanged,
    canvasV2State.tool.selected,
    selectedEntity,
  ]);

  // useLayoutEffect(() => {
  //   if (asPreview) {
  //     // Preview should not check for transparency
  //     return;
  //   }
  //   log.trace('Updating bboxes');
  //   debouncedRenderers.updateBboxes(stage, state.layers, onBboxChanged);
  // }, [stage, asPreview, state.layers, onBboxChanged]);

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);

  useEffect(
    () => () => {
      stage.destroy();
    },
    [stage]
  );
};

type Props = {
  asPreview?: boolean;
};

export const StageComponent = memo(({ asPreview = false }: Props) => {
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
      {!asPreview && <NoLayersFallback />}
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
      {!asPreview && (
        <Flex position="absolute" top={0} insetInlineStart={0}>
          <HeadsUpDisplay />
        </Flex>
      )}
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';

const NoLayersFallback = () => {
  const { t } = useTranslation();
  const layerCount = useAppSelector((s) => s.layers.layers.length);
  if (layerCount) {
    return null;
  }

  return (
    <Flex position="absolute" w="full" h="full" alignItems="center" justifyContent="center">
      <Heading color="base.200">{t('controlLayers.noLayersAdded')}</Heading>
    </Flex>
  );
};
