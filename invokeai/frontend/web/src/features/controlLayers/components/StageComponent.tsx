import { $alt, $ctrl, $meta, $shift, Box, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { HeadsUpDisplay } from 'features/controlLayers/components/HeadsUpDisplay';
import { TRANSPARENCY_CHECKER_PATTERN } from 'features/controlLayers/konva/constants';
import { setStageEventHandlers } from 'features/controlLayers/konva/events';
import { debouncedRenderers, renderers as normalRenderers } from 'features/controlLayers/konva/renderers/layers';
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
  caBboxChanged,
  caTranslated,
  eraserWidthChanged,
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerLinePointAdded,
  layerRectAdded,
  layerTranslated,
  rgBboxChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgLinePointAdded,
  rgRectAdded,
  rgTranslated,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectEntityCount } from 'features/controlLayers/store/selectors';
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

const useStageRenderer = (stage: Konva.Stage, container: HTMLDivElement | null, asPreview: boolean) => {
  const dispatch = useAppDispatch();
  const controlAdapters = useAppSelector((s) => s.canvasV2.controlAdapters);
  const ipAdapters = useAppSelector((s) => s.canvasV2.ipAdapters);
  const layers = useAppSelector((s) => s.canvasV2.layers);
  const regions = useAppSelector((s) => s.canvasV2.regions);
  const tool = useAppSelector((s) => s.canvasV2.tool);
  const selectedEntityIdentifier = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier);
  const maskFillOpacity = useAppSelector((s) => s.canvasV2.maskFillOpacity);
  const bbox = useAppSelector((s) => s.canvasV2.bbox);
  const lastCursorPos = useStore($lastCursorPos);
  const lastMouseDownPos = useStore($lastMouseDownPos);
  const isMouseDown = useStore($isMouseDown);
  const isDrawing = useStore($isDrawing);
  const selectedEntity = useMemo(() => {
    const identifier = selectedEntityIdentifier;
    if (!identifier) {
      return null;
    } else if (identifier.type === 'layer') {
      return layers.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'control_adapter') {
      return controlAdapters.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'ip_adapter') {
      return ipAdapters.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      return regions.find((i) => i.id === identifier.id) ?? null;
    } else {
      return null;
    }
  }, [controlAdapters, ipAdapters, layers, regions, selectedEntityIdentifier]);

  const currentFill = useMemo(() => {
    if (selectedEntity && selectedEntity.type === 'regional_guidance') {
      return { ...selectedEntity.fill, a: maskFillOpacity };
    }
    return tool.fill;
  }, [maskFillOpacity, selectedEntity, tool.fill]);

  const renderers = useMemo(() => (asPreview ? debouncedRenderers : normalRenderers), [asPreview]);
  const dpr = useDevicePixelRatio({ round: false });

  useLayoutEffect(() => {
    $toolState.set(tool);
    $selectedEntity.set(selectedEntity);
    $bbox.set(bbox);
    $currentFill.set(currentFill);
  }, [selectedEntity, tool, bbox, currentFill]);

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
      tool,
      currentFill,
      selectedEntity,
      lastCursorPos,
      lastMouseDownPos,
      isDrawing,
      isMouseDown
    );
  }, [
    asPreview,
    currentFill,
    isDrawing,
    isMouseDown,
    lastCursorPos,
    lastMouseDownPos,
    renderers,
    selectedEntity,
    stage,
    tool,
  ]);

  useLayoutEffect(() => {
    if (asPreview) {
      // Preview should not display tool
      return;
    }
    log.trace('Rendering bbox preview');
    renderers.renderBboxPreview(
      stage,
      bbox,
      tool.selected,
      $bbox.get,
      onBboxTransformed,
      $shift.get,
      $ctrl.get,
      $meta.get,
      $alt.get
    );
  }, [asPreview, bbox, onBboxTransformed, renderers, stage, tool.selected]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    renderers.renderLayers(
      stage,
      layers,
      controlAdapters,
      regions,
      maskFillOpacity,
      tool.selected,
      selectedEntity,
      getImageDTO,
      onPosChanged
    );
  }, [
    controlAdapters,
    layers,
    maskFillOpacity,
    onPosChanged,
    regions,
    renderers,
    selectedEntity,
    stage,
    tool.selected,
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
      {!asPreview && <NoEntitiesFallback />}
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

const NoEntitiesFallback = () => {
  const { t } = useTranslation();
  const entityCount = useAppSelector(selectEntityCount);

  if (entityCount) {
    return null;
  }

  return (
    <Flex position="absolute" w="full" h="full" alignItems="center" justifyContent="center">
      <Heading color="base.200">{t('controlLayers.noLayersAdded')}</Heading>
    </Flex>
  );
};
