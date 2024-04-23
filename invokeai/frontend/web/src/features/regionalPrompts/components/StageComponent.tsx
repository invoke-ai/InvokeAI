import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useMouseEvents } from 'features/regionalPrompts/hooks/mouseEventHooks';
import {
  $cursorPosition,
  $lastMouseDownPos,
  $tool,
  isVectorMaskLayer,
  layerBboxChanged,
  layerSelected,
  layerTranslated,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { renderers } from 'features/regionalPrompts/util/renderers';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import type { MutableRefObject } from 'react';
import { memo, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { assert } from 'tsafe';

// This will log warnings when layers > 5 - maybe use `import.meta.env.MODE === 'development'` instead?
Konva.showWarnings = false;

const log = logger('regionalPrompts');

const selectSelectedLayerColor = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  const layer = regionalPrompts.present.layers.find((l) => l.id === regionalPrompts.present.selectedLayerId);
  if (!layer) {
    return null;
  }
  assert(isVectorMaskLayer(layer), `Layer ${regionalPrompts.present.selectedLayerId} is not an RP layer`);
  return layer.previewColor;
});

const useStageRenderer = (
  stageRef: MutableRefObject<Konva.Stage>,
  container: HTMLDivElement | null,
  wrapper: HTMLDivElement | null,
  asPreview: boolean
) => {
  const dispatch = useAppDispatch();
  const width = useAppSelector((s) => s.generation.width);
  const height = useAppSelector((s) => s.generation.height);
  const state = useAppSelector((s) => s.regionalPrompts.present);
  const tool = useStore($tool);
  const { onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave, onMouseWheel } = useMouseEvents();
  const cursorPosition = useStore($cursorPosition);
  const lastMouseDownPos = useStore($lastMouseDownPos);
  const selectedLayerIdColor = useAppSelector(selectSelectedLayerColor);

  const renderLayers = useMemo(() => (asPreview ? renderers.layersDebounced : renderers.layers), [asPreview]);
  const renderToolPreview = useMemo(
    () => (asPreview ? renderers.toolPreviewDebounced : renderers.toolPreview),
    [asPreview]
  );
  const renderBbox = useMemo(() => (asPreview ? renderers.bboxDebounced : renderers.bbox), [asPreview]);
  const renderBackground = useMemo(
    () => (asPreview ? renderers.backgroundDebounced : renderers.background),
    [asPreview]
  );

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

  const onBboxMouseDown = useCallback(
    (layerId: string) => {
      dispatch(layerSelected(layerId));
    },
    [dispatch]
  );

  useLayoutEffect(() => {
    log.trace('Initializing stage');
    if (!container) {
      return;
    }
    const stage = stageRef.current.container(container);
    return () => {
      log.trace('Cleaning up stage');
      stage.destroy();
    };
  }, [container, stageRef]);

  useLayoutEffect(() => {
    log.trace('Adding stage listeners');
    if (asPreview) {
      return;
    }
    stageRef.current.on('mousedown', onMouseDown);
    stageRef.current.on('mouseup', onMouseUp);
    stageRef.current.on('mousemove', onMouseMove);
    stageRef.current.on('mouseenter', onMouseEnter);
    stageRef.current.on('mouseleave', onMouseLeave);
    stageRef.current.on('wheel', onMouseWheel);
    const stage = stageRef.current;

    return () => {
      log.trace('Cleaning up stage listeners');
      stage.off('mousedown', onMouseDown);
      stage.off('mouseup', onMouseUp);
      stage.off('mousemove', onMouseMove);
      stage.off('mouseenter', onMouseEnter);
      stage.off('mouseleave', onMouseLeave);
      stage.off('wheel', onMouseWheel);
    };
  }, [stageRef, asPreview, onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave, onMouseWheel]);

  useLayoutEffect(() => {
    log.trace('Updating stage dimensions');
    if (!wrapper) {
      return;
    }

    const stage = stageRef.current;

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
  }, [stageRef, width, height, wrapper]);

  useLayoutEffect(() => {
    log.trace('Rendering brush preview');
    if (asPreview) {
      return;
    }
    renderToolPreview(
      stageRef.current,
      tool,
      selectedLayerIdColor,
      state.globalMaskLayerOpacity,
      cursorPosition,
      lastMouseDownPos,
      state.brushSize
    );
  }, [
    asPreview,
    stageRef,
    tool,
    selectedLayerIdColor,
    state.globalMaskLayerOpacity,
    cursorPosition,
    lastMouseDownPos,
    state.brushSize,
    renderToolPreview,
  ]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    renderLayers(stageRef.current, state.layers, state.globalMaskLayerOpacity, tool, onLayerPosChanged);
  }, [stageRef, state.layers, state.globalMaskLayerOpacity, tool, onLayerPosChanged, renderLayers]);

  useLayoutEffect(() => {
    log.trace('Rendering bbox');
    if (asPreview) {
      return;
    }
    renderBbox(stageRef.current, state.layers, state.selectedLayerId, tool, onBboxChanged, onBboxMouseDown);
  }, [stageRef, asPreview, state.layers, state.selectedLayerId, tool, onBboxChanged, onBboxMouseDown, renderBbox]);

  useLayoutEffect(() => {
    log.trace('Rendering background');
    if (asPreview) {
      return;
    }
    renderBackground(stageRef.current, width, height);
  }, [stageRef, asPreview, width, height, renderBackground]);
};

type Props = {
  asPreview?: boolean;
};

export const StageComponent = memo(({ asPreview = false }: Props) => {
  const stageRef = useRef<Konva.Stage>(
    new Konva.Stage({
      container: document.createElement('div'), // We will overwrite this shortly...
    })
  );
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const [wrapper, setWrapper] = useState<HTMLDivElement | null>(null);

  const containerRef = useCallback((el: HTMLDivElement | null) => {
    setContainer(el);
  }, []);

  const wrapperRef = useCallback((el: HTMLDivElement | null) => {
    setWrapper(el);
  }, []);

  useStageRenderer(stageRef, container, wrapper, asPreview);

  return (
    <Flex overflow="hidden" w="full" h="full">
      <Flex ref={wrapperRef} w="full" h="full" alignItems="center" justifyContent="center">
        <Flex ref={containerRef} tabIndex={-1} bg="base.850" />
      </Flex>
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';
