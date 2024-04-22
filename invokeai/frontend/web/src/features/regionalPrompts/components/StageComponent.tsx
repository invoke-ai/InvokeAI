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
import { renderBackground, renderBbox, renderLayers, renderToolPreview } from 'features/regionalPrompts/util/renderers';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { atom } from 'nanostores';
import { memo, useCallback, useLayoutEffect } from 'react';
import { assert } from 'tsafe';

// This will log warnings when layers > 5 - maybe use `import.meta.env.MODE === 'development'` instead?
Konva.showWarnings = false;

const log = logger('regionalPrompts');
const $stage = atom<Konva.Stage | null>(null);
const selectSelectedLayerColor = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  const layer = regionalPrompts.present.layers.find((l) => l.id === regionalPrompts.present.selectedLayerId);
  if (!layer) {
    return null;
  }
  assert(isVectorMaskLayer(layer), `Layer ${regionalPrompts.present.selectedLayerId} is not an RP layer`);
  return layer.previewColor;
});

const useStageRenderer = (container: HTMLDivElement | null, wrapper: HTMLDivElement | null, asPreview: boolean) => {
  const dispatch = useAppDispatch();
  const width = useAppSelector((s) => s.generation.width);
  const height = useAppSelector((s) => s.generation.height);
  const state = useAppSelector((s) => s.regionalPrompts.present);
  const stage = useStore($stage);
  const tool = useStore($tool);
  const { onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave } = useMouseEvents();
  const cursorPosition = useStore($cursorPosition);
  const lastMouseDownPos = useStore($lastMouseDownPos);
  const selectedLayerIdColor = useAppSelector(selectSelectedLayerColor);

  const onLayerPosChanged = useCallback(
    (layerId: string, x: number, y: number) => {
      if (asPreview) {
        dispatch(layerTranslated({ layerId, x, y }));
      }
    },
    [dispatch, asPreview]
  );

  const onBboxChanged = useCallback(
    (layerId: string, bbox: IRect | null) => {
      if (asPreview) {
        dispatch(layerBboxChanged({ layerId, bbox }));
      }
    },
    [dispatch, asPreview]
  );

  const onBboxMouseDown = useCallback(
    (layerId: string) => {
      if (asPreview) {
        dispatch(layerSelected(layerId));
      }
    },
    [dispatch, asPreview]
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
    if (!stage || asPreview) {
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
  }, [stage, asPreview, onMouseDown, onMouseUp, onMouseMove, onMouseEnter, onMouseLeave]);

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
    log.trace('Rendering brush preview');
    if (!stage || asPreview) {
      return;
    }
    renderToolPreview(
      stage,
      tool,
      selectedLayerIdColor,
      state.globalMaskLayerOpacity,
      cursorPosition,
      lastMouseDownPos,
      state.brushSize
    );
  }, [
    asPreview,
    stage,
    tool,
    selectedLayerIdColor,
    state.globalMaskLayerOpacity,
    cursorPosition,
    lastMouseDownPos,
    state.brushSize,
  ]);

  useLayoutEffect(() => {
    log.trace('Rendering layers');
    if (!stage) {
      return;
    }
    renderLayers(stage, state.layers, state.globalMaskLayerOpacity, tool, onLayerPosChanged);
  }, [stage, state.layers, state.globalMaskLayerOpacity, tool, onLayerPosChanged]);

  useLayoutEffect(() => {
    log.trace('Rendering bbox');
    if (!stage || asPreview) {
      return;
    }
    renderBbox(stage, state.layers, state.selectedLayerId, tool, onBboxChanged, onBboxMouseDown);
  }, [stage, asPreview, state.layers, state.selectedLayerId, tool, onBboxChanged, onBboxMouseDown]);

  useLayoutEffect(() => {
    log.trace('Rendering background');
    if (!stage || asPreview) {
      return;
    }
    renderBackground(stage, width, height);
  }, [stage, asPreview, width, height]);
};

const $container = atom<HTMLDivElement | null>(null);
const containerRef = (el: HTMLDivElement | null) => {
  $container.set(el);
};
const $wrapper = atom<HTMLDivElement | null>(null);
const wrapperRef = (el: HTMLDivElement | null) => {
  $wrapper.set(el);
};

type Props = {
  asPreview?: boolean;
};

export const StageComponent = memo(({ asPreview = false }: Props) => {
  const container = useStore($container);
  const wrapper = useStore($wrapper);
  useStageRenderer(container, wrapper, asPreview);
  return (
    <Flex overflow="hidden" w="full" h="full">
      <Flex ref={wrapperRef} w="full" h="full" alignItems="center" justifyContent="center">
        <Flex
          ref={containerRef}
          tabIndex={-1}
          bg="base.850"
          p={2}
          borderRadius="base"
          borderWidth={1}
          w="min-content"
          h="min-content"
          minW={64}
          minH={64}
        />
      </Flex>
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';
