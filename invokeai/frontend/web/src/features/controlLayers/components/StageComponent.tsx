import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { $socket } from 'app/hooks/useSocketIO';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { HeadsUpDisplay } from 'features/controlLayers/components/HeadsUpDisplay';
import { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { TRANSPARENCY_CHECKER_PATTERN } from 'features/controlLayers/konva/constants';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import Konva from 'konva';
import { memo, useCallback, useEffect, useLayoutEffect, useState } from 'react';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';
import { v4 as uuidv4 } from 'uuid';

const log = logger('canvas');

const showHud = false;

// This will log warnings when layers > 5 - maybe use `import.meta.env.MODE === 'development'` instead?
Konva.showWarnings = false;

const useStageRenderer = (stage: Konva.Stage, container: HTMLDivElement | null, asPreview: boolean) => {
  const store = useAppStore();
  const socket = useStore($socket);
  const dpr = useDevicePixelRatio({ round: false });

  useLayoutEffect(() => {
    log.debug('Initializing renderer');
    if (!container) {
      // Nothing to clean up
      log.debug('No stage container, skipping initialization');
      return () => {};
    }

    if (!socket) {
      log.debug('Socket not connected, skipping initialization');
      return () => {};
    }

    const manager = new CanvasManager(stage, container, store, socket);
    manager.initialize();
    return manager.destroy;
  }, [asPreview, container, socket, stage, store]);

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);
};

type Props = {
  asPreview?: boolean;
};

const selectDynamicGrid = createSelector(selectCanvasSettingsSlice, (canvasSettings) => canvasSettings.dynamicGrid);

export const StageComponent = memo(({ asPreview = false }: Props) => {
  const dynamicGrid = useAppSelector(selectDynamicGrid);

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

  useEffect(
    () => () => {
      stage.destroy();
    },
    [stage]
  );

  return (
    <Flex position="relative" w="full" h="full" bg={dynamicGrid ? 'base.850' : 'base.900'}>
      {!dynamicGrid && (
        <Flex
          position="absolute"
          bgImage={TRANSPARENCY_CHECKER_PATTERN}
          top={0}
          right={0}
          bottom={0}
          left={0}
          opacity={0.1}
        />
      )}
      <Flex
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        ref={containerRef}
        borderRadius="base"
        border={1}
        borderStyle="solid"
        borderColor="base.700"
        overflow="hidden"
        data-testid="control-layers-canvas"
      />
      {!asPreview && (
        <Flex position="absolute" top={0} insetInlineStart={0} pointerEvents="none">
          {showHud && <HeadsUpDisplay />}
        </Flex>
      )}
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';
