import { Flex } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { $isDebugging } from 'app/store/nanostores/isDebugging';
import { useAppStore } from 'app/store/storeHooks';
import { HeadsUpDisplay } from 'features/controlLayers/components/HeadsUpDisplay';
import { CanvasManager, setCanvasManager } from 'features/controlLayers/konva/CanvasManager';
import Konva from 'konva';
import { memo, useCallback, useEffect, useLayoutEffect, useState } from 'react';
import { useDevicePixelRatio } from 'use-device-pixel-ratio';
import { v4 as uuidv4 } from 'uuid';

const log = logger('konva');

// This will log warnings when layers > 5 - maybe use `import.meta.env.MODE === 'development'` instead?
Konva.showWarnings = false;

const useStageRenderer = (stage: Konva.Stage, container: HTMLDivElement | null, asPreview: boolean) => {
  const store = useAppStore();
  const dpr = useDevicePixelRatio({ round: false });

  useLayoutEffect(() => {
    /**
     * Logs a message to the console if debugging is enabled.
     */
    const logIfDebugging = (message: string) => {
      if ($isDebugging.get()) {
        log.debug(message);
      }
    };

    logIfDebugging('Initializing renderer');
    if (!container) {
      // Nothing to clean up
      logIfDebugging('No stage container, skipping initialization');
      return () => {};
    }

    const manager = new CanvasManager(stage, container, store, logIfDebugging);
    setCanvasManager(manager);
    const cleanup = manager.initialize();
    return cleanup;
  }, [asPreview, container, stage, store]);

  useLayoutEffect(() => {
    Konva.pixelRatio = dpr;
  }, [dpr]);
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

  useEffect(
    () => () => {
      stage.destroy();
    },
    [stage]
  );

  return (
    <Flex position="relative" w="full" h="full">
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
          <HeadsUpDisplay />
        </Flex>
      )}
    </Flex>
  );
});

StageComponent.displayName = 'StageComponent';
