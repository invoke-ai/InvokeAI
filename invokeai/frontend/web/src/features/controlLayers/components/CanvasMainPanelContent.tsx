import { ContextMenu, Flex, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import { CanvasContextMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuItems';
import { CanvasDropArea } from 'features/controlLayers/components/CanvasDropArea';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { CanvasHUD } from 'features/controlLayers/components/HUD/CanvasHUD';
import { CanvasSelectedEntityStatusAlert } from 'features/controlLayers/components/HUD/CanvasSelectedEntityStatusAlert';
import { SendingToGalleryAlert } from 'features/controlLayers/components/HUD/CanvasSendingToGalleryAlert';
import { InvokeCanvasComponent } from 'features/controlLayers/components/InvokeCanvasComponent';
import { StagingAreaIsStagingGate } from 'features/controlLayers/components/StagingArea/StagingAreaIsStagingGate';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { CanvasToolbar } from 'features/controlLayers/components/Toolbar/CanvasToolbar';
import { Transform } from 'features/controlLayers/components/Transform/Transform';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { TRANSPARENCY_CHECKERBOARD_PATTERN_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import { GatedImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { memo, useCallback, useRef } from 'react';

export const CanvasMainPanelContent = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const dynamicGrid = useAppSelector(selectDynamicGrid);
  const showHUD = useAppSelector(selectShowHUD);

  const renderMenu = useCallback(() => {
    return (
      <CanvasManagerProviderGate>
        <MenuList>
          <CanvasContextMenuItems />
        </MenuList>
      </CanvasManagerProviderGate>
    );
  }, []);

  useScopeOnFocus('canvas', ref);

  return (
    <Flex
      tabIndex={-1}
      ref={ref}
      borderRadius="base"
      position="relative"
      flexDirection="column"
      height="full"
      width="full"
      gap={2}
      alignItems="center"
      justifyContent="center"
    >
      <CanvasToolbar />
      <ContextMenu<HTMLDivElement> renderMenu={renderMenu}>
        {(ref) => (
          <Flex
            ref={ref}
            position="relative"
            w="full"
            h="full"
            bg={dynamicGrid ? 'base.850' : 'base.900'}
            borderRadius="base"
          >
            {!dynamicGrid && (
              <Flex
                position="absolute"
                borderRadius="base"
                bgImage={TRANSPARENCY_CHECKERBOARD_PATTERN_DATAURL}
                top={0}
                right={0}
                bottom={0}
                left={0}
                opacity={0.1}
              />
            )}
            <InvokeCanvasComponent />
            <CanvasManagerProviderGate>
              {showHUD && (
                <Flex position="absolute" top={1} insetInlineStart={1} pointerEvents="none">
                  <CanvasHUD />
                </Flex>
              )}
              <Flex flexDir="column" position="absolute" top={1} insetInlineEnd={1} pointerEvents="none" gap={2}>
                <CanvasSelectedEntityStatusAlert />
                <SendingToGalleryAlert />
              </Flex>
            </CanvasManagerProviderGate>
          </Flex>
        )}
      </ContextMenu>
      <Flex position="absolute" bottom={4} gap={2} align="center" justify="center">
        <CanvasManagerProviderGate>
          <StagingAreaIsStagingGate>
            <StagingAreaToolbar />
          </StagingAreaIsStagingGate>
        </CanvasManagerProviderGate>
      </Flex>
      <Flex position="absolute" bottom={4}>
        <CanvasManagerProviderGate>
          <Filter />
          <Transform />
        </CanvasManagerProviderGate>
      </Flex>
      <CanvasDropArea />
      <GatedImageViewer />
    </Flex>
  );
});

CanvasMainPanelContent.displayName = 'CanvasMainPanelContent';
