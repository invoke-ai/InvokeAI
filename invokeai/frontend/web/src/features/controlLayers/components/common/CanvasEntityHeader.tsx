import type { FlexProps } from '@invoke-ai/ui-library';
import { ContextMenu, Flex, MenuList } from '@invoke-ai/ui-library';
import { ControlLayerMenuItems } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItems';
import { InpaintMaskMenuItems } from 'features/controlLayers/components/InpaintMask/InpaintMaskMenuItems';
import { IPAdapterMenuItems } from 'features/controlLayers/components/IPAdapter/IPAdapterMenuItems';
import { RasterLayerMenuItems } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItems';
import { RegionalGuidanceMenuItems } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItems';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { memo, useCallback } from 'react';
import { assert } from 'tsafe';

export const CanvasEntityHeader = memo(({ children, ...rest }: FlexProps) => {
  const entityIdentifier = useEntityIdentifierContext();
  const renderMenu = useCallback(() => {
    if (entityIdentifier.type === 'regional_guidance') {
      return (
        <MenuList>
          <RegionalGuidanceMenuItems />
        </MenuList>
      );
    }

    if (entityIdentifier.type === 'inpaint_mask') {
      return (
        <MenuList>
          <InpaintMaskMenuItems />
        </MenuList>
      );
    }

    if (entityIdentifier.type === 'raster_layer') {
      return (
        <MenuList>
          <RasterLayerMenuItems />
        </MenuList>
      );
    }

    if (entityIdentifier.type === 'control_layer') {
      return (
        <MenuList>
          <ControlLayerMenuItems />
        </MenuList>
      );
    }

    if (entityIdentifier.type === 'reference_image') {
      return (
        <MenuList>
          <IPAdapterMenuItems />
        </MenuList>
      );
    }

    assert(false, 'Unhandled entity type');
  }, [entityIdentifier]);

  return (
    <ContextMenu renderMenu={renderMenu}>
      {(ref) => (
        <Flex ref={ref} gap={2} alignItems="center" p={2} {...rest}>
          {children}
        </Flex>
      )}
    </ContextMenu>
  );
});

CanvasEntityHeader.displayName = 'CanvasEntityHeader';
