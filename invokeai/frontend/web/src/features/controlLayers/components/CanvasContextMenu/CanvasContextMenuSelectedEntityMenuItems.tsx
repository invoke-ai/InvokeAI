import { useAppSelector } from 'app/store/storeHooks';
import { ControlLayerMenuItems } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItems';
import { InpaintMaskMenuItems } from 'features/controlLayers/components/InpaintMask/InpaintMaskMenuItems';
import { IPAdapterMenuItems } from 'features/controlLayers/components/IPAdapter/IPAdapterMenuItems';
import { RasterLayerMenuItems } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItems';
import { RegionalGuidanceMenuItems } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItems';
import {
  EntityIdentifierContext,
  useEntityIdentifierContext,
} from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const CanvasContextMenuSelectedEntityMenuItemsContent = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();

  if (entityIdentifier.type === 'raster_layer') {
    return <RasterLayerMenuItems />;
  }
  if (entityIdentifier.type === 'control_layer') {
    return <ControlLayerMenuItems />;
  }
  if (entityIdentifier.type === 'inpaint_mask') {
    return <InpaintMaskMenuItems />;
  }
  if (entityIdentifier.type === 'regional_guidance') {
    return <RegionalGuidanceMenuItems />;
  }
  if (entityIdentifier.type === 'reference_image') {
    return <IPAdapterMenuItems />;
  }

  assert<Equals<typeof entityIdentifier.type, never>>(false);
});

CanvasContextMenuSelectedEntityMenuItemsContent.displayName = 'CanvasContextMenuSelectedEntityMenuItemsContent';

export const CanvasContextMenuSelectedEntityMenuItems = memo(() => {
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);

  if (!selectedEntityIdentifier) {
    return null;
  }

  return (
    <EntityIdentifierContext.Provider value={selectedEntityIdentifier}>
      <CanvasContextMenuSelectedEntityMenuItemsContent />
    </EntityIdentifierContext.Provider>
  );
});

CanvasContextMenuSelectedEntityMenuItems.displayName = 'CanvasContextMenuSelectedEntityMenuItems';
