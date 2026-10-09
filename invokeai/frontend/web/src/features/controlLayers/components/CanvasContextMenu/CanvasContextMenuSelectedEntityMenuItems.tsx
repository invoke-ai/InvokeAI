import { MenuGroup } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { ControlLayerMenuItems } from 'features/controlLayers/components/ControlLayer/ControlLayerMenuItems';
import { InpaintMaskMenuItems } from 'features/controlLayers/components/InpaintMask/InpaintMaskMenuItems';
import { RasterLayerMenuItems } from 'features/controlLayers/components/RasterLayer/RasterLayerMenuItems';
import { IPAdapterMenuItems } from 'features/controlLayers/components/RefImage/IPAdapterMenuItems';
import { RegionalGuidanceMenuItems } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceMenuItems';
import { VectorLayerMenuItems } from 'features/controlLayers/components/VectorLayer/VectorLayerMenuItems';
import { VectorPathMenuItems } from 'features/controlLayers/components/VectorLayer/VectorPathMenuItems';
import { CanvasEntityStateGate } from 'features/controlLayers/contexts/CanvasEntityStateGate';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  EntityIdentifierContext,
  useEntityIdentifierContext,
} from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityTypeString } from 'features/controlLayers/hooks/useEntityTypeString';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { PropsWithChildren } from 'react';
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
  if (entityIdentifier.type === 'vector_layer') {
    return <VectorLayerMenuItems />;
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

const CanvasContextMenuSelectedEntityMenuGroup = memo((props: PropsWithChildren) => {
  const entityIdentifier = useEntityIdentifierContext();
  const entityTypeTitle = useEntityTypeString(entityIdentifier.type);

  return <MenuGroup title={entityTypeTitle}>{props.children}</MenuGroup>;
});

CanvasContextMenuSelectedEntityMenuGroup.displayName = 'CanvasContextMenuSelectedEntityMenuGroup';

export const CanvasContextMenuSelectedEntityMenuItems = memo(
  ({ showPathActions = false }: { showPathActions?: boolean }) => {
    const canvasManager = useCanvasManager();
    const editSession = useStore(canvasManager.tool.tools.path.$editSession);
    const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
    const entityIdentifier = editSession?.entityIdentifier ?? selectedEntityIdentifier;

    if (!entityIdentifier) {
      return null;
    }

    return (
      <EntityIdentifierContext.Provider value={entityIdentifier}>
        <CanvasEntityStateGate entityIdentifier={entityIdentifier}>
          {showPathActions && entityIdentifier.type === 'vector_layer' && <VectorPathMenuItems />}
          <CanvasContextMenuSelectedEntityMenuGroup>
            <CanvasContextMenuSelectedEntityMenuItemsContent />
          </CanvasContextMenuSelectedEntityMenuGroup>
        </CanvasEntityStateGate>
      </EntityIdentifierContext.Provider>
    );
  }
);

CanvasContextMenuSelectedEntityMenuItems.displayName = 'CanvasContextMenuSelectedEntityMenuItems';
