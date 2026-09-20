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
import { useTranslation } from 'react-i18next';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const CanvasContextMenuSelectedEntityMenuItemsContent = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const canvasManager = useCanvasManager();
  const editSession = useStore(canvasManager.tool.tools.path.$editSession);

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
    if (
      editSession?.entityIdentifier.id === entityIdentifier.id &&
      editSession.entityIdentifier.type === entityIdentifier.type
    ) {
      return <VectorPathMenuItems />;
    }
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
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const canvasManager = useCanvasManager();
  const editSession = useStore(canvasManager.tool.tools.path.$editSession);
  const entityTypeTitle = useEntityTypeString(entityIdentifier.type);
  const isEditingPath =
    entityIdentifier.type === 'vector_layer' &&
    editSession?.entityIdentifier.id === entityIdentifier.id &&
    editSession.entityIdentifier.type === entityIdentifier.type;
  const title = isEditingPath ? t('controlLayers.vectorEdit.pathTitle') : entityTypeTitle;

  return <MenuGroup title={title}>{props.children}</MenuGroup>;
});

CanvasContextMenuSelectedEntityMenuGroup.displayName = 'CanvasContextMenuSelectedEntityMenuGroup';

export const CanvasContextMenuSelectedEntityMenuItems = memo(() => {
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
        <CanvasContextMenuSelectedEntityMenuGroup>
          <CanvasContextMenuSelectedEntityMenuItemsContent />
        </CanvasContextMenuSelectedEntityMenuGroup>
      </CanvasEntityStateGate>
    </EntityIdentifierContext.Provider>
  );
});

CanvasContextMenuSelectedEntityMenuItems.displayName = 'CanvasContextMenuSelectedEntityMenuItems';
