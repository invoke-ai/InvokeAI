import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIdentifierBelowThisOne } from 'features/controlLayers/hooks/useNextRenderableEntityIdentifier';
import { rasterLayerGlobalCompositeOperationChanged } from 'features/controlLayers/store/canvasSlice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';

export const RasterLayerMenuItemsBooleanSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const entityIdentifierBelowThisOne = useEntityIdentifierBelowThisOne(entityIdentifier as CanvasEntityIdentifier);

  const perform = useCallback(
    async (op: GlobalCompositeOperation) => {
      if (!entityIdentifierBelowThisOne) return;
      dispatch(rasterLayerGlobalCompositeOperationChanged({ entityIdentifier, globalCompositeOperation: op }));
      try {
        await canvasManager.compositor.mergeByEntityIdentifiers(
          [entityIdentifierBelowThisOne, entityIdentifier],
          true
        );
      } finally {
        dispatch(rasterLayerGlobalCompositeOperationChanged({ entityIdentifier, globalCompositeOperation: undefined }));
      }
    },
    [canvasManager.compositor, dispatch, entityIdentifier, entityIdentifierBelowThisOne]
  );

  const onIntersection = useCallback(() => perform('source-in'), [perform]);
  const onCutout = useCallback(() => perform('destination-in'), [perform]);
  const onCutAway = useCallback(() => perform('source-out'), [perform]);
  const onExclude = useCallback(() => perform('xor'), [perform]);

  const disabled = isBusy || !entityIdentifierBelowThisOne;

  return (
    <MenuItem {...subMenu.parentMenuItemProps} isDisabled={disabled}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.booleanOps.label')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem onClick={onIntersection} isDisabled={disabled}>
            {t('controlLayers.booleanOps.intersection')}
          </MenuItem>
          <MenuItem onClick={onCutout} isDisabled={disabled}>
            {t('controlLayers.booleanOps.cutout')}
          </MenuItem>
          <MenuItem onClick={onCutAway} isDisabled={disabled}>
            {t('controlLayers.booleanOps.cutAway')}
          </MenuItem>
          <MenuItem onClick={onExclude} isDisabled={disabled}>
            {t('controlLayers.booleanOps.exclude')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

RasterLayerMenuItemsBooleanSubMenu.displayName = 'RasterLayerMenuItemsBooleanSubMenu';