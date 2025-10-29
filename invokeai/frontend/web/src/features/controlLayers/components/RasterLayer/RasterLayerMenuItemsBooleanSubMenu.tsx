import { MenuItem } from '@invoke-ai/ui-library';
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
  const subMenu = useSubMenu();
  const canvasManager = useCanvasManager();
  const isBusy = useCanvasIsBusy();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const entityIdentifierBelowThisOne = useEntityIdentifierBelowThisOne(entityIdentifier as CanvasEntityIdentifier);
  const { t } = useTranslation();

  const perform = useCallback(
    async (op: GlobalCompositeOperation) => {
      if (!entityIdentifierBelowThisOne) return;
      // Temporarily set composite op on the selected layer to drive the merge algorithm
      dispatch(
        rasterLayerGlobalCompositeOperationChanged({ entityIdentifier, globalCompositeOperation: op })
      );
      try {
        await canvasManager.compositor.mergeByEntityIdentifiers(
          [entityIdentifierBelowThisOne, entityIdentifier],
          true
        );
      } finally {
        // No need to reset - layers are deleted on success; but in case of failure, clear the op
        dispatch(
          rasterLayerGlobalCompositeOperationChanged({ entityIdentifier, globalCompositeOperation: undefined })
        );
      }
    },
    [canvasManager.compositor, dispatch, entityIdentifier, entityIdentifierBelowThisOne]
  );

  const onIntersection = useCallback(() => perform('source-in'), [perform]);
  const onCutout = useCallback(() => perform('destination-in'), [perform]);
  const onCutAway = useCallback(() => perform('source-out'), [perform]);
  const onExclude = useCallback(() => perform('xor'), [perform]);

  return (
    <>
      <MenuItem onMouseEnter={subMenu.onOpen} onMouseLeave={subMenu.onClose} isDisabled={isBusy || !entityIdentifierBelowThisOne}>
        <SubMenuButtonContent label={t('controlLayers.booleanOps.label')} />
      </MenuItem>
      {subMenu.isOpen && (
        <>
          <MenuItem onClick={onIntersection} isDisabled={isBusy}>
            {t('controlLayers.booleanOps.intersection')}
          </MenuItem>
          <MenuItem onClick={onCutout} isDisabled={isBusy}>
            {t('controlLayers.booleanOps.cutout')}
          </MenuItem>
          <MenuItem onClick={onCutAway} isDisabled={isBusy}>
            {t('controlLayers.booleanOps.cutAway')}
          </MenuItem>
          <MenuItem onClick={onExclude} isDisabled={isBusy}>
            {t('controlLayers.booleanOps.exclude')}
          </MenuItem>
        </>
      )}
    </>
  );
});

RasterLayerMenuItemsBooleanSubMenu.displayName = 'RasterLayerMenuItemsBooleanSubMenu';