import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasControlLayerContract, CanvasRasterLayerContractV2 } from '@workbench/types';

import { Box } from '@chakra-ui/react';
import { Button, Tooltip } from '@workbench/components/ui';
import { useNotify } from '@workbench/useNotify';
import { useLayerThumbnailVersion } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import {
  getLayerFilterLaunchDisabledReason,
  getLayerFilterLaunchReasonKey,
  type LayerFilterLaunchRejectedResult,
  runLayerFilterOperation,
} from './layerPropertiesOperation';

interface LayerFilterOperationButtonProps {
  engine: CanvasEngine | null;
  layer: CanvasRasterLayerContractV2 | CanvasControlLayerContract;
  onOperationStarted(): void;
}

export const LayerFilterOperationButton = ({ engine, layer, onOperationStarted }: LayerFilterOperationButtonProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  useLayerThumbnailVersion(engine, layer.id);

  const disabledReason = getLayerFilterLaunchDisabledReason({
    hasEngine: engine !== null,
    hasExportableContent: engine?.hasExportableLayerContent(layer.id) ?? false,
    isEnabled: layer.isEnabled,
    isLocked: layer.isLocked,
  });

  const onOperationRejected = useCallback(
    (result: LayerFilterLaunchRejectedResult) => {
      notify.error(t('widgets.layers.actions.actionFailed'), t(getLayerFilterLaunchReasonKey(result)));
    },
    [notify, t]
  );

  const start = useCallback(() => {
    if (!engine || disabledReason) {
      return;
    }
    runLayerFilterOperation(() => engine.startFilterOperation(layer.id), onOperationStarted, onOperationRejected);
  }, [disabledReason, engine, layer.id, onOperationRejected, onOperationStarted]);

  return (
    <Tooltip
      content={disabledReason ? t(getLayerFilterLaunchReasonKey(disabledReason)) : ''}
      disabled={disabledReason === null}
    >
      <Box w="full">
        <Button disabled={disabledReason !== null} size="xs" variant="outline" w="full" onClick={start}>
          {t('widgets.layers.control.filter')}
        </Button>
      </Box>
    </Tooltip>
  );
};
