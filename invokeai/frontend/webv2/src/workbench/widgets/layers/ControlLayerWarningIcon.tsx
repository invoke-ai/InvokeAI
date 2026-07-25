import type { CanvasLayerContract } from '@workbench/canvas-engine/api';

import { Icon } from '@chakra-ui/react';
import { useModelsSelector } from '@features/models';
import { Tooltip } from '@platform/ui';
import { getControlLayerAttentionReason } from '@workbench/controlLayerChecks';
import { TriangleAlertIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useSelectedMainModel } from './useSelectedMainModel';

/**
 * Quiet per-row indicator for a control layer that would be rejected at
 * generation time (no model, incompatible adapter, …). Renders nothing for
 * other layer types and healthy control layers.
 */
export const ControlLayerWarningIcon = ({ layer }: { layer: CanvasLayerContract }) => {
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const mainModel = useSelectedMainModel();

  if (layer.type !== 'control' || !layer.isEnabled || !mainModel) {
    return null;
  }
  const reason = getControlLayerAttentionReason(layer, mainModel.base, models);
  if (!reason) {
    return null;
  }

  return (
    <Tooltip content={t(`widgets.layers.control.validation.${reason}`)}>
      <Icon
        aria-label={t('widgets.layers.control.needsAttention')}
        as={TriangleAlertIcon}
        boxSize="3"
        color="fg.warning"
        flexShrink={0}
      />
    </Tooltip>
  );
};
