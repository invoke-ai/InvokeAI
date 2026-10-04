import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type { CanvasLayerContract } from '@workbench/canvas-engine/api';

import { Icon } from '@chakra-ui/react';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { useModelsSelector } from '@features/models';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Tooltip } from '@platform/ui';
import { getControlLayerAttentionReason } from '@workbench/controlLayerChecks';
import { TriangleAlertIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { useSelectedMainModel } from './useSelectedMainModel';

const selectCapabilitiesStatus = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.status;

/** `contributing` is the effective enablement; a layer gated by a group is not validated. */
export const ControlLayerWarningIcon = ({
  layer,
  contributing = layer.isEnabled,
}: {
  layer: CanvasLayerContract;
  contributing?: boolean;
}) => {
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const mainModel = useSelectedMainModel();
  const capabilitiesStatus = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesStatus
  );
  // Read inside the store's selector: whether the adapter kind is supported comes from the capability
  // table, and a call memoised on the layer and model would keep its first answer after the table loads.
  const reason = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      () =>
        layer.type === 'control' && contributing && mainModel
          ? getControlLayerAttentionReason(layer, mainModel.base, models)
          : null,
      [contributing, layer, mainModel, models]
    )
  );

  // Distinguish loading/failed capabilities from unsupported adapters; failures should point to retry.
  if (!reason || (reason === 'capabilities_unavailable' && capabilitiesStatus !== 'error')) {
    return null;
  }

  const message =
    reason === 'capabilities_unavailable'
      ? `${t('widgets.layers.control.capabilitiesLoadFailed')} ${t('widgets.layers.control.capabilitiesRetryHint')}`
      : t(`widgets.layers.control.validation.${reason}`);

  return (
    <Tooltip content={message}>
      <Icon
        aria-label={`${t('widgets.layers.control.needsAttention')}: ${message}`}
        as={TriangleAlertIcon}
        boxSize="3"
        color="fg.warning"
        flexShrink={0}
      />
    </Tooltip>
  );
};
