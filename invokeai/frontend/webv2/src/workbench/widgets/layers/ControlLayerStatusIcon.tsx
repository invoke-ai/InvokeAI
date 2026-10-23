import type { ArchitectureCapabilitiesSnapshot } from '@features/generation/runtime';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { Box, Icon, Spinner, VisuallyHidden } from '@chakra-ui/react';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { useModelsSelector } from '@features/models';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  createControlLayerCheckContext,
  describeControlLayer,
  describeControlLayerReason,
} from '@workbench/controlLayerChecks';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useControlDetectionStatus } from '@workbench/widgets/canvas/engineStoreHooks';
import { TriangleAlertIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { useSelectedMainModel } from './useSelectedMainModel';

const selectCapabilitiesStatus = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.status;
const selectControlModelMemory = (preferences: WorkbenchPreferences) => preferences.controlModelMemory;

/**
 * A control layer row's status: a spinner while its map is being detected, else a warning when the layer needs
 * attention, judged as its Properties panel judges it. `contributing` is the effective enablement; a layer gated by
 * a group is not validated.
 */
export const ControlLayerStatusIcon = ({
  layer,
  contributing = layer.isEnabled,
  engine,
}: {
  contributing?: boolean;
  /** The canvas engine whose detection status the row reflects. */
  engine: object | null;
  layer: CanvasControlLayerContract;
}) => {
  const { t } = useTranslation();
  const detection = useControlDetectionStatus(engine, layer.id);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const mainModel = useSelectedMainModel();
  const remembered = useWorkbenchPreferenceSelector(selectControlModelMemory);
  const capabilitiesStatus = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    selectCapabilitiesStatus
  );
  // Read inside the store's selector: whether a kind is supported comes from the capability table, and a call
  // memoised on the layer and model would keep its first answer after the table loads.
  const reason = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      () =>
        contributing
          ? describeControlLayer(layer, {
              ...createControlLayerCheckContext(mainModel?.base ?? null, remembered),
              mainModel,
              models,
            }).reason
          : null,
      [contributing, layer, mainModel, models, remembered]
    )
  );

  if (detection.status === 'processing') {
    const label = t('widgets.layers.control.detecting', {
      detector: t(`widgets.layers.control.filters.${detection.detector}`, detection.detector),
    });
    return (
      <Tooltip content={label}>
        <Box alignItems="center" display="inline-flex" flexShrink={0}>
          <Spinner boxSize="3" color="fg.muted" />
          <VisuallyHidden>{label}</VisuallyHidden>
        </Box>
      </Tooltip>
    );
  }

  // Distinguish loading/failed capabilities from unsupported adapters; failures should point to retry.
  if (!reason || (reason === 'capabilities_unavailable' && capabilitiesStatus !== 'error')) {
    return null;
  }

  const message =
    reason === 'capabilities_unavailable'
      ? `${t('widgets.layers.control.capabilitiesLoadFailed')} ${t('widgets.layers.control.capabilitiesRetryHint')}`
      : describeControlLayerReason(t, reason);

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
