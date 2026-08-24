import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasEngineHandle } from '@workbench/widgets/canvas/useCanvasEngine';
import type { Dispatch } from 'react';

import { HStack, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { reorderSelectionWithinGroupsByKind, type LayerReorderKind } from '@workbench/canvasLayerOps';
import {
  ArrowDownIcon,
  ArrowUpIcon,
  ChevronsDownIcon,
  ChevronsUpIcon,
  EyeIcon,
  EyeOffIcon,
  LockIcon,
  LockOpenIcon,
  Trash2Icon,
} from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { deleteLayersActions } from './layerMultiSelectionModel';

type MultiSelectionEngine = Pick<CanvasEngineHandle, 'layers'>;
const BULK_TOOLTIP_POSITIONING = { placement: 'top' } as const;

interface LayerMultiSelectionActionsProps {
  dispatch: Dispatch<CanvasProjectMutation>;
  editingLocked: boolean;
  engine: MultiSelectionEngine | null;
  layers: readonly CanvasLayerContract[];
  selectedIds: readonly string[];
  selectedLayerId: string | null;
}

const applyBulkStructural = (
  engine: MultiSelectionEngine | null,
  dispatch: Dispatch<CanvasProjectMutation>,
  label: string,
  forward: CanvasProjectMutation,
  inverse: CanvasProjectMutation
): void => {
  if (engine) {
    engine.layers.commitStructural(label, forward, inverse);
  } else {
    dispatch(forward);
  }
};

export const LayerMultiSelectionActions = ({
  dispatch,
  editingLocked,
  engine,
  layers,
  selectedIds,
  selectedLayerId,
}: LayerMultiSelectionActionsProps) => {
  const { t } = useTranslation();
  const selected = useMemo(() => {
    const ids = new Set(selectedIds);
    return layers.filter((layer) => ids.has(layer.id));
  }, [layers, selectedIds]);
  const allEnabled = selected.every((layer) => layer.isEnabled);
  const allLocked = selected.every((layer) => layer.isLocked);
  const hasLocked = selected.some((layer) => layer.isLocked);

  const reorder = useCallback(
    (kind: LayerReorderKind, label: string) => {
      const next = reorderSelectionWithinGroupsByKind(layers, selectedIds, kind);
      if (!next) {
        return;
      }
      applyBulkStructural(
        engine,
        dispatch,
        label,
        { orderedIds: next, type: 'reorderCanvasLayers' },
        { orderedIds: layers.map((layer) => layer.id), type: 'reorderCanvasLayers' }
      );
    },
    [dispatch, engine, layers, selectedIds]
  );

  const moveToFront = useCallback(
    () => reorder('front', t('widgets.layers.actions.moveSelectedToFront')),
    [reorder, t]
  );
  const moveForward = useCallback(
    () => reorder('forward', t('widgets.layers.actions.moveSelectedForward')),
    [reorder, t]
  );
  const moveBackward = useCallback(
    () => reorder('backward', t('widgets.layers.actions.moveSelectedBackward')),
    [reorder, t]
  );
  const moveToBack = useCallback(() => reorder('back', t('widgets.layers.actions.moveSelectedToBack')), [reorder, t]);

  const toggleEnabled = useCallback(() => {
    const isEnabled = !allEnabled;
    applyBulkStructural(
      engine,
      dispatch,
      t(isEnabled ? 'widgets.layers.actions.enableSelected' : 'widgets.layers.actions.disableSelected'),
      { type: 'setCanvasLayersEnabled', updates: selected.map((layer) => ({ id: layer.id, isEnabled })) },
      {
        type: 'setCanvasLayersEnabled',
        updates: selected.map((layer) => ({ id: layer.id, isEnabled: layer.isEnabled })),
      }
    );
  }, [allEnabled, dispatch, engine, selected, t]);

  const toggleLocked = useCallback(() => {
    const isLocked = !allLocked;
    applyBulkStructural(
      engine,
      dispatch,
      t(isLocked ? 'widgets.layers.actions.lockSelected' : 'widgets.layers.actions.unlockSelected'),
      {
        enabledUpdates: [],
        lockedUpdates: selected.map((layer) => ({ id: layer.id, isLocked })),
        type: 'applyCanvasLayerStackMutation',
      },
      {
        enabledUpdates: [],
        lockedUpdates: selected.map((layer) => ({ id: layer.id, isLocked: layer.isLocked })),
        type: 'applyCanvasLayerStackMutation',
      }
    );
  }, [allLocked, dispatch, engine, selected, t]);

  const deleteSelected = useCallback(() => {
    const actions = deleteLayersActions(layers, selectedIds, selectedLayerId);
    if (actions) {
      applyBulkStructural(
        engine,
        dispatch,
        t('widgets.layers.actions.deleteSelected'),
        actions.forward,
        actions.inverse
      );
    }
  }, [dispatch, engine, layers, selectedIds, selectedLayerId, t]);

  return (
    <HStack
      aria-label={t('widgets.layers.actions.selectedCount', { count: selected.length })}
      gap="0.5"
      minH="10"
      px="2"
      role="toolbar"
    >
      <Text color="fg.muted" flex="1" fontSize="2xs" fontWeight="700">
        {t('widgets.layers.actions.selectedCount', { count: selected.length })}
      </Text>
      <BulkActionButton
        disabled={editingLocked}
        icon={ChevronsUpIcon}
        label={t('widgets.layers.actions.moveSelectedToFront')}
        onClick={moveToFront}
      />
      <BulkActionButton
        disabled={editingLocked}
        icon={ArrowUpIcon}
        label={t('widgets.layers.actions.moveSelectedForward')}
        onClick={moveForward}
      />
      <BulkActionButton
        disabled={editingLocked}
        icon={ArrowDownIcon}
        label={t('widgets.layers.actions.moveSelectedBackward')}
        onClick={moveBackward}
      />
      <BulkActionButton
        disabled={editingLocked}
        icon={ChevronsDownIcon}
        label={t('widgets.layers.actions.moveSelectedToBack')}
        onClick={moveToBack}
      />
      <BulkActionButton
        disabled={editingLocked}
        icon={allEnabled ? EyeOffIcon : EyeIcon}
        label={t(allEnabled ? 'widgets.layers.actions.disableSelected' : 'widgets.layers.actions.enableSelected')}
        onClick={toggleEnabled}
      />
      <BulkActionButton
        disabled={editingLocked}
        icon={allLocked ? LockOpenIcon : LockIcon}
        label={t(allLocked ? 'widgets.layers.actions.unlockSelected' : 'widgets.layers.actions.lockSelected')}
        onClick={toggleLocked}
      />
      <BulkActionButton
        colorPalette="red"
        disabled={editingLocked || hasLocked}
        icon={Trash2Icon}
        label={t('widgets.layers.actions.deleteSelected')}
        onClick={deleteSelected}
      />
    </HStack>
  );
};

export default LayerMultiSelectionActions;

const BulkActionButton = ({
  colorPalette,
  disabled,
  icon: Icon,
  label,
  onClick,
}: {
  colorPalette?: string;
  disabled: boolean;
  icon: typeof ArrowUpIcon;
  label: string;
  onClick: () => void;
}) => (
  <Tooltip content={label} positioning={BULK_TOOLTIP_POSITIONING}>
    <IconButton
      aria-label={label}
      colorPalette={colorPalette}
      disabled={disabled}
      size="xs"
      variant="ghost"
      onClick={onClick}
    >
      <Icon />
    </IconButton>
  </Tooltip>
);
