import type { DragEndEvent } from '@dnd-kit/core';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { LucideIcon } from 'lucide-react';
import type { Dispatch } from 'react';

import { Collapsible, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { closestCenter, DndContext, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { restrictToParentElement, restrictToVerticalAxis } from '@dnd-kit/modifiers';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { canMergeVisibleRasters } from '@workbench/canvas-engine/document/mergeVisible';
import { IconButton, toaster, Tooltip } from '@workbench/components/ui';
import { useCanvasDocumentEditingLocked } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectName } from '@workbench/WorkbenchContext';
import { ChevronDownIcon, EyeIcon, EyeOffIcon, FileDownIcon, LayersIcon, PlusIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerGroupKey } from './layerGroups';

import { groupAddItemId } from './addLayerMenu';
import { canExportRasterPsd, getGroupActions, isGroupAllVisible, planGroupVisibilityToggle } from './layerGroupActions';
import { reorderWithinGroup } from './layerGroups';
import { LayerListItem } from './LayerListItem';
import { applyStructural } from './layerOps';
import { useAddLayer } from './useAddLayer';

const DND_MODIFIERS = [restrictToVerticalAxis, restrictToParentElement];
const POINTER_SENSOR_OPTIONS = { activationConstraint: { distance: 6 } } as const;

interface LayerGroupSectionProps {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  groupKey: LayerGroupKey;
  groupLayers: readonly CanvasLayerContract[];
  isCollapsed: boolean;
  layers: readonly CanvasLayerContract[];
  onToggleCollapse: (groupKey: LayerGroupKey) => void;
  selectedLayerId: string | null;
}

/**
 * One collapsible type-group section: a header (chevron + name + count + a
 * right-aligned action cluster) and, when expanded, the group's rows in a
 * self-contained `DndContext`. Scoping DnD to the group means a drag can never see
 * (or drop onto) another group's rows — cross-group moves are structurally
 * impossible. Collapsing a group unmounts only its own rows, so drag-and-drop in
 * the other (still-expanded) groups is unaffected.
 */
export const LayerGroupSection = ({
  dispatch,
  engine,
  groupKey,
  groupLayers,
  isCollapsed,
  layers,
  onToggleCollapse,
  selectedLayerId,
}: LayerGroupSectionProps) => {
  const { t } = useTranslation();
  const editingLocked = useCanvasDocumentEditingLocked(engine);

  const sensors = useSensors(useSensor(PointerSensor, POINTER_SENSOR_OPTIONS));

  const globalIndexById = useMemo(() => {
    const map = new Map<string, number>();
    layers.forEach((layer, index) => map.set(layer.id, index));
    return map;
  }, [layers]);

  const groupIds = useMemo(() => groupLayers.map((layer) => layer.id), [groupLayers]);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      if (editingLocked) {
        return;
      }
      const { active, over } = event;
      if (!over) {
        return;
      }
      const next = reorderWithinGroup(layers, String(active.id), String(over.id));
      if (!next) {
        return;
      }
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.actions.reorder'),
        { orderedIds: next, type: 'reorderCanvasLayers' },
        { orderedIds: layers.map((layer) => layer.id), type: 'reorderCanvasLayers' }
      );
    },
    [dispatch, editingLocked, engine, layers, t]
  );

  const handleToggleCollapse = useCallback(() => onToggleCollapse(groupKey), [groupKey, onToggleCollapse]);

  return (
    <Stack gap="1">
      <HStack gap="1" px="1">
        <IconButton
          aria-label={t(isCollapsed ? 'widgets.layers.groupActions.expand' : 'widgets.layers.groupActions.collapse')}
          color="fg.subtle"
          size="2xs"
          variant="ghost"
          onClick={handleToggleCollapse}
        >
          <Icon
            as={ChevronDownIcon}
            boxSize="3.5"
            transform={isCollapsed ? 'rotate(-90deg)' : undefined}
            transitionDuration="fast"
            transitionProperty="transform"
          />
        </IconButton>
        <Text
          color="fg.subtle"
          cursor="pointer"
          flex="1"
          fontSize="2xs"
          fontWeight="700"
          textTransform="uppercase"
          truncate
          userSelect="none"
          onClick={handleToggleCollapse}
        >
          {t(`widgets.layers.groups.${groupKey}`)} ({groupLayers.length})
        </Text>
        <GroupActions
          dispatch={dispatch}
          engine={engine}
          editingLocked={editingLocked}
          groupKey={groupKey}
          groupLayers={groupLayers}
          layers={layers}
        />
      </HStack>
      {/* unmountOnExit: collapsing must UNMOUNT the rows (not just hide them) so an
          open per-row properties popover — and any control-layer filter preview it
          hosts — is torn down with them (Task 38/39 cleanup contract). */}
      <Collapsible.Root lazyMount open={!isCollapsed} unmountOnExit>
        <Collapsible.Content>
          <DndContext
            collisionDetection={closestCenter}
            modifiers={DND_MODIFIERS}
            sensors={sensors}
            onDragEnd={handleDragEnd}
          >
            <SortableContext items={groupIds} strategy={verticalListSortingStrategy}>
              <Stack gap="0.5">
                {groupLayers.map((layer) => (
                  <LayerListItem
                    key={layer.id}
                    dispatch={dispatch}
                    engine={engine}
                    index={globalIndexById.get(layer.id) ?? 0}
                    isSelected={layer.id === selectedLayerId}
                    layer={layer}
                    layers={layers}
                  />
                ))}
              </Stack>
            </SortableContext>
          </DndContext>
        </Collapsible.Content>
      </Collapsible.Root>
    </Stack>
  );
};

/**
 * The right-aligned group-header action cluster. The set of actions is data
 * (`getGroupActions`), so Task 44's PSD export button drops in by extending that
 * array + this switch — no new prop wiring.
 */
const GroupActions = ({
  dispatch,
  editingLocked,
  engine,
  groupKey,
  groupLayers,
  layers,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  editingLocked: boolean;
  engine: CanvasEngine | null;
  groupKey: LayerGroupKey;
  groupLayers: readonly CanvasLayerContract[];
  layers: readonly CanvasLayerContract[];
}) => {
  const { t } = useTranslation();
  const addLayer = useAddLayer();
  const projectName = useActiveProjectName();
  const allVisible = isGroupAllVisible(groupLayers);
  // Enablement uses the SAME planner the engine op executes (`planMergeVisibleRuns`
  // over the GLOBAL array — run-splitting depends on interleaved non-participants),
  // so the button is enabled exactly when clicking it will merge something.
  const canMerge = !editingLocked && !!engine && groupKey === 'raster' && canMergeVisibleRasters(layers);
  const canExport = !!engine && groupKey === 'raster' && canExportRasterPsd(layers);

  const handleNew = useCallback(() => addLayer(groupAddItemId(groupKey)), [addLayer, groupKey]);

  const handleToggleVisibility = useCallback(() => {
    const { forward, inverse } = planGroupVisibilityToggle(groupLayers);
    applyStructural(
      engine,
      dispatch,
      t('widgets.layers.groupActions.toggleVisibility'),
      { type: 'setCanvasLayersEnabled', updates: forward },
      { type: 'setCanvasLayersEnabled', updates: inverse }
    );
  }, [dispatch, engine, groupLayers, t]);

  const handleMergeVisible = useCallback(() => {
    // The engine folds ALL visible mergeable rasters (reorder + merge per step)
    // and pre-flights every participant before touching pixels; 'not-ready'
    // means a cache is still decoding — nothing was merged, so tell the user
    // instead of silently half-working. Engine pixel work: not undoable,
    // mirroring the per-row "merge down".
    if (engine?.mergeVisibleRasterLayers() === 'not-ready') {
      toaster.create({ title: t('widgets.layers.groupActions.mergeNotReady'), type: 'warning' });
    }
  }, [engine, t]);

  const handleExportPsd = useCallback(() => {
    if (!engine) {
      return;
    }
    // Read-only: no dispatch, no history. The engine lazily loads `ag-psd`,
    // bakes the raster layers, and triggers the download. Surface the refusal
    // cases (still-loading caches / oversized bounds / nothing to export).
    void engine.exportRasterLayersToPsd(projectName).then((result) => {
      if (result === 'not-ready') {
        toaster.create({ title: t('widgets.layers.groupActions.exportNotReady'), type: 'warning' });
      } else if (result === 'too-large') {
        toaster.create({ title: t('widgets.layers.groupActions.exportTooLarge'), type: 'warning' });
      } else if (result === 'nothing') {
        toaster.create({ title: t('widgets.layers.groupActions.exportNothing'), type: 'warning' });
      }
    });
  }, [engine, projectName, t]);

  const actions = getGroupActions(groupKey);

  return (
    <HStack gap="0.5">
      {actions.map((action) => {
        switch (action) {
          case 'mergeVisible':
            return (
              <GroupActionButton
                key={action}
                disabled={!canMerge}
                icon={LayersIcon}
                label={t('widgets.layers.groupActions.mergeVisible')}
                onClick={handleMergeVisible}
              />
            );
          case 'exportPsd':
            return (
              <GroupActionButton
                key={action}
                disabled={editingLocked || !canExport}
                icon={FileDownIcon}
                label={t('widgets.layers.groupActions.exportPsd')}
                onClick={handleExportPsd}
              />
            );
          case 'toggleVisibility':
            return (
              <GroupActionButton
                key={action}
                disabled={editingLocked}
                icon={allVisible ? EyeIcon : EyeOffIcon}
                label={t(allVisible ? 'widgets.layers.groupActions.hideAll' : 'widgets.layers.groupActions.showAll')}
                onClick={handleToggleVisibility}
              />
            );
          case 'new':
            return (
              <GroupActionButton
                key={action}
                disabled={editingLocked}
                icon={PlusIcon}
                label={t('widgets.layers.groupActions.new')}
                onClick={handleNew}
              />
            );
        }
      })}
    </HStack>
  );
};

const GroupActionButton = ({
  disabled,
  icon,
  label,
  onClick,
}: {
  disabled?: boolean;
  icon: LucideIcon;
  label: string;
  onClick: () => void;
}) => (
  <Tooltip content={label}>
    <IconButton aria-label={label} color="fg.subtle" disabled={disabled} size="2xs" variant="ghost" onClick={onClick}>
      <Icon as={icon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);
