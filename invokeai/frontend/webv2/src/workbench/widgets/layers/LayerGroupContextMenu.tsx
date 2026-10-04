import type {
  CanvasAdjustmentEntry,
  CanvasColorLabel,
  CanvasGroupContract,
  LayerStackKind,
  LayerStackMoveKind,
} from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { LucideIcon } from 'lucide-react';
import type { ComponentProps } from 'react';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { MenuActionItem, MenuContent, RenameDialog } from '@platform/ui';
import { collectSubtree, getDocumentIndex, isOverlayStack } from '@workbench/canvas-engine/api';
import { publishLayerPanelSelection, useLayerPanelState } from '@workbench/layerPanelState';
import { useNotify } from '@workbench/useNotify';
import { reportLayerOperation, usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import {
  ArrowDownIcon,
  ArrowDownToLineIcon,
  ArrowUpIcon,
  ArrowUpToLineIcon,
  ChevronRightIcon,
  CircleIcon,
  CircleOffIcon,
  CopyIcon,
  EyeIcon,
  EyeOffIcon,
  FolderPlusIcon,
  LockIcon,
  LockOpenIcon,
  PaletteIcon,
  PencilIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
  UngroupIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { COLOR_LABEL_ITEMS } from './colorLabels';
import { ADJUSTMENT_ADD_ITEMS } from './layerContextActions';
import { canGroupSelection, groupLayers, ungroupLayers } from './layerGroupCommands';
import { createIdentityAdjustment } from './layerOps';

const SUBMENU_POSITIONING = { placement: 'right-start' } as const;

const noop = (): void => undefined;

export type LayerGroupContextMenuEngine = Pick<CanvasEngineHandle, 'document' | 'layers' | 'projectId'>;

type MenuPositioning = ComponentProps<typeof Menu.Root>['positioning'];

const ARRANGE: readonly { icon: LucideIcon; key: string; kind: LayerStackMoveKind }[] = [
  { icon: ArrowUpToLineIcon, key: 'moveToFront', kind: 'front' },
  { icon: ArrowUpIcon, key: 'moveForward', kind: 'forward' },
  { icon: ArrowDownIcon, key: 'moveBackward', kind: 'backward' },
  { icon: ArrowDownToLineIcon, key: 'moveToBack', kind: 'back' },
];

interface LayerGroupContextMenuProps {
  /** The viewport box the menu opens at: a row's menu button or a right-click point. */
  anchor: { x: number; y: number; width: number; height: number };
  editingLocked: boolean;
  engine: LayerGroupContextMenuEngine | null;
  group: CanvasGroupContract;
  stack: LayerStackKind;
  onClose: () => void;
}

/** The group menu the panel host opens for one group at a time: naming, grouping, arrangement, and state edits. */
export const LayerGroupContextMenu = ({
  anchor,
  editingLocked,
  engine,
  group,
  onClose,
  stack,
}: LayerGroupContextMenuProps) => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const notify = useNotify();
  const projectId = useActiveProjectId();
  const document = useActiveProjectSelector((project) => project.canvas.document);
  const { selectedIds } = useLayerPanelState(projectId, document.selectedLayerId);
  const [menuOpen, setMenuOpen] = useState(true);
  const [renaming, setRenaming] = useState(false);
  // Choosing Rename closes the menu; the host's surface must outlive it until the rename dialog has animated out.
  const renameChosen = useRef(false);
  // Every request brings a new anchor. One for this same group while its rename animates out keeps this instance
  // mounted, so it must reopen the menu here.
  const [requestAnchor, setRequestAnchor] = useState(anchor);
  if (anchor !== requestAnchor) {
    setRequestAnchor(anchor);
    setMenuOpen(true);
  }

  const selection = useCallback(
    (): readonly string[] => (selectedIds.includes(group.id) ? selectedIds : [group.id]),
    [group.id, selectedIds]
  );
  // What a locked ancestor or a locked descendant takes off the table, so no item leads to a refusal.
  const entry = getDocumentIndex(document).byId.get(group.id);
  const frozen = entry?.ancestorsLocked ?? false;
  const holdsLocked =
    frozen || group.isLocked || (entry ? collectSubtree(entry.node).some((node) => node.isLocked) : false);
  const groupable = canGroupSelection(engine?.document.model() ?? null, selection());
  const positioning = useMemo<MenuPositioning>(
    () => ({ getAnchorRect: () => anchor, placement: 'bottom-start' }),
    [anchor]
  );
  const handleOpenChange = useCallback(
    (details: { open: boolean }) => {
      if (details.open) {
        return;
      }
      setMenuOpen(false);
      if (!renameChosen.current) {
        onClose();
      }
    },
    [onClose]
  );

  const patch = useCallback(
    (label: string, forward: Partial<Pick<CanvasGroupContract, 'name' | 'isEnabled' | 'isLocked'>>) =>
      commitPrepared(label, (model) => model.prepare({ id: group.id, patch: forward, type: 'patch' })),
    [commitPrepared, group.id]
  );

  const handleRename = useCallback(
    (name: string) => {
      patch(t('widgets.layers.actions.rename'), { name });
    },
    [patch, t]
  );
  const openRename = useCallback(() => {
    renameChosen.current = true;
    setRenaming(true);
  }, []);
  const closeRename = useCallback(() => setRenaming(false), []);
  // A menu reopened while the rename animated out now owns the surface and closes it itself.
  const handleRenameExitComplete = useCallback(() => {
    renameChosen.current = false;
    if (!menuOpen) {
      onClose();
    }
  }, [menuOpen, onClose]);
  const handleToggleEnabled = useCallback(
    () => patch(t('widgets.layers.actions.toggleVisibility'), { isEnabled: !group.isEnabled }),
    [group.isEnabled, patch, t]
  );
  const handleToggleLock = useCallback(
    () => patch(t('widgets.layers.actions.toggleLock'), { isLocked: !group.isLocked }),
    [group.isLocked, patch, t]
  );
  const handleToggleHidden = useCallback(
    () =>
      commitPrepared(t('widgets.layers.actions.toggleHidden'), (model) =>
        model.prepare({ type: 'set-hidden', updates: [{ id: group.id, isHidden: group.isHidden !== true }] })
      ),
    [commitPrepared, group.id, group.isHidden, t]
  );
  const handleGroup = useCallback(
    () => groupLayers(engine, projectId, selection(), t('widgets.layers.actions.group')),
    [engine, projectId, selection, t]
  );
  const handleUngroup = useCallback(
    () => ungroupLayers(engine, selection(), t('widgets.layers.actions.ungroup')),
    [engine, selection, t]
  );
  const handleDuplicate = useCallback(async () => {
    if (!engine) {
      return;
    }
    try {
      const result = await engine.layers.duplicateLayers(selection());
      if (result.status === 'duplicated') {
        publishLayerPanelSelection({ primaryId: result.selectedLayerId, projectId, selectedIds: result.duplicateIds });
        return;
      }
      if (result.status === 'busy') {
        return;
      }
      if (result.status !== 'nothing') {
        reportLayerOperation(result.status, notify.error, t);
        return;
      }
    } catch {
      // The engine leaves the document unchanged on a rejected transaction.
    }
    notify.error(t('widgets.layers.actions.actionFailed'), t('widgets.layers.actions.copyFailed'));
  }, [engine, notify, projectId, selection, t]);
  const handleDelete = useCallback(
    () =>
      commitPrepared(t('widgets.layers.actions.delete'), (model) =>
        model.prepare({ ids: selection(), type: 'remove' })
      ),
    [commitPrepared, selection, t]
  );
  const handleArrange = useCallback(
    (kind: LayerStackMoveKind, label: string) => () =>
      commitPrepared(label, (model) => model.prepare({ ids: selection(), kind, type: 'move' })),
    [commitPrepared, selection]
  );

  const locked = editingLocked;
  const hideable = isOverlayStack(stack);
  const adjustable = stack === 'raster';

  const handleAddAdjustment = useCallback(
    (type: CanvasAdjustmentEntry['type']) => () => {
      const entry = createIdentityAdjustment(type);
      const before = group.adjustments ?? [];
      commitPrepared(t('widgets.layers.menu.addAdjustment'), (model) =>
        model.prepare({
          before: { adjustments: [...before], layerType: 'group' },
          config: { adjustments: [...before, entry], layerType: 'group' },
          id: group.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, group.adjustments, group.id, t]
  );

  const handleSetColorLabel = useCallback(
    (label: CanvasColorLabel | null) => () =>
      commitPrepared(t('widgets.layers.menu.colorLabel'), (model) =>
        model.prepare({ id: group.id, patch: { colorLabel: label ?? undefined }, type: 'patch' })
      ),
    [commitPrepared, group.id, t]
  );

  const items = (
    <MenuContent minW="13rem" py="1">
      {ARRANGE.map((entry) => (
        <MenuActionItem
          key={entry.kind}
          disabled={locked || frozen}
          icon={entry.icon}
          label={t(`widgets.layers.actions.${entry.key}`)}
          value={entry.kind}
          onSelect={handleArrange(entry.kind, t(`widgets.layers.actions.${entry.key}`))}
        />
      ))}
      <Menu.Separator borderColor="border.subtle" />
      <MenuActionItem
        disabled={locked}
        icon={PencilIcon}
        label={t('widgets.layers.actions.rename')}
        value="rename"
        onSelect={openRename}
      />
      <MenuActionItem
        disabled={locked || !engine || frozen}
        icon={CopyIcon}
        label={t('widgets.layers.actions.duplicate')}
        value="duplicate"
        onSelect={handleDuplicate}
      />
      <MenuActionItem
        disabled={locked || !groupable}
        icon={FolderPlusIcon}
        label={t('widgets.layers.actions.group')}
        value="group"
        onSelect={handleGroup}
      />
      <MenuActionItem
        disabled={locked || frozen || group.isLocked}
        icon={UngroupIcon}
        label={t('widgets.layers.actions.ungroup')}
        value="ungroup"
        onSelect={handleUngroup}
      />
      {adjustable ? (
        locked || frozen || group.isLocked ? (
          <MenuActionItem
            disabled
            icon={SlidersHorizontalIcon}
            label={t('widgets.layers.menu.addAdjustment')}
            value="add-adjustment"
            onSelect={noop}
          />
        ) : (
          <Menu.Root positioning={SUBMENU_POSITIONING}>
            <Menu.TriggerItem aria-label={t('widgets.layers.menu.addAdjustment')}>
              <HStack gap="2" minW="0" w="full">
                <Icon as={SlidersHorizontalIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
                <Text flex="1" fontSize="md">
                  {t('widgets.layers.menu.addAdjustment')}
                </Text>
                <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
              </HStack>
            </Menu.TriggerItem>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="13rem" py="1">
                  {ADJUSTMENT_ADD_ITEMS.map((item) => (
                    <MenuActionItem
                      key={item.type}
                      icon={item.icon}
                      label={t(item.labelKey)}
                      value={`add-${item.type}`}
                      onSelect={handleAddAdjustment(item.type)}
                    />
                  ))}
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        )
      ) : null}
      <Menu.Separator borderColor="border.subtle" />
      <MenuActionItem
        disabled={locked}
        icon={group.isEnabled ? CircleOffIcon : CircleIcon}
        label={t(group.isEnabled ? 'widgets.layers.actions.disableGroup' : 'widgets.layers.actions.enableGroup')}
        value="enabled"
        onSelect={handleToggleEnabled}
      />
      {hideable ? (
        <MenuActionItem
          disabled={locked}
          icon={group.isHidden ? EyeIcon : EyeOffIcon}
          label={t('widgets.layers.actions.toggleHidden')}
          value="hidden"
          onSelect={handleToggleHidden}
        />
      ) : null}
      <MenuActionItem
        disabled={locked}
        icon={group.isLocked ? LockOpenIcon : LockIcon}
        label={t(group.isLocked ? 'widgets.layers.actions.unlock' : 'widgets.layers.actions.lock')}
        value="lock"
        onSelect={handleToggleLock}
      />
      {!locked ? (
        <Menu.Root positioning={SUBMENU_POSITIONING}>
          <Menu.TriggerItem aria-label={t('widgets.layers.menu.colorLabel')}>
            <HStack gap="2" minW="0" w="full">
              <Icon as={PaletteIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
              <Text flex="1" fontSize="md">
                {t('widgets.layers.menu.colorLabel')}
              </Text>
              <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
            </HStack>
          </Menu.TriggerItem>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="10rem" py="1">
                {COLOR_LABEL_ITEMS.map((item) => (
                  <MenuActionItem
                    key={item.value}
                    icon={CircleIcon}
                    iconColor={item.hex}
                    label={t(item.labelKey, { defaultValue: item.defaultLabel })}
                    value={`color-label-${item.value}`}
                    onSelect={handleSetColorLabel(item.value)}
                  />
                ))}
                <MenuActionItem
                  disabled={group.colorLabel === undefined}
                  icon={CircleOffIcon}
                  label={t('widgets.layers.labels.none', { defaultValue: 'None' })}
                  value="color-label-none"
                  onSelect={handleSetColorLabel(null)}
                />
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      ) : (
        <MenuActionItem
          disabled
          icon={PaletteIcon}
          label={t('widgets.layers.menu.colorLabel')}
          value="color-label"
          onSelect={noop}
        />
      )}
      <Menu.Separator borderColor="border.subtle" />
      <MenuActionItem
        disabled={locked || holdsLocked}
        icon={Trash2Icon}
        label={t('widgets.layers.actions.delete')}
        tone="danger"
        value="delete"
        onSelect={handleDelete}
      />
    </MenuContent>
  );

  return (
    <>
      <Menu.Root lazyMount open={menuOpen} positioning={positioning} unmountOnExit onOpenChange={handleOpenChange}>
        <Portal>
          <Menu.Positioner>{items}</Menu.Positioner>
        </Portal>
      </Menu.Root>
      <RenameDialog
        initialName={group.name}
        isOpen={renaming}
        label={t('widgets.layers.actions.rename')}
        submitLabel={t('widgets.layers.actions.rename')}
        title={t('widgets.layers.actions.rename')}
        onClose={closeRename}
        onExitComplete={handleRenameExitComplete}
        onSubmit={handleRename}
      />
    </>
  );
};
