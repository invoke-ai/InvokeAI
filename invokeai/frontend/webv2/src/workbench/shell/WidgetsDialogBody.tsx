/* oxlint-disable react-perf/jsx-no-jsx-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget } from '@workbench/widgetContracts';

import { Box, Portal, Stack } from '@chakra-ui/react';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListSectionHeader } from '@platform/ui/list/ListSectionHeader';
import { ListStack } from '@platform/ui/list/ListStack';
import { WidgetIcon } from '@workbench/iconResolver';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import {
  addWidgetToWorkspace,
  closeWidgetPlacement,
  dockFloatingPlacement,
  removeFloatingPlacement,
} from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCanEditStructure } from '@workbench/workspace/useStructureLock';
import { useWorkspaceOpen } from '@workbench/workspace/useWorkspaceOpen';
import { startCustomizing } from '@workbench/workspace/workspaceNavigation';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { closeWidgetsDialog, getWidgetsDialogReturnFocus } from './widgetsDialogStore';
import { canAddWidget, getWidgetsListRows, type WidgetsListRow } from './widgetsListModel';

const REGIONS: readonly WidgetRegion[] = ['left', 'center', 'right', 'bottom'];

const PLACEMENT_LABEL_KEYS = {
  bottom: 'shell.widgets.placement.bottom',
  center: 'shell.widgets.placement.center',
  floating: 'shell.widgets.placement.floating',
  hidden: 'shell.widgets.placement.hidden',
  left: 'shell.widgets.placement.left',
  right: 'shell.widgets.placement.right',
} as const;

/** A row's key: the instance it lists, or the widget type for a row offering to add one. */
const getRowKey = (row: WidgetsListRow): string => row.instanceId ?? `type:${row.widget.manifest.id}`;

/** After a row's action, focus lands on what the list offers now, never on the page. */
const focusRow = (row: WidgetsListRow) =>
  requestAnimationFrame(() => {
    const list = document.querySelector('[data-widgets-list]');
    const rowElement =
      list?.querySelector(`[data-widgets-row="${CSS.escape(getRowKey(row))}"]`) ??
      list?.querySelector(`[data-widgets-type="${CSS.escape(row.widget.manifest.id)}"]`);

    // A row with nothing left to press hands focus to the dialog's Done.
    (
      rowElement?.querySelector<HTMLElement>('button:not([aria-disabled="true"])') ??
      list?.closest('[role="dialog"]')?.querySelector<HTMLElement>('[data-widgets-done]')
    )?.focus();
  });

export const WidgetsDialogBody = ({ isOpen, onExitComplete }: { isOpen: boolean; onExitComplete: () => void }) => {
  const { t } = useTranslation();
  const commands = useWorkbenchCommands();
  const { widgets } = commands;
  const queries = useWorkbenchQueries();
  const open = useWorkspaceOpen();
  // Locked, the list only shows: Add, Dock and Remove are Customize's, one press away in the footer.
  const canEditStructure = useCanEditStructure();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const project = useWorkbenchSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  // Show moves focus to the widget; closing then must not pull it back to whatever opened the list.
  const [returnsFocus, setReturnsFocus] = useState(true);
  const allWidgets = useMemo(() => {
    const byId = new Map<string, RegisteredWidget>();

    for (const region of REGIONS) {
      for (const widget of getWidgetsForRegion(region)) {
        if (widget.status === 'enabled') {
          byId.set(widget.manifest.id, widget);
        }
      }
    }

    return [...byId.values()];
  }, [getWidgetsForRegion]);
  const rows = useMemo(
    () =>
      getWidgetsListRows({
        availability: queries.getAvailability(),
        getLabel: (manifest) => resolveWidgetLabel(manifest, t),
        project,
        widgets: allWidgets,
      }),
    [allWidgets, project, queries, t]
  );
  const placedRows = useMemo(() => rows.filter((row) => row.instanceId !== null), [rows]);
  const hiddenRows = useMemo(() => rows.filter((row) => row.instanceId === null), [rows]);

  const getFinalFocus = useCallback(() => (returnsFocus ? getWidgetsDialogReturnFocus() : null), [returnsFocus]);
  const handleOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      closeWidgetsDialog();
    }
  }, []);
  const actions = useMemo(
    () => ({
      add: (row: WidgetsListRow) => {
        addWidgetToWorkspace({ getWidgetsForRegion, widget: row.widget, widgets });
        focusRow(row);
      },
      dock: (row: WidgetsListRow) => {
        if (row.instanceId) {
          dockFloatingPlacement({ instanceId: row.instanceId, project, widgets });
          focusRow(row);
        }
      },
      remove: (row: WidgetsListRow) => {
        if (row.instanceId === null) {
          return;
        }
        if (row.placement.kind === 'floating') {
          removeFloatingPlacement({ instanceId: row.instanceId, project, widgets });
        } else if (row.placement.kind === 'docked') {
          closeWidgetPlacement({
            availability: queries.getAvailability(),
            instanceId: row.instanceId,
            project,
            region: row.placement.region,
            widgets,
          });
        }
        focusRow(row);
      },
      customize: () => startCustomizing({ commands, queries }, { focusBar: false }),
      show: (row: WidgetsListRow) => {
        setReturnsFocus(false);
        closeWidgetsDialog();
        // The list describes this workspace, so Show stays in it.
        open({ typeId: row.widget.manifest.id }, { mode: 'stay' });
      },
    }),
    [commands, getWidgetsForRegion, open, project, queries, widgets]
  );

  return (
    <Dialog.Root
      lazyMount
      finalFocusEl={getFinalFocus}
      open={isOpen}
      restoreFocus={returnsFocus}
      scrollBehavior="inside"
      unmountOnExit
      onExitComplete={onExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content maxW="32rem">
            <Dialog.Header>
              <Dialog.Title>{t('shell.widgets.title')}</Dialog.Title>
              <Dialog.CloseTrigger asChild>
                <CloseButton />
              </Dialog.CloseTrigger>
            </Dialog.Header>
            <Dialog.Body>
              <Stack data-widgets-list="" gap="4">
                <Dialog.Description color="fg.muted" fontSize="sm">
                  {t(canEditStructure ? 'shell.widgets.description' : 'shell.widgets.lockedDescription')}
                </Dialog.Description>
                <WidgetsGroup
                  actions={actions}
                  canEditStructure={canEditStructure}
                  label={t('shell.widgets.inLayout')}
                  rows={placedRows}
                />
                <WidgetsGroup
                  actions={actions}
                  canEditStructure={canEditStructure}
                  label={t('shell.widgets.notInLayout')}
                  rows={hiddenRows}
                />
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              {canEditStructure ? null : (
                <Button me="auto" variant="outline" onClick={actions.customize}>
                  {t('shell.workspace.customize')}
                </Button>
              )}
              <Button data-widgets-done="" variant="ghost" onClick={closeWidgetsDialog}>
                {t('common.done')}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

type WidgetsListActions = Record<'add' | 'dock' | 'remove' | 'show', (row: WidgetsListRow) => void>;

const WidgetsGroup = ({
  actions,
  canEditStructure,
  label,
  rows,
}: {
  actions: WidgetsListActions;
  canEditStructure: boolean;
  label: string;
  rows: readonly WidgetsListRow[];
}) =>
  rows.length === 0 ? null : (
    // Row surfaces bleed into the dialog padding so their text lines up with the description above.
    <Stack gap="0" mx="-2">
      <ListSectionHeader count={rows.length} label={label} />
      <ListStack label={label} dividers>
        {rows.map((row) => (
          <WidgetsRow key={getRowKey(row)} actions={actions} canEditStructure={canEditStructure} row={row} />
        ))}
      </ListStack>
    </Stack>
  );

/** Rows are static: their buttons are the tab stops, so a row's surface never takes focus itself. */
const WidgetsRow = ({
  actions,
  canEditStructure,
  row,
}: {
  actions: WidgetsListActions;
  canEditStructure: boolean;
  row: WidgetsListRow;
}) => {
  const { t } = useTranslation();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const placementLabel = t(
    PLACEMENT_LABEL_KEYS[row.placement.kind === 'docked' ? row.placement.region : row.placement.kind]
  );
  const isOpenable = canEditStructure && canAddWidget(row.widget, getWidgetsForRegion);
  const run = (action: keyof WidgetsListActions) => () => actions[action](row);
  // A type can be placed more than once (a Gallery view and a Gallery panel), so placed rows name where they are.
  const name =
    row.instanceId === null ? row.label : t('shell.widgets.rowName', { label: row.label, placement: placementLabel });

  return (
    // The list item carries the keys focusRow looks rows up by; ListItem renders the row inside it.
    <Box data-widgets-row={getRowKey(row)} data-widgets-type={row.widget.manifest.id} role="listitem">
      <ListItem
        actions={
          row.instanceId === null ? (
            isOpenable ? (
              <Button
                aria-label={t('shell.widgets.addNamed', { label: name })}
                size="xs"
                variant="outline"
                onClick={run('add')}
              >
                {t('shell.widgets.add')}
              </Button>
            ) : null
          ) : (
            <>
              <Button
                aria-label={t('shell.widgets.showNamed', { label: name })}
                size="xs"
                variant="outline"
                onClick={run('show')}
              >
                {t('shell.widgets.show')}
              </Button>
              {canEditStructure && row.placement.kind === 'floating' ? (
                <Button
                  aria-label={t('shell.widgets.dockNamed', { label: name })}
                  size="xs"
                  variant="ghost"
                  onClick={run('dock')}
                >
                  {t('shell.widgets.dock')}
                </Button>
              ) : null}
              {/* The center always keeps one view, so its last one cannot be removed; the button says why. */}
              {canEditStructure ? (
                <Button
                  aria-label={t('shell.widgets.removeNamed', { label: name })}
                  size="xs"
                  unavailableReason={row.isRequired ? t('widgets.requiredCenterView') : undefined}
                  variant="ghost"
                  onClick={run('remove')}
                >
                  {t('shell.widgets.remove')}
                </Button>
              ) : null}
            </>
          )
        }
        description={placementLabel}
        leading={<WidgetIcon boxSize="4" color="fg.muted" flexShrink={0} icon={row.widget.manifest.icon} />}
        role="presentation"
        title={row.label}
        titleTruncate="end"
      />
    </Box>
  );
};
