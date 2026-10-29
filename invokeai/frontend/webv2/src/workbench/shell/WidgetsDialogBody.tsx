import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget } from '@workbench/widgetContracts';

import { HStack, Portal, Stack, Text } from '@chakra-ui/react';
import { Button, CloseButton, useUnavailableControl } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Tooltip } from '@platform/ui/Tooltip';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { WidgetIcon } from '@workbench/iconResolver';
import { revealOrOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import {
  closeWidgetPlacement,
  dockFloatingPlacement,
  removeFloatingPlacement,
} from '@workbench/widgetPlacementCommands';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { closeWidgetsDialog, getWidgetsDialogReturnFocus } from './widgetsDialogStore';
import { addWidgetToLayout, canAddWidget, getWidgetsListRows, type WidgetsListRow } from './widgetsListModel';

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

/**
 * After a row's action replaces its buttons, focus lands on what the list offers now, never on the page: the same row
 * when it stays, else the first row of that widget type.
 */
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

/**
 * Every widget and where it is in this layout, with the way to bring it back: Show raises or opens it (and closes the
 * list), Add puts one in the layout, Dock returns a window to its panel, Remove takes it out (with an Undo toast).
 */
export const WidgetsDialogBody = ({ isOpen, onExitComplete }: { isOpen: boolean; onExitComplete: () => void }) => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
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
        getLabel: (manifest) => resolveWidgetLabel(manifest, t),
        getWidgetById,
        project,
        widgets: allWidgets,
      }),
    [allWidgets, getWidgetById, project, t]
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
        addWidgetToLayout({ getWidgetsForRegion, widget: row.widget, widgets });
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
            getWidgetById,
            instanceId: row.instanceId,
            project,
            region: row.placement.region,
            widgets,
          });
        }
        focusRow(row);
      },
      show: (row: WidgetsListRow) => {
        setReturnsFocus(false);
        closeWidgetsDialog();
        revealOrOpenWorkbenchWidget({ focus, getWidgetsForRegion, queries, widgets }, row.widget.manifest.id);
      },
    }),
    [focus, getWidgetById, getWidgetsForRegion, project, queries, widgets]
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
                  {t('shell.widgets.description')}
                </Dialog.Description>
                <WidgetsGroup actions={actions} label={t('shell.widgets.inLayout')} rows={placedRows} />
                <WidgetsGroup actions={actions} label={t('shell.widgets.notInLayout')} rows={hiddenRows} />
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
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
  label,
  rows,
}: {
  actions: WidgetsListActions;
  label: string;
  rows: readonly WidgetsListRow[];
}) => {
  const headingId = useId();

  return rows.length === 0 ? null : (
    <Stack gap="1">
      <Text color="fg.subtle" fontSize="xs" fontWeight="700" id={headingId} textTransform="uppercase">
        {label}
      </Text>
      <Stack aria-labelledby={headingId} as="ul" gap="0.5" listStyleType="none" m="0" p="0">
        {rows.map((row) => (
          <WidgetsRow key={getRowKey(row)} actions={actions} row={row} />
        ))}
      </Stack>
    </Stack>
  );
};

const WidgetsRow = ({ actions, row }: { actions: WidgetsListActions; row: WidgetsListRow }) => {
  const { t } = useTranslation();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const placementLabel = t(
    PLACEMENT_LABEL_KEYS[row.placement.kind === 'docked' ? row.placement.region : row.placement.kind]
  );
  const isOpenable = canAddWidget(row.widget, getWidgetsForRegion);
  const run = (action: keyof WidgetsListActions) => () => actions[action](row);
  // A type can be placed more than once (a Gallery view and a Gallery panel), so placed rows name where they are.
  const name =
    row.instanceId === null ? row.label : t('shell.widgets.rowName', { label: row.label, placement: placementLabel });

  return (
    <HStack
      as="li"
      borderColor="border.subtle"
      borderWidth="1px"
      data-widgets-row={getRowKey(row)}
      data-widgets-type={row.widget.manifest.id}
      gap="2"
      px="2.5"
      py="1.5"
      rounded="md"
    >
      <WidgetIcon boxSize="3.5" color="fg.muted" flexShrink={0} icon={row.widget.manifest.icon} />
      <Stack flex="1" gap="0" minW="0">
        <Text fontSize="sm" fontWeight="600" truncate>
          {row.label}
        </Text>
        <Text color="fg.muted" fontSize="xs">
          {placementLabel}
        </Text>
      </Stack>
      <HStack flexShrink={0} gap="1">
        {row.instanceId === null ? (
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
            {row.placement.kind === 'floating' ? (
              <Button
                aria-label={t('shell.widgets.dockNamed', { label: name })}
                size="xs"
                variant="ghost"
                onClick={run('dock')}
              >
                {t('shell.widgets.dock')}
              </Button>
            ) : null}
            <RemoveButton label={name} required={row.isRequired} onRemove={run('remove')} />
          </>
        )}
      </HStack>
    </HStack>
  );
};

/** The center always keeps one view, so its last one cannot be removed; the button says why instead of vanishing. */
const RemoveButton = ({ label, onRemove, required }: { label: string; onRemove: () => void; required: boolean }) => {
  const { t } = useTranslation();
  const unavailable = useUnavailableControl(required ? t('widgets.requiredCenterView') : undefined, undefined);
  const button = (
    <Button
      {...unavailable?.props}
      aria-label={t('shell.widgets.removeNamed', { label })}
      opacity={required ? 0.5 : undefined}
      size="xs"
      variant="ghost"
      onClick={required ? undefined : onRemove}
    >
      {t('shell.widgets.remove')}
      {unavailable?.description}
    </Button>
  );

  return required ? <Tooltip content={t('widgets.requiredCenterView')}>{button}</Tooltip> : button;
};
