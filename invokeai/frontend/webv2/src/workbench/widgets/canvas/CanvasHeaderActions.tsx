import type { Rect } from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { Project } from '@workbench/projectContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, HStack, Icon, Menu, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { useModifierHeld } from '@platform/react/useModifierHeld';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Group } from '@platform/ui/Group';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { getCanvasEngine } from '@workbench/canvas-operations/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import {
  BugIcon,
  CheckIcon,
  ChevronDownIcon,
  DatabaseIcon,
  FilePlusIcon,
  FrameIcon,
  MaximizeIcon,
  Redo2Icon,
  SaveIcon,
  SquareDashedBottomIcon,
  Trash2Icon,
  Undo2Icon,
} from 'lucide-react';
import { useCallback, useEffect, useEffectEvent, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { CanvasHeaderCommandContext } from './canvasHeaderCommands';
import type { ResolvedCanvasSettings } from './canvasSettings';

import { useModelGridSize } from './bboxGrid';
import {
  applyFitBbox,
  confirmNewCanvas as confirmNewCanvasDocument,
  executeCanvasHeaderCommand,
  zoomAtViewportCentre,
} from './canvasHeaderCommands';
import { CANVAS_SNAP_TO_GRID_KEY, canvasSettingsEqual, resolveCanvasSettings } from './canvasSettings';
import { useCanvasCanRedo, useCanvasCanUndo, useCanvasDocumentEditingLocked, useCanvasZoom } from './engineStoreHooks';
import { computeFitBboxToLayers, computeFitBboxToMasks } from './fitBbox';
import { useCanvasGallerySave } from './useCanvasGallerySave';
import { reportStructuralCommit } from './useStructuralCommit';
import { formatZoomPercent, zoomMenuOptions } from './zoomOptions';

type CanvasHeaderEngine = Pick<
  CanvasEngineHandle,
  'diagnostics' | 'document' | 'exports' | 'history' | 'interaction' | 'layers' | 'lifecycle' | 'projectId' | 'viewport'
>;

const ZOOM_OPTIONS = zoomMenuOptions();
const MENU_POSITIONING = { placement: 'bottom-end' } as const;

/** Reads the persisted, default-applied canvas settings for the active project. */
const selectCanvasSettings = (project: Project): ResolvedCanvasSettings =>
  resolveCanvasSettings(getProjectWidgetValues(project, 'canvas'));

/** Reads the active generate model's base, so fit-bbox snaps to the same grid the bbox tool uses. */
const selectModelBase = (project: Project): string | null => {
  const values = getProjectWidgetValues(project, 'generate') as { model?: { base?: unknown } } | undefined;
  return typeof values?.model?.base === 'string' ? values.model.base : null;
};

const selectModelVariant = (project: Project): string | null => {
  const values = getProjectWidgetValues(project, 'generate') as { model?: { variant?: unknown } } | undefined;
  return typeof values?.model?.variant === 'string' ? values.model.variant : null;
};

export const CanvasHeaderActions = ({ runtime }: WidgetViewProps) => {
  const engine = useCanvasEngine();
  return engine ? <CanvasHeaderActionsInner engine={engine} runtime={runtime} /> : null;
};

const CanvasHeaderActionsInner = ({
  engine,
  runtime,
}: {
  engine: CanvasHeaderEngine;
  runtime: WidgetViewProps['runtime'];
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const zoom = useCanvasZoom(engine);
  const canUndo = useCanvasCanUndo(engine);
  const canRedo = useCanvasCanRedo(engine);
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  const { isSaving, save: saveToGallery } = useCanvasGallerySave(engine);
  const document = useActiveProjectSelector((project) => project.canvas.document);
  const modelBase = useActiveProjectSelector(selectModelBase);
  const modelVariant = useActiveProjectSelector(selectModelVariant);
  const settings = useActiveProjectSelector(selectCanvasSettings, canvasSettingsEqual);

  const [isNewCanvasOpen, setIsNewCanvasOpen] = useState(false);
  const closeNewCanvas = useCallback(() => setIsNewCanvasOpen(false), []);
  const openNewCanvas = useCallback(() => {
    if (!editingLocked) {
      setIsNewCanvasOpen(true);
    }
  }, [editingLocked]);

  const setZoom = (value: number) => zoomAtViewportCentre(engine, value);

  // Fit-bbox honors the snap-to-grid setting: snapping off ⇒ grid 1 (a plain round).
  const modelGrid = useModelGridSize(modelBase, modelVariant);
  const gridSize = settings[CANVAS_SNAP_TO_GRID_KEY] ? modelGrid : 1;
  const fitLayersRect = useMemo(() => computeFitBboxToLayers(document, gridSize), [document, gridSize]);
  const fitMasksRect = useMemo(() => computeFitBboxToMasks(document, gridSize), [document, gridSize]);

  const commandContext = (): CanvasHeaderCommandContext => ({
    document,
    editingLocked,
    engine,
    fitLayersRect,
    fitMasksRect,
    openNewCanvas,
    reportStructuralCommit: (result) => reportStructuralCommit(result, notify.error, t),
    saveToGallery: (region) => void saveToGallery(region),
    t,
  });

  const applyFit = (rect: Rect | null, refit: boolean) => applyFitBbox(commandContext(), rect, refit);

  const confirmNewCanvas = useCallback(
    () => confirmNewCanvasDocument({ document, editingLocked, engine }),
    [document, editingLocked, engine]
  );

  // Register catalog commands once while useEffectEvent reads current fit rects/dialog openers; new-session uses
  // the button's confirmation flow.
  const executeHeaderCommand = useEffectEvent((commandId: string) =>
    executeCanvasHeaderCommand(commandId, commandContext())
  );

  useEffect(() => {
    const entries = [
      ['canvas.fitBboxToLayers', t('widgets.canvas.controls.fitBboxToLayers'), ['shift+n']],
      ['canvas.fitBboxToMasks', t('widgets.canvas.controls.fitBboxToMasks'), ['shift+b']],
      // No default keys — assignable through the hotkeys settings.
      ['canvas.saveToGallery', t('widgets.canvas.contextMenu.saveCanvasToGallery'), []],
      ['canvas.saveBboxToGallery', t('widgets.canvas.contextMenu.saveBboxToGallery'), []],
      ['canvas.newSession', t('widgets.canvas.controls.newSession'), []],
    ] as const;
    const disposers = entries.flatMap(([id, title, defaultKeys]) => [
      runtime.commands.register({ handler: () => executeHeaderCommand(id), id, title }),
      runtime.hotkeys.register({ allowInEditable: false, commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);
    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);

  return (
    <HStack gap="0.5">
      <Menu.Root positioning={MENU_POSITIONING}>
        <Menu.Trigger asChild>
          <IconButton aria-label={t('widgets.canvas.controls.zoomLevel')} minW="4rem" px="2" size="2xs" variant="ghost">
            <HStack gap="1">
              <Text fontSize="xs" fontVariantNumeric="tabular-nums">
                {formatZoomPercent(zoom)}
              </Text>
              <ChevronDownIcon size={12} />
            </HStack>
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="7rem" py="1">
              {ZOOM_OPTIONS.map((option) => (
                <Menu.Item key={option.value} value={option.label} onClick={() => setZoom(option.value)}>
                  <CheckIcon size={12} opacity={formatZoomPercent(zoom) === option.label ? 1 : 0} />
                  <Menu.ItemText fontSize="xs">{option.label}</Menu.ItemText>
                </Menu.Item>
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <Tooltip content={t('widgets.canvas.controls.fitToView')}>
        <IconButton
          aria-label={t('widgets.canvas.controls.fitToView')}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={() => engine.viewport.fitToView()}
        >
          <MaximizeIcon />
        </IconButton>
      </Tooltip>

      <Tooltip content={t('widgets.canvas.controls.fitBboxToLayers')}>
        <IconButton
          aria-label={t('widgets.canvas.controls.fitBboxToLayers')}
          color="fg.muted"
          disabled={editingLocked || !fitLayersRect}
          size="2xs"
          variant="ghost"
          onClick={() => applyFit(fitLayersRect, true)}
        >
          <FrameIcon />
        </IconButton>
      </Tooltip>

      <Tooltip content={t('widgets.canvas.controls.fitBboxToMasks')}>
        <IconButton
          aria-label={t('widgets.canvas.controls.fitBboxToMasks')}
          color="fg.muted"
          disabled={editingLocked || !fitMasksRect}
          size="2xs"
          variant="ghost"
          onClick={() => applyFit(fitMasksRect, false)}
        >
          <SquareDashedBottomIcon />
        </IconButton>
      </Tooltip>

      <HeaderDivider />

      <Tooltip content={t('widgets.canvas.commands.undo')}>
        <IconButton
          aria-label={t('widgets.canvas.commands.undo')}
          color="fg.muted"
          disabled={editingLocked || !canUndo}
          size="2xs"
          variant="ghost"
          onClick={() => engine.history.undo()}
        >
          <Undo2Icon />
        </IconButton>
      </Tooltip>

      <Tooltip content={t('widgets.canvas.commands.redo')}>
        <IconButton
          aria-label={t('widgets.canvas.commands.redo')}
          color="fg.muted"
          disabled={editingLocked || !canRedo}
          size="2xs"
          variant="ghost"
          onClick={() => engine.history.redo()}
        >
          <Redo2Icon />
        </IconButton>
      </Tooltip>

      <HeaderDivider />

      <Menu.Root positioning={MENU_POSITIONING}>
        <Group attached css={SPLIT_GROUP_CSS}>
          <Tooltip content={t('widgets.canvas.contextMenu.saveCanvasToGallery')}>
            <IconButton
              aria-label={t('widgets.canvas.contextMenu.saveCanvasToGallery')}
              color="fg.muted"
              disabled={editingLocked || isSaving}
              size="2xs"
              variant="ghost"
              onClick={() => void saveToGallery('canvas')}
            >
              {isSaving ? <Spinner size="xs" /> : <SaveIcon />}
            </IconButton>
          </Tooltip>
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('widgets.canvas.controls.moreSaveOptions')}
              color="fg.muted"
              disabled={editingLocked || isSaving}
              minW="0"
              size="2xs"
              variant="ghost"
              w="6"
            >
              <ChevronDownIcon size={12} />
            </IconButton>
          </Menu.Trigger>
        </Group>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="11rem" py="1">
              <Menu.Item value="save-bbox" onClick={() => void saveToGallery('bbox')}>
                <Icon as={SaveIcon} boxSize="3.5" color="fg.subtle" />
                <Menu.ItemText fontSize="xs">{t('widgets.canvas.contextMenu.saveBboxToGallery')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <Menu.Root positioning={MENU_POSITIONING}>
        <Tooltip content={t('widgets.canvas.controls.newSession')}>
          <span style={{ display: 'inline-flex' }}>
            <Menu.Trigger asChild>
              <IconButton
                aria-label={t('widgets.canvas.controls.newSession')}
                color="fg.muted"
                disabled={editingLocked}
                size="2xs"
                variant="ghost"
              >
                <FilePlusIcon />
              </IconButton>
            </Menu.Trigger>
          </span>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="11rem" py="1">
              <Menu.Item value="new-canvas" onClick={openNewCanvas}>
                <Icon as={FilePlusIcon} boxSize="3.5" color="fg.subtle" />
                <Menu.ItemText fontSize="xs">{t('widgets.canvas.controls.newCanvas')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <ConfirmDialog
        body={t('widgets.canvas.controls.newCanvasConfirm')}
        confirmLabel={t('widgets.canvas.controls.newCanvas')}
        isOpen={isNewCanvasOpen}
        title={t('widgets.canvas.controls.newCanvas')}
        onClose={closeNewCanvas}
        onConfirm={confirmNewCanvas}
      />
    </HStack>
  );
};

/** A thin vertical rule separating header-action groups (matching legacy's dividers). */
// Ghost buttons have no border to collapse, and the attached overlap would
// leave the save button under 24px of unobscured target.
const SPLIT_GROUP_CSS = { '& > *:not(:last-child)': { marginEnd: 0 } } as const;

const HeaderDivider = () => <Box bg="border.subtle" flexShrink={0} h="4" mx="1" w="1px" />;

/** Diagnostics use an existing engine; opening settings must never acquire a canvas engine lease. */
export const CanvasSettingsActions = (_props: WidgetViewProps) => {
  const projectId = useActiveProjectId();
  const engine = getCanvasEngine(projectId);
  return engine ? <CanvasSettingsActionsInner engine={engine} /> : null;
};

const CanvasSettingsActionsInner = ({ engine }: { engine: CanvasHeaderEngine }) => {
  const { t } = useTranslation();
  const shiftHeld = useModifierHeld('Shift');
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  if (!shiftHeld) {
    return null;
  }
  return (
    <Stack borderTopWidth="1px" borderColor="border.subtle" gap="1" pt="2">
      <Text color="fg.subtle" fontSize="2xs" textTransform="uppercase">
        {t('widgets.canvas.settings.sections.debug')}
      </Text>
      <Button size="xs" variant="ghost" justifyContent="start" onClick={() => void engine.diagnostics.clearCaches()}>
        <DatabaseIcon />
        {t('widgets.canvas.settings.clearCaches')}
      </Button>
      <Button size="xs" variant="ghost" justifyContent="start" onClick={() => engine.diagnostics.logDebugInfo()}>
        <BugIcon />
        {t('widgets.canvas.settings.logDebugInfo')}
      </Button>
      <Button
        disabled={editingLocked}
        size="xs"
        variant="ghost"
        justifyContent="start"
        onClick={() => engine.history.clearHistory()}
      >
        <Trash2Icon />
        {t('widgets.canvas.settings.clearHistory')}
      </Button>
    </Stack>
  );
};
