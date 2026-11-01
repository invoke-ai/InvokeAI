import type { CanvasDocumentContractV3, Rect } from '@workbench/canvas-engine/api';
import type { CanvasGallerySaveRegion, CreateFromBboxDestination } from '@workbench/canvas-operations/api';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { Project } from '@workbench/projectContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Checkbox, HStack, Menu, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { useModifierHeld } from '@platform/react/useModifierHeld';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MenuActionItem, MenuContent, MenuHeadingGroup, type MenuActionItemProps } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { getBaseRasterContentBounds, getDocumentLeaves } from '@workbench/canvas-engine/api';
import { getCanvasEngine, isBlankCanvasDocument } from '@workbench/canvas-operations/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { ShortcutKeycaps } from '@workbench/hotkeys/keyGlyphs';
import { useShortcutBinding } from '@workbench/hotkeys/useShortcutBinding';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectId, useActiveProjectSelector, useWidgetValuesSelector } from '@workbench/WorkbenchContext';
import {
  BugIcon,
  ChevronDownIcon,
  DatabaseIcon,
  FullscreenIcon,
  Redo2Icon,
  SaveIcon,
  ScanIcon,
  Trash2Icon,
  Undo2Icon,
  ZoomInIcon,
} from 'lucide-react';
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { CanvasHeaderCommandContext } from './canvasHeaderCommands';

import { useModelGridSize } from './bboxGrid';
import { CreateFromBboxMenuItems } from './CanvasCreateFromBboxSubmenu';
import {
  applyFitBbox,
  CANVAS_HEADER_HOTKEYS,
  executeCanvasHeaderCommand,
  zoomAtViewportCentre,
} from './canvasHeaderCommands';
import { getInFlightCanvasItems } from './canvasRestorePoints';
import { CANVAS_SNAP_TO_GRID_KEY, canvasSettingsEqual, resolveCanvasSettings } from './canvasSettings';
import {
  useCanvasCanRedo,
  useCanvasCanUndo,
  useCanvasDocumentEditingLocked,
  useCanvasRedoLabel,
  useCanvasUndoLabel,
  useCanvasZoom,
} from './engineStoreHooks';
import { computeFitBboxToLayers, computeFitBboxToMasks } from './fitBbox';
import { FitFrameToMasksIcon } from './frameIcons';
import { readCanvasCompositingSettings } from './invoke/canvasCompositing';
import { findCanvasSurface } from './surfaceFocus';
import { useCanvasGallerySave } from './useCanvasGallerySave';
import { useClearCanvas } from './useCanvasRecovery';
import { useCreateFromBbox } from './useCreateFromBbox';
import { reportStructuralCommit } from './useStructuralCommit';
import { formatZoomPercent, zoomMenuOptions } from './zoomOptions';

type CanvasHeaderEngine = Pick<
  CanvasEngineHandle,
  | 'diagnostics'
  | 'document'
  | 'exports'
  | 'history'
  | 'interaction'
  | 'layers'
  | 'lifecycle'
  | 'projectId'
  | 'tools'
  | 'viewport'
>;

const ZOOM_OPTIONS = zoomMenuOptions();
const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const ZOOM_MENU_POSITIONING = { placement: 'bottom-start' } as const;

/** Reads the active generate model's base, so fit-bbox snaps to the same grid the bbox tool uses. */
const selectModelBase = (project: Project): string | null => {
  const values = getProjectWidgetValues(project, 'generate') as { model?: { base?: unknown } } | undefined;
  return typeof values?.model?.base === 'string' ? values.model.base : null;
};

/** The project's clamped compositing mask blur, which fit-to-masks clears so the feathered edge stays in frame. */
const selectMaskBlur = (values: Record<string, unknown>): number => readCanvasCompositingSettings(values).maskBlur;

const selectModelVariant = (project: Project): string | null => {
  const values = getProjectWidgetValues(project, 'generate') as { model?: { variant?: unknown } } | undefined;
  return typeof values?.model?.variant === 'string' ? values.model.variant : null;
};

export const CanvasHeaderActions = ({ instance, runtime }: WidgetViewProps) => {
  const engine = useCanvasEngine();
  return engine ? <CanvasHeaderActionsInner engine={engine} instanceId={instance.id} runtime={runtime} /> : null;
};

const CanvasHeaderActionsInner = ({
  engine,
  instanceId,
  runtime,
}: {
  engine: CanvasHeaderEngine;
  instanceId: string;
  runtime: WidgetViewProps['runtime'];
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const zoom = useCanvasZoom(engine);
  const zoomLabel = formatZoomPercent(zoom);
  const canUndo = useCanvasCanUndo(engine);
  const canRedo = useCanvasCanRedo(engine);
  const undoLabel = useCanvasUndoLabel(engine);
  const redoLabel = useCanvasRedoLabel(engine);
  const undoText = undoLabel
    ? t('widgets.canvas.controls.undoStep', { label: undoLabel })
    : t('widgets.canvas.controls.undo');
  const redoText = redoLabel
    ? t('widgets.canvas.controls.redoStep', { label: redoLabel })
    : t('widgets.canvas.controls.redo');
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  const { describeDestination, isSaving, save: saveToGallery } = useCanvasGallerySave(engine);
  const { createFromBbox, isCreating } = useCreateFromBbox(engine);
  const saveMenuIds = useTooltipTriggerIds();
  const zoomMenuIds = useTooltipTriggerIds();
  const frameMenuIds = useTooltipTriggerIds();
  const frameLabel = t('widgets.canvas.controls.frame');
  const lockReason = editingLocked ? t('widgets.canvas.controls.operationRunning') : undefined;
  const saveLabel = isSaving
    ? t('widgets.canvas.controls.savingToGallery')
    : t('widgets.canvas.contextMenu.saveToGallery');
  const document = useActiveProjectSelector((project) => project.canvas.document);
  const modelBase = useActiveProjectSelector(selectModelBase);
  const modelVariant = useActiveProjectSelector(selectModelVariant);
  const settings = useWidgetValuesSelector('canvas', resolveCanvasSettings, canvasSettingsEqual);
  const maskBlur = useWidgetValuesSelector('canvas', selectMaskBlur);

  const [isNewCanvasOpen, setIsNewCanvasOpen] = useState(false);
  const [stopGenerations, setStopGenerations] = useState(true);
  const closeNewCanvas = useCallback(() => setIsNewCanvasOpen(false), []);
  const openNewCanvas = useCallback(() => {
    if (!editingLocked) {
      setStopGenerations(true);
      setIsNewCanvasOpen(true);
    }
  }, [editingLocked, setStopGenerations]);
  const onStopGenerationsChange = useCallback(
    (details: { checked: boolean | 'indeterminate' }) => setStopGenerations(details.checked === true),
    []
  );
  const clearCanvas = useClearCanvas();
  const layerCount = getDocumentLeaves(document).length;
  const stagedCount = useActiveProjectSelector((project) => project.canvas.stagingArea.pendingImages.length);
  // Nothing to bring back: Clear keeps no restore point, so the dialog does not promise one.
  const isBlank = stagedCount === 0 && isBlankCanvasDocument(document);
  const inFlightCount = useActiveProjectSelector(
    (project) => getInFlightCanvasItems(project.canvas, project.queue.items).length
  );

  const setZoom = (value: number) => zoomAtViewportCentre(engine, value);

  // Fit-bbox honors the snap-to-grid setting: snapping off ⇒ grid 1 (a plain round).
  const modelGrid = useModelGridSize(modelBase, modelVariant);
  const gridSize = settings[CANVAS_SNAP_TO_GRID_KEY] ? modelGrid : 1;
  const fitLayersRect = useMemo(() => computeFitBboxToLayers(document, gridSize), [document, gridSize]);
  const fitMasksRect = useMemo(
    () => computeFitBboxToMasks(document, gridSize, maskBlur),
    [document, gridSize, maskBlur]
  );

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

  // Select the Frame tool, then hand focus to the surface once the menu has returned it to the trigger.
  const editFrame = () => {
    engine.tools.setTool('bbox');
    requestAnimationFrame(() => findCanvasSurface(instanceId)?.focus({ preventScroll: true }));
  };

  // The save destination is only certain at save time; the menu names it as resolved when it opens.
  const [destination, setDestination] = useState<{ label: string | null; projectId: string } | null>(null);
  const destinationRequest = useRef(0);
  const onSaveMenuOpenChange = (details: { open: boolean }) => {
    if (!details.open) {
      return;
    }
    const request = ++destinationRequest.current;
    const projectId = engine.projectId;
    void describeDestination().then((label) => {
      if (destinationRequest.current === request) {
        setDestination({ label, projectId });
      }
    });
  };
  const destinationLabel = destination?.projectId === engine.projectId ? destination.label : null;

  const clearBody = useMemo(
    () => (
      <Stack gap="2">
        <Text>{t('widgets.canvas.controls.clearConfirmLayers', { count: layerCount })}</Text>
        {stagedCount > 0 ? (
          <Text>{t('widgets.canvas.controls.clearConfirmStaged', { count: stagedCount })}</Text>
        ) : null}
        {isBlank ? null : <Text>{t('widgets.canvas.controls.clearConfirmRestore')}</Text>}
        {inFlightCount > 0 ? (
          <>
            <Text>{t('widgets.canvas.controls.clearConfirmInFlight', { count: inFlightCount })}</Text>
            <Checkbox.Root checked={stopGenerations} size="sm" onCheckedChange={onStopGenerationsChange}>
              <Checkbox.HiddenInput />
              <Checkbox.Control>
                <Checkbox.Indicator />
              </Checkbox.Control>
              <Checkbox.Label>{t('widgets.canvas.controls.clearConfirmStop', { count: inFlightCount })}</Checkbox.Label>
            </Checkbox.Root>
          </>
        ) : null}
      </Stack>
    ),
    [inFlightCount, isBlank, layerCount, onStopGenerationsChange, stagedCount, stopGenerations, t]
  );

  // The lock is the second gate behind the dialog: an operation that started while it was open owns the document.
  const confirmNewCanvas = useCallback(
    () => (editingLocked ? undefined : clearCanvas(stopGenerations && inFlightCount > 0)),
    [clearCanvas, editingLocked, inFlightCount, stopGenerations]
  );

  // Register catalog commands once while useEffectEvent reads current fit rects/dialog openers; new-session uses
  // the button's confirmation flow.
  const executeHeaderCommand = useEffectEvent((commandId: string) =>
    executeCanvasHeaderCommand(commandId, commandContext())
  );

  useEffect(() => {
    const disposers = CANVAS_HEADER_HOTKEYS.flatMap(([id, titleKey, defaultKeys]) => {
      const title = t(titleKey);
      return [
        runtime.commands.register({ handler: () => executeHeaderCommand(id), id, title }),
        runtime.hotkeys.register({ allowInEditable: false, commandId: id, defaultKeys: [...defaultKeys], id, title }),
      ];
    });
    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);

  return (
    <HStack gap="0.5">
      <Menu.Root ids={zoomMenuIds} positioning={ZOOM_MENU_POSITIONING}>
        <Tooltip content={t('widgets.canvas.controls.zoom')} ids={zoomMenuIds}>
          <Menu.Trigger asChild>
            <Button
              aria-label={t('widgets.canvas.controls.zoomLevel', { zoom: zoomLabel })}
              color="fg.muted"
              fontVariantNumeric="tabular-nums"
              size="sm"
              variant="ghost"
            >
              <ZoomInIcon aria-hidden />
              {/* Wide enough for "500%" in tabular digits, so the trigger keeps its width as the zoom changes. */}
              <Text fontSize="md" minW="6ch" textAlign="center">
                {zoomLabel}
              </Text>
              <ChevronDownIcon size={12} />
            </Button>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="9rem">
              <MenuActionItem
                label={t('widgets.canvas.controls.fitToView')}
                value="fit"
                onSelect={() => engine.viewport.fitToView()}
              />
              {ZOOM_OPTIONS.map((option) => (
                <MenuActionItem
                  key={option.value}
                  disabled={zoomLabel === option.label}
                  label={option.label}
                  value={`zoom-${option.label}`}
                  onSelect={() => setZoom(option.value)}
                />
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <Menu.Root ids={frameMenuIds} lazyMount positioning={ZOOM_MENU_POSITIONING} unmountOnExit>
        <Tooltip content={frameLabel} ids={frameMenuIds}>
          <Menu.Trigger asChild>
            <Button aria-label={frameLabel} color="fg.muted" size="sm" variant="ghost">
              <ScanIcon aria-hidden />
              <ChevronDownIcon size={12} />
            </Button>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="15rem" py="1">
              <MenuActionItem
                icon={ScanIcon}
                label={t('widgets.canvas.controls.editFrame')}
                value="edit-frame"
                onSelect={editFrame}
              />
              <Menu.Separator />
              <CommandMenuItem
                commandId="canvas.fitBboxToLayers"
                disabled={editingLocked || !fitLayersRect}
                icon={FullscreenIcon}
                label={t('widgets.canvas.controls.fitBboxToLayers')}
                value="fit-layers"
                onSelect={() => applyFit(fitLayersRect, true)}
              />
              <CommandMenuItem
                commandId="canvas.fitBboxToMasks"
                disabled={editingLocked || !fitMasksRect}
                icon={FitFrameToMasksIcon}
                label={t('widgets.canvas.controls.fitBboxToMasks')}
                value="fit-masks"
                onSelect={() => applyFit(fitMasksRect, false)}
              />
              <Menu.Separator />
              <MenuHeadingGroup
                label={t('widgets.canvas.controls.useFrameAs')}
                trailing={t('widgets.canvas.controls.useFrameAsNote')}
              >
                {/* The context menu's Create from frame items; one composite runs at a time. */}
                <CreateFromBboxMenuItems
                  disabled={isSaving || isCreating}
                  unavailableReason={lockReason}
                  onCreate={(target: CreateFromBboxDestination) => void createFromBbox(target)}
                />
              </MenuHeadingGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <HeaderDivider />

      <Tooltip content={undoText}>
        <IconButton
          aria-label={undoText}
          color="fg.muted"
          disabled={editingLocked || !canUndo}
          size="sm"
          variant="ghost"
          onClick={() => engine.history.undo()}
        >
          <Undo2Icon />
        </IconButton>
      </Tooltip>

      <Tooltip content={redoText}>
        <IconButton
          aria-label={redoText}
          color="fg.muted"
          disabled={editingLocked || !canRedo}
          size="sm"
          variant="ghost"
          onClick={() => engine.history.redo()}
        >
          <Redo2Icon />
        </IconButton>
      </Tooltip>

      <HeaderDivider />

      <Menu.Root
        ids={saveMenuIds}
        lazyMount
        positioning={MENU_POSITIONING}
        unmountOnExit
        onOpenChange={onSaveMenuOpenChange}
      >
        <Tooltip content={saveLabel} ids={saveMenuIds}>
          <Menu.Trigger asChild>
            <Button
              aria-busy={isSaving || undefined}
              aria-label={saveLabel}
              color="fg.muted"
              disabled={editingLocked || isSaving || isCreating}
              size="sm"
              variant="ghost"
            >
              {isSaving ? <Spinner boxSize="3.5" /> : <SaveIcon aria-hidden />}
              <ChevronDownIcon size={12} />
            </Button>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="15rem" py="1">
              <SaveMenuItems
                board={destinationLabel}
                document={document}
                onSave={(region: CanvasGallerySaveRegion) => void saveToGallery(region)}
              />
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <IconButton
        aria-label={t('widgets.canvas.controls.clearCanvas')}
        color="fg.muted"
        size="sm"
        tooltip={t('widgets.canvas.controls.clearCanvas')}
        unavailableReason={editingLocked ? t('widgets.canvas.controls.operationRunning') : undefined}
        variant="ghost"
        onClick={openNewCanvas}
      >
        <Trash2Icon />
      </IconButton>

      <ConfirmDialog
        body={clearBody}
        confirmLabel={t('widgets.canvas.controls.clearCanvas')}
        isOpen={isNewCanvasOpen}
        title={t('widgets.canvas.controls.clearCanvasTitle')}
        onClose={closeNewCanvas}
        onConfirm={confirmNewCanvas}
      />
    </HStack>
  );
};

/** A menu item trailed by its command's effective first binding, remaps included, as keycaps. */
const CommandMenuItem = ({ commandId, ...item }: Omit<MenuActionItemProps, 'shortcut'> & { commandId: string }) => {
  const parts = useShortcutBinding(commandId)?.parts;
  // The binding's parts are a fresh array on every read; their serialized form is the stable key.
  const partsKey = parts ? JSON.stringify(parts) : null;
  const shortcut = useMemo(
    () => (partsKey ? <ShortcutKeycaps parts={JSON.parse(partsKey) as string[]} /> : undefined),
    [partsKey]
  );
  return <MenuActionItem {...item} shortcut={shortcut} />;
};

/** "{w} × {h} · {board}", dropping whichever part is unknown; undefined when neither is. */
const describeSave = (
  t: (key: string, options?: Record<string, unknown>) => string,
  size: { height: number; width: number } | null,
  board: string | null
): string | undefined => {
  if (size && board) {
    return t('widgets.canvas.controls.saveHint', { ...size, board });
  }
  if (size) {
    return t('widgets.canvas.controls.saveHintSize', size);
  }
  return board ?? undefined;
};

/**
 * The frame save, then the whole canvas, each saying what it would save and where. Mounted only while the menu is
 * open, so the canvas content bounds (the rect the canvas save exports) are planned then, not on every edit.
 */
const SaveMenuItems = ({
  board,
  document,
  onSave,
}: {
  board: string | null;
  document: CanvasDocumentContractV3;
  onSave: (region: CanvasGallerySaveRegion) => void;
}) => {
  const { t } = useTranslation();
  const content = useMemo(() => getBaseRasterContentBounds(document), [document]);
  const frameSize = { height: Math.round(document.bbox.height), width: Math.round(document.bbox.width) };

  return (
    <>
      <MenuActionItem
        hint={describeSave(t, frameSize, board)}
        icon={SaveIcon}
        label={t('widgets.canvas.contextMenu.saveBboxToGallery')}
        value="save-bbox"
        onSelect={() => onSave('bbox')}
      />
      <MenuActionItem
        hint={describeSave(t, content ? { height: content.height, width: content.width } : null, board)}
        icon={SaveIcon}
        label={t('widgets.canvas.contextMenu.saveCanvasToGallery')}
        value="save-canvas"
        onSelect={() => onSave('canvas')}
      />
    </>
  );
};

/** A thin vertical rule separating header-action groups (matching legacy's dividers). */
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
  if (!shiftHeld) {
    return null;
  }
  return (
    <Stack borderTopWidth="1px" borderColor="border.subtle" gap="1" pt="2">
      <Text color="fg.subtle" fontSize="xs" textTransform="uppercase">
        {t('widgets.canvas.settings.sections.debug')}
      </Text>
      <Button variant="ghost" justifyContent="start" onClick={() => void engine.diagnostics.clearCaches()}>
        <DatabaseIcon />
        {t('widgets.canvas.settings.clearCaches')}
      </Button>
      <Button variant="ghost" justifyContent="start" onClick={() => engine.diagnostics.logDebugInfo()}>
        <BugIcon />
        {t('widgets.canvas.settings.logDebugInfo')}
      </Button>
    </Stack>
  );
};
