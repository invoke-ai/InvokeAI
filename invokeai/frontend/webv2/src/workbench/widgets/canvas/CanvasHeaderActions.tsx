/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { Project, WidgetViewProps } from '@workbench/types';

import { Box, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { createNewCanvasStateV2 } from '@workbench/canvasMigration';
import { ConfirmDialog, IconButton, MenuContent, Tooltip } from '@workbench/components/ui';
import { useModifierHeld } from '@workbench/useModifierHeld';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import {
  BugIcon,
  CheckIcon,
  ChevronDownIcon,
  DatabaseIcon,
  FilePlusIcon,
  FrameIcon,
  MaximizeIcon,
  Redo2Icon,
  SettingsIcon,
  SquareDashedBottomIcon,
  Trash2Icon,
  Undo2Icon,
} from 'lucide-react';
import { Fragment, useCallback, useEffect, useEffectEvent, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { CanvasBooleanSetting, ResolvedCanvasSettings } from './canvasSettings';

import { gridSizeForModelBase } from './bboxGrid';
import {
  CANVAS_SETTING_SECTIONS,
  CANVAS_SETTINGS,
  CANVAS_SNAP_TO_GRID_KEY,
  canvasSettingsEqual,
  resolveCanvasSettings,
} from './canvasSettings';
import { useCanvasCanRedo, useCanvasCanUndo, useCanvasDocumentEditingLocked, useCanvasZoom } from './engineStoreHooks';
import { computeFitBboxToLayers, computeFitBboxToMasks } from './fitBbox';
import { useCanvasEngine } from './useCanvasEngine';
import { formatZoomPercent, zoomMenuOptions } from './zoomOptions';

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

/**
 * Canvas widget header actions, in legacy toolbar order: the zoom-percent menu, a
 * reset-view (fit content to view) button, fit-bbox-to-layers / fit-bbox-to-masks,
 * undo / redo, a new-session menu, and the sectioned settings popover. Rendered in
 * the widget frame's header slot; resolves the shared engine like the layers header
 * does, and renders nothing until it is available.
 *
 * Excluded per product decision: the project (save/load) menu, a save-to-gallery
 * button (its command/hotkey stays), and the snapshot menu.
 */
export const CanvasHeaderActions = ({ runtime }: WidgetViewProps) => {
  const engine = useCanvasEngine();
  return engine ? <CanvasHeaderActionsInner engine={engine} runtime={runtime} /> : null;
};

const CanvasHeaderActionsInner = ({
  engine,
  runtime,
}: {
  engine: CanvasEngine;
  runtime: WidgetViewProps['runtime'];
}) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const zoom = useCanvasZoom(engine);
  const canUndo = useCanvasCanUndo(engine);
  const canRedo = useCanvasCanRedo(engine);
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  const document = useActiveProjectSelector((project) => project.canvas.document);
  const modelBase = useActiveProjectSelector(selectModelBase);
  const settings = useActiveProjectSelector(selectCanvasSettings, canvasSettingsEqual);

  const [isNewCanvasOpen, setIsNewCanvasOpen] = useState(false);
  const closeNewCanvas = useCallback(() => setIsNewCanvasOpen(false), []);
  const openNewCanvas = useCallback(() => {
    if (!editingLocked) {
      setIsNewCanvasOpen(true);
    }
  }, [editingLocked]);

  const setZoom = (value: number) => {
    const viewport = engine.getViewport();
    const size = viewport.getViewportSize();
    viewport.zoomAtPoint(value, { x: size.width / 2, y: size.height / 2 });
  };

  // Fit-bbox honors the snap-to-grid setting: snapping off ⇒ grid 1 (a plain round).
  const gridSize = settings[CANVAS_SNAP_TO_GRID_KEY] ? gridSizeForModelBase(modelBase) : 1;
  const fitLayersRect = useMemo(() => computeFitBboxToLayers(document, gridSize), [document, gridSize]);
  const fitMasksRect = useMemo(() => computeFitBboxToMasks(document, gridSize), [document, gridSize]);

  // One undoable `setCanvasBbox` (inverse restores the current bbox) — exactly how a
  // manual bbox-tool edit commits. `refit` re-centers the view afterward (legacy
  // re-fits the stage after fit-to-layers, but not after fit-to-masks).
  const applyFit = (rect: Rect | null, refit: boolean) => {
    if (editingLocked || !rect) {
      return;
    }
    engine.commitStructural(
      t('widgets.canvas.commands.fitBbox'),
      { bbox: rect, type: 'setCanvasBbox' },
      { bbox: document.bbox, type: 'setCanvasBbox' }
    );
    if (refit) {
      engine.fitToView();
    }
  };

  const confirmNewCanvas = useCallback(() => {
    if (editingLocked) {
      return;
    }
    // A wholesale document replace (seeded with one empty inpaint mask, matching
    // Task 42's new-canvas init) at the current dimensions. The engine's mirror
    // treats this as a document swap and clears the canvas history by design, so
    // this is intentionally NOT undoable — the confirm dialog is the safety net.
    dispatch({
      document: createNewCanvasStateV2(document.width, document.height).document,
      type: 'replaceCanvasDocument',
    });
  }, [dispatch, document.height, document.width, editingLocked]);

  // Commands (hotkey-assignable; catalog ids `canvas.fitBboxToLayers` /
  // `canvas.fitBboxToMasks` / `canvas.newSession`). `useEffectEvent` reads the
  // latest fit rects / dialog opener without re-registering per document change.
  // The new-session command routes through the SAME confirm dialog as the button.
  const executeHeaderCommand = useEffectEvent((commandId: string) => {
    if (commandId === 'canvas.fitBboxToLayers') {
      applyFit(fitLayersRect, true);
    } else if (commandId === 'canvas.fitBboxToMasks') {
      applyFit(fitMasksRect, false);
    } else if (commandId === 'canvas.newSession') {
      openNewCanvas();
    }
  });
  useEffect(() => {
    const entries = [
      ['canvas.fitBboxToLayers', t('widgets.canvas.controls.fitBboxToLayers'), ['shift+n']],
      ['canvas.fitBboxToMasks', t('widgets.canvas.controls.fitBboxToMasks'), ['shift+b']],
      // No default keys — assignable through the hotkeys settings.
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
        <IconButton color="fg.muted" size="2xs" variant="ghost" onClick={() => engine.fitToView()}>
          <MaximizeIcon />
        </IconButton>
      </Tooltip>

      <Tooltip content={t('widgets.canvas.controls.fitBboxToLayers')}>
        <IconButton
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
          color="fg.muted"
          disabled={editingLocked || !canUndo}
          size="2xs"
          variant="ghost"
          onClick={() => engine.undo()}
        >
          <Undo2Icon />
        </IconButton>
      </Tooltip>

      <Tooltip content={t('widgets.canvas.commands.redo')}>
        <IconButton
          color="fg.muted"
          disabled={editingLocked || !canRedo}
          size="2xs"
          variant="ghost"
          onClick={() => engine.redo()}
        >
          <Redo2Icon />
        </IconButton>
      </Tooltip>

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

      <CanvasSettingsMenu engine={engine} />

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
const HeaderDivider = () => <Box bg="border.subtle" flexShrink={0} h="4" mx="1" w="1px" />;

/**
 * The gear-icon settings popover: the data-driven boolean preferences grouped into
 * Behavior / Display / Grid sections, plus a Shift-revealed Debug group of engine
 * actions (matching legacy `CanvasSettingsPopover`). `closeOnSelect={false}` keeps
 * it open while toggling. Settings persist per-project and never enter undo history.
 */
const CanvasSettingsMenu = ({ engine }: { engine: CanvasEngine }) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const settings = useActiveProjectSelector(selectCanvasSettings, canvasSettingsEqual);
  // Shift reveals the Debug section, matching legacy `useShiftModifier` (event-driven).
  const shiftHeld = useModifierHeld('Shift');

  const toggle = (setting: CanvasBooleanSetting) => {
    dispatch({
      type: 'patchWidgetValues',
      values: { [setting.key]: !settings[setting.key] },
      widgetId: 'canvas',
    });
  };

  return (
    <Menu.Root closeOnSelect={false} positioning={MENU_POSITIONING}>
      <Menu.Trigger asChild>
        <IconButton aria-label={t('widgets.canvas.settings.label')} color="fg.muted" size="2xs" variant="ghost">
          <SettingsIcon />
        </IconButton>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="15rem" py="1">
            {CANVAS_SETTING_SECTIONS.map((section, sectionIndex) => (
              <Fragment key={section}>
                {sectionIndex > 0 ? <Menu.Separator borderColor="border.subtle" /> : null}
                <Menu.ItemGroup>
                  <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                    {t(`widgets.canvas.settings.sections.${section}`)}
                  </Menu.ItemGroupLabel>
                  {CANVAS_SETTINGS.filter((setting) => setting.section === section).map((setting) => (
                    <Menu.Item key={setting.key} value={setting.key} onClick={() => toggle(setting)}>
                      <Icon as={CheckIcon} boxSize="3.5" opacity={settings[setting.key] ? 1 : 0} />
                      <Menu.ItemText fontSize="xs">{t(setting.labelKey)}</Menu.ItemText>
                    </Menu.Item>
                  ))}
                </Menu.ItemGroup>
              </Fragment>
            ))}
            {shiftHeld ? (
              <>
                <Menu.Separator borderColor="border.subtle" />
                <Menu.ItemGroup>
                  <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                    {t('widgets.canvas.settings.sections.debug')}
                  </Menu.ItemGroupLabel>
                  <Menu.Item value="debug-clear-caches" onClick={() => void engine.clearCaches()}>
                    <Icon as={DatabaseIcon} boxSize="3.5" color="fg.subtle" />
                    <Menu.ItemText fontSize="xs">{t('widgets.canvas.settings.clearCaches')}</Menu.ItemText>
                  </Menu.Item>
                  <Menu.Item value="debug-log-info" onClick={() => engine.logDebugInfo()}>
                    <Icon as={BugIcon} boxSize="3.5" color="fg.subtle" />
                    <Menu.ItemText fontSize="xs">{t('widgets.canvas.settings.logDebugInfo')}</Menu.ItemText>
                  </Menu.Item>
                  <Menu.Item value="debug-clear-history" onClick={() => engine.clearHistory()}>
                    <Icon as={Trash2Icon} boxSize="3.5" color="fg.subtle" />
                    <Menu.ItemText fontSize="xs">{t('widgets.canvas.settings.clearHistory')}</Menu.ItemText>
                  </Menu.Item>
                </Menu.ItemGroup>
              </>
            ) : null}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
