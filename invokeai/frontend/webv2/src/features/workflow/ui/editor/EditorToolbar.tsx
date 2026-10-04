import { Box, Icon, Popover, Portal, Slider, Stack, Text } from '@chakra-ui/react';
import {
  useWorkflowNotifications,
  useWorkflowPreferencesSelector,
  useWorkflowProjectSelector,
} from '@features/workflow/ui/WorkflowUiContext';
import { IconButton, PopoverContent, Toolbar, ToolbarButton, ToolbarSeparator, Tooltip } from '@platform/ui';
import { useReactFlow } from '@xyflow/react';
import {
  BlendIcon,
  BoxSelectIcon,
  CameraIcon,
  EraserIcon,
  HandIcon,
  LassoIcon,
  CircleArrowUpIcon,
  MaximizeIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';

/**
 * Pane drag follows the selected pan/box/lasso tool; eraser clicks delete. Shift selects while panning, and middle
 * mouse pans during box selection.
 */
export type EditorTool = 'pan' | 'box-select' | 'lasso' | 'eraser';

const TOOLS: { icon: typeof HandIcon; id: EditorTool; labelKey: string }[] = [
  { icon: HandIcon, id: 'pan', labelKey: 'widgets.workflow.toolbar.pan' },
  { icon: BoxSelectIcon, id: 'box-select', labelKey: 'widgets.workflow.toolbar.boxSelect' },
  { icon: LassoIcon, id: 'lasso', labelKey: 'widgets.workflow.toolbar.lasso' },
  { icon: EraserIcon, id: 'eraser', labelKey: 'widgets.workflow.toolbar.eraser' },
];

const POPOVER_POSITIONING = { placement: 'right' } as const;
const TOOLTIP_POSITIONING = { placement: 'right-start' } as const;

/**
 * Clears the center region's floating chrome, whose inset already includes the
 * gap below the islands. In the left and bottom panels the variable is
 * undefined and the strip falls back to its own top margin.
 */
const EDITOR_TOOLBAR_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

const waitForExportFrame = () =>
  new Promise<void>((resolve) => {
    if (document.visibilityState === 'hidden') {
      window.setTimeout(resolve, 0);
    } else {
      window.requestAnimationFrame(() => resolve());
    }
  });

export const EditorToolbar = ({
  nodeOpacity,
  tool,
  updatableNodeCount = 0,
  onExportPrepare,
  onExportComplete,
  onNodeOpacityChange,
  onToolChange,
  onUpdateNodes,
}: {
  nodeOpacity: number;
  tool: EditorTool;
  /** Nodes with a newer same-major template; the update button shows only while there are some. */
  updatableNodeCount?: number;
  onNodeOpacityChange: (opacity: number) => void;
  onExportPrepare: () => Promise<HTMLElement>;
  onExportComplete: () => void;
  onToolChange: (tool: EditorTool) => void;
  onUpdateNodes?: () => void;
}) => {
  const { t } = useTranslation();
  const { fitView, getNodes, getNodesBounds, zoomIn, zoomOut } = useReactFlow();
  const reduceMotion = useWorkflowPreferencesSelector((preferences) => preferences.reduceMotion);
  const workflowName = useWorkflowProjectSelector((project) => project.activeWorkflow.document.name);
  const notifications = useWorkflowNotifications();
  const opacityTriggerId = useId();
  const isExportingWorkflowRef = useRef(false);
  const exportButtonRef = useRef<HTMLButtonElement>(null);
  const [isExportingWorkflow, setIsExportingWorkflow] = useState(false);
  const fitViewDuration = reduceMotion ? 0 : 300;
  const fallbackWorkflowName = t('widgets.workflow.untitled');
  const exportFailedLabel = t('widgets.workflow.exportImageFailed');
  const opacityIds = useMemo(() => ({ trigger: opacityTriggerId }), [opacityTriggerId]);
  const opacityValue = useMemo(() => [Math.round(nodeOpacity * 100)], [nodeOpacity]);
  const onZoomInClick = useCallback(() => void zoomIn(), [zoomIn]);
  const onZoomOutClick = useCallback(() => void zoomOut(), [zoomOut]);
  const onFitViewClick = useCallback(() => void fitView({ duration: fitViewDuration }), [fitView, fitViewDuration]);
  const onExportWorkflowClick = useCallback(() => {
    if (isExportingWorkflowRef.current) {
      return;
    }

    isExportingWorkflowRef.current = true;
    const hadFocus = document.activeElement === exportButtonRef.current;
    setIsExportingWorkflow(true);
    void (async () => {
      try {
        const { exportWorkflowAsPng, preflightWorkflowImageExport } = await import('./workflowImageExport');
        const bounds = getNodesBounds(getNodes());
        // Refuse from the live editor's bounds before mounting the whole graph a second time.
        let outcome = preflightWorkflowImageExport(bounds);
        if (!outcome) {
          const flowElement = await onExportPrepare();
          await waitForExportFrame();
          await waitForExportFrame();
          outcome = await exportWorkflowAsPng({ bounds, fallbackWorkflowName, flowElement, workflowName });
        }
        if (outcome.status === 'busy') {
          notifications.info(t('widgets.workflow.exportImageBusy'));
        } else if (outcome.status === 'too-large') {
          notifications.error(
            t('widgets.workflow.exportImageTooLarge'),
            t('widgets.workflow.exportImageTooLargeDetail')
          );
        } else if (outcome.status === 'exported' && outcome.reduced) {
          notifications.info(
            t('widgets.workflow.exportImageReduced'),
            t('widgets.workflow.exportImageReducedDetail', { height: outcome.height, width: outcome.width })
          );
        }
      } catch {
        notifications.error(exportFailedLabel);
      } finally {
        isExportingWorkflowRef.current = false;
        flushSync(() => setIsExportingWorkflow(false));
        // The busy camera is disabled, which drops its focus; return it unless focus has moved on since.
        if (hadFocus && document.activeElement === document.body) {
          exportButtonRef.current?.focus();
        }
        onExportComplete();
      }
    })();
  }, [
    exportFailedLabel,
    fallbackWorkflowName,
    getNodes,
    getNodesBounds,
    notifications,
    onExportPrepare,
    onExportComplete,
    t,
    workflowName,
  ]);
  const fitViewRef = useRef<HTMLButtonElement>(null);
  // The update button leaves with the last outdated node; keyboard focus steps to its stable neighbour first.
  const onUpdateNodesClick = useCallback(() => {
    fitViewRef.current?.focus();
    onUpdateNodes?.();
  }, [onUpdateNodes]);
  const onSliderValueChange = useCallback(
    (event: { value: number[] }) => onNodeOpacityChange((event.value[0] ?? 100) / 100),
    [onNodeOpacityChange]
  );

  return (
    <Box data-workflow-export-control="true" left="2" position="absolute" top={EDITOR_TOOLBAR_TOP} zIndex="5">
      <Toolbar>
        {TOOLS.map(({ icon, id, labelKey }) => (
          <EditorToolButton
            key={id}
            icon={icon}
            id={id}
            isActive={tool === id}
            label={t(labelKey)}
            onToolChange={onToolChange}
          />
        ))}
        <ToolbarSeparator />
        <ToolbarButton icon={ZoomInIcon} label={t('widgets.workflow.toolbar.zoomIn')} onClick={onZoomInClick} />
        <ToolbarButton icon={ZoomOutIcon} label={t('widgets.workflow.toolbar.zoomOut')} onClick={onZoomOutClick} />
        <ToolbarButton
          ref={fitViewRef}
          icon={MaximizeIcon}
          label={t('widgets.workflow.toolbar.fitView')}
          onClick={onFitViewClick}
        />
        <ToolbarButton
          ref={exportButtonRef}
          aria-busy={isExportingWorkflow}
          disabled={isExportingWorkflow}
          icon={CameraIcon}
          label={t('widgets.workflow.exportAsPng')}
          loading={isExportingWorkflow}
          onClick={onExportWorkflowClick}
        />
        {updatableNodeCount > 0 ? (
          <>
            <ToolbarSeparator />
            <ToolbarButton
              color="fg.warning"
              icon={CircleArrowUpIcon}
              label={t('nodes.updateAllNodes', { count: updatableNodeCount })}
              onClick={onUpdateNodesClick}
            />
          </>
        ) : null}
        <ToolbarSeparator />
        <Popover.Root ids={opacityIds} positioning={POPOVER_POSITIONING}>
          <Tooltip
            content={t('widgets.workflow.toolbar.nodeOpacity')}
            ids={opacityIds}
            positioning={TOOLTIP_POSITIONING}
          >
            <Popover.Trigger asChild>
              {/*
               * Render a plain button because asChild would clone ToolbarButton's Tooltip wrapper. Match xs sizing
               * so one child cannot stretch the column.
               */}
              <IconButton
                aria-label={t('widgets.workflow.toolbar.nodeOpacity')}
                aria-pressed={nodeOpacity < 1}
                variant={nodeOpacity < 1 ? 'solid' : 'ghost'}
              >
                <Icon as={BlendIcon} boxSize="3.5" />
              </IconButton>
            </Popover.Trigger>
          </Tooltip>
          <Portal>
            <Popover.Positioner>
              <PopoverContent w="12rem">
                <Popover.Body p="3">
                  <Stack gap="1.5">
                    <Text color="fg.muted" fontSize="xs" fontWeight="600">
                      {t('widgets.workflow.toolbar.nodeOpacityValue', { value: opacityValue[0] })}
                    </Text>
                    <Slider.Root max={100} min={20} step={5} value={opacityValue} onValueChange={onSliderValueChange}>
                      <Slider.Control>
                        <Slider.Track>
                          <Slider.Range />
                        </Slider.Track>
                        <Slider.Thumbs />
                      </Slider.Control>
                    </Slider.Root>
                  </Stack>
                </Popover.Body>
              </PopoverContent>
            </Popover.Positioner>
          </Portal>
        </Popover.Root>
      </Toolbar>
    </Box>
  );
};

const EditorToolButton = ({
  icon,
  id,
  isActive,
  label,
  onToolChange,
}: {
  icon: typeof HandIcon;
  id: EditorTool;
  isActive: boolean;
  label: string;
  onToolChange: (tool: EditorTool) => void;
}) => {
  const onClick = useCallback(() => onToolChange(id), [id, onToolChange]);

  return <ToolbarButton icon={icon} isActive={isActive} label={label} onClick={onClick} />;
};
