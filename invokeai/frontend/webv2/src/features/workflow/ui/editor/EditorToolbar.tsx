import { Box, Icon, Popover, Portal, Slider, Stack, Text } from '@chakra-ui/react';
import { useWorkflowPreferencesSelector } from '@features/workflow/ui/WorkflowUiContext';
import { IconButton, PopoverContent, Toolbar, ToolbarButton, ToolbarSeparator, Tooltip } from '@platform/ui';
import { useReactFlow } from '@xyflow/react';
import {
  BlendIcon,
  BoxSelectIcon,
  EraserIcon,
  HandIcon,
  LassoIcon,
  CircleArrowUpIcon,
  MaximizeIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from 'lucide-react';
import { useCallback, useId, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Pane drag follows the selected pan/box/lasso tool; eraser clicks delete. Shift selects while panning, and middle
 * mouse pans during box selection.
 */
export type EditorTool = 'pan' | 'box-select' | 'lasso' | 'eraser';

const TOOLS: { icon: typeof HandIcon; id: EditorTool; label: string }[] = [
  { icon: HandIcon, id: 'pan', label: 'Pan (drag to move the viewport)' },
  { icon: BoxSelectIcon, id: 'box-select', label: 'Box select (drag to select nodes)' },
  { icon: LassoIcon, id: 'lasso', label: 'Lasso select (draw around nodes)' },
  { icon: EraserIcon, id: 'eraser', label: 'Eraser (click nodes or edges to delete)' },
];

const POPOVER_POSITIONING = { placement: 'right' } as const;
const TOOLTIP_POSITIONING = { placement: 'right-start' } as const;

/**
 * Clears the center region's floating chrome, whose inset already includes the
 * gap below the islands. In the left and bottom panels the variable is
 * undefined and the strip falls back to its own top margin.
 */
const EDITOR_TOOLBAR_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

export const EditorToolbar = ({
  nodeOpacity,
  tool,
  updatableNodeCount = 0,
  onNodeOpacityChange,
  onToolChange,
  onUpdateNodes,
}: {
  nodeOpacity: number;
  tool: EditorTool;
  /** Nodes with a newer same-major template; the update button shows only while there are some. */
  updatableNodeCount?: number;
  onNodeOpacityChange: (opacity: number) => void;
  onToolChange: (tool: EditorTool) => void;
  onUpdateNodes?: () => void;
}) => {
  const { t } = useTranslation();
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const reduceMotion = useWorkflowPreferencesSelector((preferences) => preferences.reduceMotion);
  const opacityTriggerId = useId();
  const fitViewDuration = reduceMotion ? 0 : 300;
  const opacityIds = useMemo(() => ({ trigger: opacityTriggerId }), [opacityTriggerId]);
  const opacityValue = useMemo(() => [Math.round(nodeOpacity * 100)], [nodeOpacity]);
  const onZoomInClick = useCallback(() => void zoomIn(), [zoomIn]);
  const onZoomOutClick = useCallback(() => void zoomOut(), [zoomOut]);
  const onFitViewClick = useCallback(() => void fitView({ duration: fitViewDuration }), [fitView, fitViewDuration]);
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
    <Box left="2" position="absolute" top={EDITOR_TOOLBAR_TOP} zIndex="5">
      <Toolbar>
        {TOOLS.map(({ icon, id, label }) => (
          <EditorToolButton
            key={id}
            icon={icon}
            id={id}
            isActive={tool === id}
            label={label}
            onToolChange={onToolChange}
          />
        ))}
        <ToolbarSeparator />
        <ToolbarButton icon={ZoomInIcon} label="Zoom in" onClick={onZoomInClick} />
        <ToolbarButton icon={ZoomOutIcon} label="Zoom out" onClick={onZoomOutClick} />
        <ToolbarButton ref={fitViewRef} icon={MaximizeIcon} label="Fit view" onClick={onFitViewClick} />
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
          <Tooltip content="Node opacity" ids={opacityIds} positioning={TOOLTIP_POSITIONING}>
            <Popover.Trigger asChild>
              {/*
               * Render a plain button because asChild would clone ToolbarButton's Tooltip wrapper. Match xs sizing
               * so one child cannot stretch the column.
               */}
              <IconButton
                aria-label="Node opacity"
                aria-pressed={nodeOpacity < 1}
                size="xs"
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
                    <Text color="fg.muted" fontSize="2xs" fontWeight="600">
                      Node opacity · {Math.round(nodeOpacity * 100)}%
                    </Text>
                    <Slider.Root
                      max={100}
                      min={20}
                      size="sm"
                      step={5}
                      value={opacityValue}
                      onValueChange={onSliderValueChange}
                    >
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
