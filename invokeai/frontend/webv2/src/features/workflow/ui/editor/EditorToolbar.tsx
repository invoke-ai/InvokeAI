import { Box, Icon, Popover, Portal, Slider, Stack, Text } from '@chakra-ui/react';
import { useWorkflowPreferencesSelector } from '@features/workflow/ui/WorkflowUiContext';
import { IconButton, Toolbar, ToolbarButton, ToolbarSeparator, Tooltip } from '@platform/ui';
import { useReactFlow } from '@xyflow/react';
import {
  BlendIcon,
  BoxSelectIcon,
  EraserIcon,
  HandIcon,
  LassoIcon,
  MaximizeIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from 'lucide-react';
import { useCallback, useId, useMemo } from 'react';

/**
 * The editor's single tool strip, docked to the left edge and topped out
 * directly beneath the region's floating chrome islands: interaction tools on
 * top (legacy-toolbar style), viewport actions and the node-opacity slider
 * below.
 * - pan: dragging the pane moves the viewport (Shift-drag still box-selects)
 * - box-select: dragging the pane draws a selection rectangle (middle-mouse pans)
 * - lasso: dragging the pane draws a freeform selection
 * - eraser: clicking nodes or edges deletes them
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
  onNodeOpacityChange,
  onToolChange,
}: {
  nodeOpacity: number;
  tool: EditorTool;
  onNodeOpacityChange: (opacity: number) => void;
  onToolChange: (tool: EditorTool) => void;
}) => {
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const reduceMotion = useWorkflowPreferencesSelector((preferences) => preferences.reduceMotion);
  const opacityTriggerId = useId();
  const fitViewDuration = reduceMotion ? 0 : 300;
  const opacityIds = useMemo(() => ({ trigger: opacityTriggerId }), [opacityTriggerId]);
  const opacityValue = useMemo(() => [Math.round(nodeOpacity * 100)], [nodeOpacity]);
  const onZoomInClick = useCallback(() => void zoomIn(), [zoomIn]);
  const onZoomOutClick = useCallback(() => void zoomOut(), [zoomOut]);
  const onFitViewClick = useCallback(() => void fitView({ duration: fitViewDuration }), [fitView, fitViewDuration]);
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
        <ToolbarButton icon={MaximizeIcon} label="Fit view" onClick={onFitViewClick} />
        <ToolbarSeparator />
        <Popover.Root ids={opacityIds} positioning={POPOVER_POSITIONING}>
          <Tooltip content="Node opacity" ids={opacityIds} positioning={TOOLTIP_POSITIONING}>
            <Popover.Trigger asChild>
              {/* Matches what `ToolbarButton` renders rather than being one:
                  `Popover.Trigger asChild` would clone that component's Tooltip
                  wrapper instead of the button. Size is load bearing — `Toolbar`
                  is a column Stack, so it stretches every sibling to the widest
                  child, and one `sm` button here squared off the whole strip at
                  36px while the rest stayed 32px tall. */}
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
              <Popover.Content w="12rem">
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
              </Popover.Content>
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
