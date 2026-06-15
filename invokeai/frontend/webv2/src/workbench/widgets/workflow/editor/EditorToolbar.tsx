import { Box, Popover, Portal, Slider, Stack, Text } from '@chakra-ui/react';
import { Toolbar, ToolbarButton, ToolbarSeparator } from '@workbench/components/ui';
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

/**
 * The editor's single tool strip, docked to the left edge: interaction tools
 * on top (legacy-toolbar style), viewport actions and the node-opacity slider
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

  return (
    <Box left="3" position="absolute" top="50%" transform="translateY(-50%)" zIndex="5">
      <Toolbar>
        {TOOLS.map(({ icon, id, label }) => (
          <ToolbarButton key={id} icon={icon} isActive={tool === id} label={label} onClick={() => onToolChange(id)} />
        ))}
        <ToolbarSeparator />
        <ToolbarButton icon={ZoomInIcon} label="Zoom in" onClick={() => void zoomIn()} />
        <ToolbarButton icon={ZoomOutIcon} label="Zoom out" onClick={() => void zoomOut()} />
        <ToolbarButton icon={MaximizeIcon} label="Fit view" onClick={() => void fitView({ duration: 300 })} />
        <ToolbarSeparator />
        <Popover.Root positioning={{ placement: 'right' }}>
          <Popover.Trigger asChild>
            <ToolbarButton color={nodeOpacity < 1 ? 'accent.solid' : undefined} icon={BlendIcon} label="Node opacity" />
          </Popover.Trigger>
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
                      value={[Math.round(nodeOpacity * 100)]}
                      onValueChange={(event) => onNodeOpacityChange((event.value[0] ?? 100) / 100)}
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
