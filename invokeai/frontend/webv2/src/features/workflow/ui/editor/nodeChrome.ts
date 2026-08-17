import type { BoxProps } from '@chakra-ui/react';

const NODE_HOVER_RING = '0 0 0 2px {colors.accent.solid/50}, {shadows.md}';
const NODE_RUNNING_RING = '0 0 0 2px {colors.brand.solid/70}, 0 0 10px {colors.brand.solid/50}';
const NODE_SELECTED_RING = '0 0 0 2px {colors.accent.solid}, {shadows.md}';

/** Shared spacing keeps editable workflow nodes aligned with their manager previews. */
export const WORKFLOW_NODE_DENSITY = {
  bodyPaddingY: '1',
  headerPaddingY: '1.5',
  rowPaddingX: '3',
  rowPaddingY: '0.5',
} as const;

export const getWorkflowNodeChromeProps = ({
  invalid = false,
  running = false,
  selected,
}: {
  invalid?: boolean;
  running?: boolean;
  selected: boolean;
}): BoxProps => ({
  borderColor: invalid ? 'red.solid' : running ? 'brand.solid' : 'border.emphasized',
  borderWidth: '1px',
  shadow: selected ? NODE_SELECTED_RING : running ? NODE_RUNNING_RING : 'sm',
  transition: 'border-color var(--wb-motion-duration-fast) ease, box-shadow var(--wb-motion-duration-fast) ease',
  _hover: selected ? undefined : { shadow: NODE_HOVER_RING },
});
