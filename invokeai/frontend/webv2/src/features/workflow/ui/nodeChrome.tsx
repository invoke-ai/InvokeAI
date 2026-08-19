import type { BoxProps, FlexProps } from '@chakra-ui/react';
import type { FieldType } from '@features/workflow/contracts';
import type { CSSProperties, ReactNode } from 'react';

import { Box, Icon } from '@chakra-ui/react';
import { getFieldTypeColor, isModelFieldType } from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { InfoIcon } from 'lucide-react';

/**
 * The single source of the workflow-node visual language. Every surface that
 * draws a node — the editor's flow nodes, the node manager's static preview
 * cards, the form builder — styles it from here so the renderings cannot
 * drift apart.
 */

const NODE_HOVER_RING = '0 0 0 2px {colors.accent.solid/50}, {shadows.md}';
const NODE_RUNNING_RING = '0 0 0 2px {colors.brand.solid/70}, 0 0 10px {colors.brand.solid/50}';
const NODE_SELECTED_RING = '0 0 0 2px {colors.accent.solid}, {shadows.md}';

/** The node body surface. `flowThemeCss` bridges it to `--wb-node-surface` for inline xyflow styles. */
export const WORKFLOW_NODE_SURFACE_TOKEN = 'bg.muted';

/** Shared spacing keeps editable workflow nodes aligned with their manager previews. */
export const WORKFLOW_NODE_DENSITY = {
  bodyPaddingY: '1',
  headerGap: '2',
  headerPaddingEnd: '2',
  headerPaddingStart: '2.5',
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

/** The node surface itself: chrome plus background, radius, and base type size. */
export const getWorkflowNodeShellProps = (state: {
  invalid?: boolean;
  running?: boolean;
  selected: boolean;
}): BoxProps => ({
  bg: 'bg',
  fontSize: 'xs',
  rounded: 'lg',
  ...getWorkflowNodeChromeProps(state),
});

/** Titled header strip. `roundedBottom` for collapsed nodes where nothing renders beneath it. */
export const getWorkflowNodeHeaderProps = ({ roundedBottom = false }: { roundedBottom?: boolean } = {}): FlexProps => ({
  alignItems: 'center',
  bg: 'bg.subtle',
  borderBottomRadius: roundedBottom ? 'lg' : undefined,
  borderBottomWidth: roundedBottom ? '0' : '1px',
  borderColor: 'border.subtle',
  borderTopRadius: 'lg',
  gap: WORKFLOW_NODE_DENSITY.headerGap,
  pe: WORKFLOW_NODE_DENSITY.headerPaddingEnd,
  ps: WORKFLOW_NODE_DENSITY.headerPaddingStart,
  py: WORKFLOW_NODE_DENSITY.headerPaddingY,
});

/**
 * Field-row body. The header above supplies the divider, so the body draws no
 * top border. Inferred (not `BoxProps`) so it spreads into `Stack`/`Flex`
 * without tripping over the conflicting HTML `direction` prop types.
 */
export const getWorkflowNodeBodyProps = ({ roundedBottom = true }: { roundedBottom?: boolean } = {}) => ({
  bg: WORKFLOW_NODE_SURFACE_TOKEN,
  borderBottomRadius: roundedBottom ? ('lg' as const) : ('none' as const),
  py: WORKFLOW_NODE_DENSITY.bodyPaddingY,
});

export const WORKFLOW_NODE_HANDLE_SIZE = 12;
/** Raw px (equal to `radii.xs`) so the inline xyflow flavor renders identically to the Chakra one. */
const HANDLE_ANGULAR_RADIUS = 2;
const HANDLE_BORDER_WIDTH = 2;
const HANDLE_RING_WIDTH = 1.5;

/**
 * One visual grammar for field handles: filled = single cardinality,
 * angular = model/batch types, diamond = batch.
 */
const getHandleVisual = (type: FieldType) => ({
  color: getFieldTypeColor(type),
  isAngular: isModelFieldType(type) || type.batch,
  isDiamond: Boolean(type.batch),
  isFilled: type.cardinality === 'SINGLE',
});

/**
 * Inline-CSS flavor for xyflow `<Handle>`s, which live outside Chakra's style
 * pipeline. xyflow's own `.react-flow__handle-left/right` CSS centers the
 * handle on its `left`/`right` coordinate via `translate(∓50%, -50%)`; the
 * diamond transform must restate that centering before rotating, so it needs
 * the side.
 */
export const getWorkflowNodeHandleStyle = (type: FieldType, side: 'left' | 'right'): CSSProperties => {
  const visual = getHandleVisual(type);

  return {
    background: visual.isFilled ? visual.color : 'var(--wb-node-surface)',
    border: visual.isFilled ? 'none' : `${HANDLE_BORDER_WIDTH}px solid ${visual.color}`,
    borderRadius: visual.isAngular ? HANDLE_ANGULAR_RADIUS : '50%',
    boxShadow: `0 0 0 ${HANDLE_RING_WIDTH}px var(--wb-node-surface)`,
    height: WORKFLOW_NODE_HANDLE_SIZE,
    transform: visual.isDiamond ? `translate(${side === 'left' ? '-' : ''}50%, -50%) rotate(45deg)` : undefined,
    width: WORKFLOW_NODE_HANDLE_SIZE,
  };
};

/**
 * Static stand-in for a connection handle in non-flow contexts (manager preview
 * cards): same size, tint, and shape rules as the editor handles, centered on
 * the node border.
 */
export const WorkflowNodeHandleDot = ({ side, type }: { side: 'left' | 'right'; type: FieldType }) => {
  const visual = getHandleVisual(type);

  return (
    <Box
      bg={visual.isFilled ? visual.color : WORKFLOW_NODE_SURFACE_TOKEN}
      borderColor={visual.color}
      borderRadius={visual.isAngular ? `${HANDLE_ANGULAR_RADIUS}px` : 'full'}
      borderWidth={visual.isFilled ? '0' : `${HANDLE_BORDER_WIDTH}px`}
      boxShadow={`0 0 0 ${HANDLE_RING_WIDTH}px {colors.${WORKFLOW_NODE_SURFACE_TOKEN}}`}
      boxSize={`${WORKFLOW_NODE_HANDLE_SIZE}px`}
      position="absolute"
      top="50%"
      transform={`translate(${side === 'left' ? '-50%' : '50%'}, -50%)${visual.isDiamond ? ' rotate(45deg)' : ''}`}
      {...(side === 'left' ? { left: '0' } : { right: '0' })}
    />
  );
};

const INFO_TOOLTIP_POSITIONING = { placement: 'top-end' } as const;

/** The quiet header info affordance: a `fg.subtle` icon whose tooltip carries the node details. */
export const WorkflowNodeInfoIcon = ({ content, label }: { content: ReactNode; label: string }) => (
  <Tooltip content={content} positioning={INFO_TOOLTIP_POSITIONING} showArrow>
    <Icon aria-label={label} as={InfoIcon} boxSize="3.5" color="fg.subtle" />
  </Tooltip>
);
