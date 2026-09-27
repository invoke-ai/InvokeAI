import type { BoxProps, FlexProps } from '@chakra-ui/react';
import type { FieldType } from '@features/workflow/contracts';
import type { CSSProperties, ReactNode } from 'react';

import { Box, Icon } from '@chakra-ui/react';
import { getFieldTypeColor, isModelFieldType } from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { CircleAlertIcon, CircleCheckIcon, InfoIcon } from 'lucide-react';

/** Share node styling across editor, static previews, and form builder to prevent visual drift. */

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

/** How the node's last invocation ended, once it is no longer running. */
export type WorkflowNodeOutcome = 'completed' | 'failed';

export interface WorkflowNodeChromeState {
  invalid?: boolean;
  /** The node's version differs from its template's; it still runs, so the tint is a warning, not an error. */
  outdated?: boolean;
  outcome?: WorkflowNodeOutcome | null;
  running?: boolean;
  selected: boolean;
}

const getNodeBorderColor = ({ invalid, outcome, outdated, running }: WorkflowNodeChromeState): string => {
  if (invalid) {
    return 'red.solid';
  }
  if (running) {
    return 'brand.solid';
  }
  // A failed run outranks a stale version: the nodes that cannot be updated are the ones most likely to fail.
  if (outcome === 'failed') {
    return 'border.error';
  }
  if (outdated) {
    return 'border.warning';
  }
  if (outcome === 'completed') {
    return 'fg.success';
  }

  return 'border.emphasized';
};

export const getWorkflowNodeChromeProps = (state: WorkflowNodeChromeState): BoxProps => ({
  borderColor: getNodeBorderColor(state),
  borderWidth: '1px',
  shadow: state.selected ? NODE_SELECTED_RING : state.running ? NODE_RUNNING_RING : 'sm',
  transition: 'border-color var(--wb-motion-duration-fast) ease, box-shadow var(--wb-motion-duration-fast) ease',
  _hover: state.selected ? undefined : { shadow: NODE_HOVER_RING },
});

/** The node surface itself: chrome plus background, radius, and base type size. */
export const getWorkflowNodeShellProps = (state: WorkflowNodeChromeState): BoxProps => ({
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

/** Header supplies the divider. Inferred styles avoid BoxProps direction conflicts when spread into Stack/Flex. */
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

/** Inline handle diamonds must retain side-specific xyflow centering before rotation. */
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

/** Header mark for a node whose invocation finished; the failure tooltip carries the backend message. */
export const WorkflowNodeOutcomeIcon = ({
  error,
  label,
  outcome,
}: {
  error?: string | null;
  label: string;
  outcome: WorkflowNodeOutcome;
}) => {
  const icon = (
    <Icon
      aria-label={label}
      as={outcome === 'completed' ? CircleCheckIcon : CircleAlertIcon}
      boxSize="3.5"
      color={outcome === 'completed' ? 'fg.success' : 'fg.error'}
      flexShrink={0}
      role="img"
    />
  );

  return outcome === 'failed' ? (
    <Tooltip content={error || label} positioning={INFO_TOOLTIP_POSITIONING} showArrow>
      {icon}
    </Tooltip>
  ) : (
    icon
  );
};
