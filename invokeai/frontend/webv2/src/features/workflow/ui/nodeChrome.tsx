import type { BoxProps, FlexProps, IconButtonProps } from '@chakra-ui/react';
import type { FieldType } from '@features/workflow/contracts';
import type { LucideIcon } from 'lucide-react';
import type { CSSProperties, ReactNode } from 'react';

import { Box, Icon, IconButton, ProgressCircle, Spinner } from '@chakra-ui/react';
import { getFieldTypeColor, isModelFieldType } from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { CheckIcon, InfoIcon, OctagonXIcon, SquareArrowUpIcon, TriangleAlertIcon } from 'lucide-react';
import { createContext, useContext } from 'react';

const WorkflowImageExportContext = createContext(false);

export const WorkflowImageExportProvider = ({
  children,
  isExporting,
}: {
  children: ReactNode;
  isExporting: boolean;
}) => <WorkflowImageExportContext value={isExporting}>{children}</WorkflowImageExportContext>;

export const useIsWorkflowImageExport = () => useContext(WorkflowImageExportContext);

/** Share node styling across editor, static previews, and form builder to prevent visual drift. */

/** The node body surface. `flowThemeCss` bridges it to `--wb-node-surface` for inline xyflow styles. */
export const WORKFLOW_NODE_SURFACE_TOKEN = 'bg.muted';

/** Shared spacing keeps editable workflow nodes aligned with their manager previews. */
export const WORKFLOW_NODE_DENSITY = {
  bodyPaddingY: '1',
  headerGap: '2',
  /** A header's inset from the border to a button at its start or end; the same on both sides. */
  headerButtonInset: '1',
  /** A header's inset from the border to text at its start or end. */
  headerTextInset: '2.5',
  /**
   * A field row's label line: the label's line height and the height of the line that holds it and its port, which is
   * centred on it. The browser lays text out on whole pixels in the node's own coordinates, before the flow's zoom, so
   * a shorter line box centred in a taller line starts between pixels and its glyphs round up to a pixel away from the
   * port; zoomed in, that pixel grows.
   */
  labelLine: '20px',
  rowPaddingX: '3',
  rowPaddingY: '0.5',
} as const;

/** Independent facts about a node; `getWorkflowNodeAppearance` composes them, so no state overwrites another. */
export interface WorkflowNodeChromeState {
  /** A value is missing or invalid, or the node's type is unknown. */
  invalid?: boolean;
  /** The node's last invocation failed. It is drawn like an invalid node, so it shows when zoomed out. */
  failed?: boolean;
  /** The node's version differs from its template's; it still runs, so the tint is a warning, not an error. */
  outdated?: boolean;
  running?: boolean;
  selected: boolean;
}

/** The colour family of a node's outline and header. */
export type WorkflowNodeTone = 'error' | 'neutral' | 'warning';

export interface WorkflowNodeAppearance {
  tone: WorkflowNodeTone;
  border: string;
  /** The outline under the pointer: accent on a neutral node; a toned node takes a stronger step of its own tone. */
  hoverBorder: string;
  /** The selection ring, outside the border, in the node's state colour (accent when neutral); null when unselected. */
  selectionRing: string | null;
  /** The running glow, independent of tone and selection. */
  glows: boolean;
}

// The theme has no `border.warning`; warning borders use the warning foreground.
const TONE_BORDERS: Record<WorkflowNodeTone, string> = {
  error: 'border.error',
  neutral: 'border.emphasized',
  warning: 'fg.warning',
};

/**
 * A tone's hover step: its colour pulled toward the foreground, so it brightens on dark themes and deepens on light
 * ones and always differs from the resting outline.
 */
const getStrongerToneColor = (token: string): string => `color-mix(in oklab, {colors.${token}} 60%, {colors.fg})`;

export const getWorkflowNodeAppearance = ({
  failed,
  invalid,
  outdated,
  running,
  selected,
}: WorkflowNodeChromeState): WorkflowNodeAppearance => {
  const tone: WorkflowNodeTone = invalid || failed ? 'error' : outdated ? 'warning' : 'neutral';
  // A toned node keeps its colour in its selection ring, as legacy drew it, and under the pointer.
  const stateColor = tone === 'neutral' ? 'accent.solid' : TONE_BORDERS[tone];

  return {
    border: TONE_BORDERS[tone],
    glows: Boolean(running),
    hoverBorder: tone === 'neutral' ? 'accent.solid' : getStrongerToneColor(TONE_BORDERS[tone]),
    selectionRing: selected ? stateColor : null,
    tone,
  };
};

/**
 * The surface behind the node, so a ring's gap reads as space between the node and its ring: the canvas inside a flow,
 * the panel outside one (the form builder's cards).
 */
const RING_GAP = 'var(--xy-background-color, {colors.bg.panel})';
const NODE_RUNNING_GLOW = '0 0 0 2px {colors.brand.solid/70}, 0 0 10px {colors.brand.solid/50}';
/**
 * Keyboard focus draws a ring of its own just beyond the selection ring, so focus, selection and tone read at once. It
 * is set from CSS on the flow node's own focus, so focusing a node renders nothing. Every ring stays within 6px of the
 * border, where a port's dot covers it: an edge leaves the dot's outer side, so no ring is drawn across an edge.
 */
const NODE_FOCUS_RING_PROPERTY = '--wb-node-focus-ring';
const NODE_FOCUS_RING = `0 0 0 4px ${RING_GAP}, 0 0 0 6px {colors.fg}`;

export const getWorkflowNodeChromeProps = (state: WorkflowNodeChromeState): BoxProps => {
  const { border, glows, hoverBorder, selectionRing } = getWorkflowNodeAppearance(state);
  // Earlier shadows paint on top: the selection ring over the focus ring's gap, both over the glow.
  const shadows = [
    selectionRing ? `0 0 0 2px ${RING_GAP}, 0 0 0 4px {colors.${selectionRing}}` : null,
    `var(${NODE_FOCUS_RING_PROPERTY}, 0 0 #0000)`,
    glows ? NODE_RUNNING_GLOW : null,
    '{shadows.sm}',
  ];

  return {
    borderColor: border,
    borderWidth: '1px',
    shadow: shadows.filter(Boolean).join(', '),
    transition: 'border-color var(--wb-motion-duration-fast) ease, box-shadow var(--wb-motion-duration-fast) ease',
    _hover: { borderColor: hoverBorder },
    css: { '.react-flow__node:focus-visible &': { [NODE_FOCUS_RING_PROPERTY]: NODE_FOCUS_RING } },
  };
};

/** The node surface itself: chrome plus background, radius, and base type size. */
export const getWorkflowNodeShellProps = (state: WorkflowNodeChromeState): BoxProps => ({
  bg: 'bg',
  fontSize: 'md',
  rounded: 'lg',
  ...getWorkflowNodeChromeProps(state),
});

const HEADER_TONES: Record<WorkflowNodeTone, Pick<FlexProps, 'bg' | 'color'>> = {
  error: { bg: 'bg.error', color: 'fg.error' },
  neutral: { bg: 'bg.subtle' },
  warning: { bg: 'bg.warning', color: 'fg.warning' },
};

/**
 * Titled header strip, 30px high, tinted by the node's tone; its title and ghost buttons take the tone's foreground.
 * `roundedBottom` for collapsed nodes where nothing renders beneath it.
 */
export const getWorkflowNodeHeaderProps = ({
  endsWithButton = false,
  roundedBottom = false,
  startsWithButton = false,
  tone = 'neutral',
}: {
  /** The header ends in a ghost button (status, info): it sits as far from the right border as one at the start. */
  endsWithButton?: boolean;
  roundedBottom?: boolean;
  /** The header starts with a ghost button (collapse). */
  startsWithButton?: boolean;
  tone?: WorkflowNodeTone;
} = {}): FlexProps => ({
  alignItems: 'center',
  borderBottomRadius: roundedBottom ? 'lg' : undefined,
  borderBottomWidth: roundedBottom ? '0' : '1px',
  borderColor: 'border.subtle',
  borderTopRadius: 'lg',
  gap: WORKFLOW_NODE_DENSITY.headerGap,
  minH: '1.875rem',
  pe: endsWithButton ? WORKFLOW_NODE_DENSITY.headerButtonInset : WORKFLOW_NODE_DENSITY.headerTextInset,
  ps: startsWithButton ? WORKFLOW_NODE_DENSITY.headerButtonInset : WORKFLOW_NODE_DENSITY.headerTextInset,
  py: '0.5',
  ...HEADER_TONES[tone],
});

/** A header tint of the button's own colour, so a red or amber header never gets a grey block. */
const HEADER_BUTTON_FILL = 'color-mix(in oklab, currentColor 14%, transparent)';

/** The header's 22px ghost icon buttons (collapse, status, info), coloured by the header they sit on. */
const WORKFLOW_NODE_HEADER_BUTTON_PROPS = {
  boxSize: '1.375rem',
  color: 'inherit',
  flexShrink: 0,
  minW: '1.375rem',
  p: '0',
  rounded: 'sm',
  size: 'xs',
  variant: 'ghost',
  _expanded: { bg: HEADER_BUTTON_FILL },
  _focusVisible: { bg: HEADER_BUTTON_FILL },
  _hover: { bg: HEADER_BUTTON_FILL },
} as const satisfies IconButtonProps;

/** Marks a field row, and the control or static value inside it, for the body's bottom inset. */
export const WORKFLOW_NODE_FIELD_ROW_ATTRIBUTE = 'data-node-field-row';
export const WORKFLOW_NODE_FIELD_CONTROL_ATTRIBUTE = 'data-node-field-control';

/**
 * When the body ends in a field's control, the space under that control matches the rows' side inset, so the control
 * sits evenly inside the node. Both come from `rowPaddingX`; the last row's bottom padding makes up the difference
 * from the body's own. A body ending in a label-only row keeps its tighter rhythm.
 */
const BODY_LAST_CONTROL_CSS = {
  [`& > [${WORKFLOW_NODE_FIELD_ROW_ATTRIBUTE}]:last-child:has([${WORKFLOW_NODE_FIELD_CONTROL_ATTRIBUTE}], [data-workflow-export-field-value])`]:
    {
      pb: `calc({spacing.${WORKFLOW_NODE_DENSITY.rowPaddingX}} - {spacing.${WORKFLOW_NODE_DENSITY.bodyPaddingY}})`,
    },
};

/** Header supplies the divider. Inferred styles avoid BoxProps direction conflicts when spread into Stack/Flex. */
export const getWorkflowNodeBodyProps = ({ roundedBottom = true }: { roundedBottom?: boolean } = {}) => ({
  bg: WORKFLOW_NODE_SURFACE_TOKEN,
  borderBottomRadius: roundedBottom ? ('lg' as const) : ('none' as const),
  css: BODY_LAST_CONTROL_CSS,
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

/** A batch port's diamond: the handle's own rotation, which `flowThemeCss` turns back for the port's grab area. */
export const WORKFLOW_NODE_DIAMOND_ROTATION = 'rotate(45deg)';

/** Inline handle diamonds must retain side-specific xyflow centering before rotation. */
export const getWorkflowNodeHandleStyle = (type: FieldType, side: 'left' | 'right'): CSSProperties => {
  const visual = getHandleVisual(type);

  return {
    background: visual.isFilled ? visual.color : 'var(--wb-node-surface)',
    border: visual.isFilled ? 'none' : `${HANDLE_BORDER_WIDTH}px solid ${visual.color}`,
    borderRadius: visual.isAngular ? HANDLE_ANGULAR_RADIUS : '50%',
    boxShadow: `0 0 0 ${HANDLE_RING_WIDTH}px var(--wb-node-surface)`,
    height: WORKFLOW_NODE_HANDLE_SIZE,
    transform: visual.isDiamond
      ? `translate(${side === 'left' ? '-' : ''}50%, -50%) ${WORKFLOW_NODE_DIAMOND_ROTATION}`
      : undefined,
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
      transform={`translate(${side === 'left' ? '-50%' : '50%'}, -50%)${visual.isDiamond ? ` ${WORKFLOW_NODE_DIAMOND_ROTATION}` : ''}`}
      {...(side === 'left' ? { left: '0' } : { right: '0' })}
    />
  );
};

const PANEL_POSITIONING = { placement: 'top-end' } as const;
/** A plain status mark takes a header button's box, so the header's end inset holds. */
const WORKFLOW_NODE_STATUS_MARK_SIZE = { boxSize: '1.375rem' } as const;

/** A header ghost button whose tooltip is its hover and focus panel; like every tooltip, it mounts only while open. */
export const WorkflowNodeHeaderButton = ({
  children,
  closeOnClick = true,
  content,
  ...props
}: Omit<IconButtonProps, 'content'> & { 'aria-label': string; closeOnClick?: boolean; content: ReactNode }) => (
  <Tooltip closeOnClick={closeOnClick} content={content} positioning={PANEL_POSITIONING} showArrow>
    <IconButton {...WORKFLOW_NODE_HEADER_BUTTON_PROPS} {...props}>
      {children}
    </IconButton>
  </Tooltip>
);

/** The header info button; its panel carries the node details. */
export const WorkflowNodeInfoButton = ({
  content,
  label,
  onPreload,
}: {
  content: ReactNode;
  label: string;
  /** Starts loading a lazy panel body when the pointer or focus arrives, before the tooltip opens. */
  onPreload?: () => void;
}) => (
  <WorkflowNodeHeaderButton
    aria-label={label}
    closeOnClick={false}
    content={content}
    data-node-info-icon="true"
    onFocus={onPreload}
    onPointerEnter={onPreload}
  >
    <Icon as={InfoIcon} boxSize="3.5" />
  </WorkflowNodeHeaderButton>
);

/** What a node's status button reports; only one shows, the caller picks which. */
export type WorkflowNodeStatus = 'completed' | 'failed' | 'invalid' | 'running' | 'updatable' | 'versionMismatch';

/** No status glyph is a circle, so none reads as the info button's circled "i" beside it. */
const STATUS_GLYPHS: Record<Exclude<WorkflowNodeStatus, 'running'>, { color: string; icon: LucideIcon }> = {
  completed: { color: 'fg.success', icon: CheckIcon },
  failed: { color: 'fg.error', icon: OctagonXIcon },
  invalid: { color: 'fg.error', icon: TriangleAlertIcon },
  updatable: { color: 'fg.warning', icon: SquareArrowUpIcon },
  versionMismatch: { color: 'fg.warning', icon: TriangleAlertIcon },
};

/** The one progress indicator a running node has: a ring, or a spinner while progress is unknown. */
const NodeProgressRing = ({ progress }: { progress: number | null }) =>
  progress === null ? (
    <Spinner aria-hidden borderWidth="2px" boxSize="3.5" />
  ) : (
    <ProgressCircle.Root aria-hidden size="xs" value={Math.round(progress * 100)}>
      <ProgressCircle.Circle>
        <ProgressCircle.Track stroke="{colors.border.emphasized}" />
        <ProgressCircle.Range stroke="currentColor" />
      </ProgressCircle.Circle>
    </ProgressCircle.Root>
  );

/**
 * The header status button: its glyph keeps its status colour, and its panel says what the status means. With no panel
 * content it is a plain mark named by `label`, with no button or tooltip to mount.
 */
export const WorkflowNodeStatusButton = ({
  content,
  label,
  onPreload,
  progress = null,
  status,
}: {
  content: ReactNode;
  label: string;
  /** Starts loading a lazy panel body when the pointer or focus arrives, before the tooltip opens. */
  onPreload?: () => void;
  progress?: number | null;
  status: WorkflowNodeStatus;
}) =>
  content === null ? (
    <Box
      aria-label={label}
      color={status === 'running' ? 'brand.fg' : STATUS_GLYPHS[status].color}
      data-node-status={status}
      data-node-status-indicator="true"
      display="grid"
      flexShrink={0}
      placeItems="center"
      role="img"
      {...WORKFLOW_NODE_STATUS_MARK_SIZE}
    >
      {status === 'running' ? (
        <NodeProgressRing progress={progress} />
      ) : (
        <Icon aria-hidden as={STATUS_GLYPHS[status].icon} boxSize="3.5" />
      )}
    </Box>
  ) : (
    <WorkflowNodeHeaderButton
      aria-label={label}
      closeOnClick={false}
      color={status === 'running' ? 'brand.fg' : STATUS_GLYPHS[status].color}
      content={content}
      data-node-status={status}
      data-node-status-indicator="true"
      onFocus={onPreload}
      onPointerEnter={onPreload}
    >
      {status === 'running' ? (
        <NodeProgressRing progress={progress} />
      ) : (
        <Icon as={STATUS_GLYPHS[status].icon} boxSize="3.5" />
      )}
    </WorkflowNodeHeaderButton>
  );
