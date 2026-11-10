import type {
  WorkflowPreviewLayout,
  WorkflowPreviewMetrics,
  WorkflowPreviewModel,
  WorkflowPreviewNode,
  WorkflowPreviewPill,
} from '@features/workflow/core/workflowPreview';
import type {
  Edge as FlowEdge,
  EdgeProps,
  EdgeTypes,
  Node as FlowNode,
  NodeMouseHandler,
  NodeProps,
  NodeTypes,
  ReactFlowInstance,
  Viewport,
} from '@xyflow/react';
import type { TFunction } from 'i18next';

import { Box, chakra, HStack, Icon, Stack, type SystemStyleObject, Text } from '@chakra-ui/react';
import {
  chooseWorkflowPreviewChipWidth,
  findWorkflowPreviewNodes,
  getWorkflowPreviewMetrics,
  WORKFLOW_PREVIEW_METRICS,
} from '@features/workflow/core/workflowPreview';
import '@xyflow/react/dist/style.css';
import { LOOP_LINKAGE_STROKE } from '@features/workflow/ui/loopLinkage';
import { useWorkflowPreferencesSelector } from '@features/workflow/ui/WorkflowUiContext';
import { createKeyedTransientStore, type KeyedTransientStore } from '@platform/state/externalStore';
import { Button, IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  Background,
  BackgroundVariant,
  getBezierPath,
  getViewportForBounds,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useStore,
} from '@xyflow/react';
import { FlagIcon, MinusIcon, PencilLineIcon, PlusIcon, TriangleAlertIcon, XIcon } from 'lucide-react';
import {
  createContext,
  memo,
  type KeyboardEvent as ReactKeyboardEvent,
  type Ref,
  type RefObject,
  use,
  useCallback,
  useId,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { flowThemeCss, getFlowColorMode } from './flowTheme';
import { formatWorkflowPreviewValue, getWorkflowNodeSummary } from './nodeSummaries';

/**
 * The library preview's graph of a saved workflow document (`buildWorkflowPreviewModel`): compact title chips, opened
 * simplified when anything folds, at a readable zoom. Selecting a node opens a card beside it and lights its
 * connections. Selection and find matches reach nodes and edges through a keyed store, so a change re-renders only the
 * nodes whose emphasis changes; dimming the rest is the container's CSS.
 */

export interface WorkflowDocumentGraphHandle {
  /** Closes the node card; false when none was open, so Escape can go on to leave the preview. */
  closeNodeCard: () => boolean;
  /** Emphasises the nodes matching a title or type id; a blank query clears it. */
  find: (query: string) => void;
  /** Centres the next match after the current one. */
  revealNextMatch: () => void;
}

export interface WorkflowDocumentGraphProps {
  ref?: Ref<WorkflowDocumentGraphHandle>;
  /** The drawing, laid out at the narrowest chips; the graph widens them where its stage has the room. */
  model: WorkflowPreviewModel;
  /** The same drawing laid out at other metrics (a wider chip), handed the last such layout to keep what it can. */
  relayout: (metrics: WorkflowPreviewMetrics, previous: WorkflowPreviewModel | null) => WorkflowPreviewModel;
  /** Nodes with a model requirement this server has not resolved. */
  needsModel: ReadonlySet<string>;
  /** A node to open selected, its card showing (an input's node, from the Inputs view). */
  initialNodeId?: string | null;
  /**
   * Whether the initial node's card takes focus (default). A card carried over from a previous drawing takes it only
   * when focus was lost with that drawing.
   */
  focusInitialCard?: boolean;
  /** Hears which node's card is open, as it opens and closes. */
  onCardChange?: (nodeId: string | null) => void;
  /** Hears how many drawn nodes match the find query, whenever the query or the drawn nodes change. */
  onMatchCountChange?: (count: number) => void;
  /** Where a compiled source's fields were set ("Generate → Steps"); the node card links each to its setting. */
  fieldSources?: WorkflowPreviewFieldSources;
}

export interface WorkflowPreviewFieldSources {
  /** The setting a field's value comes from, or null. */
  describe: (nodeId: string, fieldName: string) => string | null;
  /** Goes to the settings; the graph's host closes. */
  onSelect: () => void;
}

type Emphasis = 'selected' | 'related' | 'match' | 'current';
type EmphasisStore = KeyedTransientStore<string, Emphasis>;

interface ChipData extends Record<string, unknown> {
  node: WorkflowPreviewNode;
  pills: readonly WorkflowPreviewPill[];
  needsModel: boolean;
  summary: string | null;
}

type ChipNode = FlowNode<ChipData, 'chip'>;
type PreviewEdgeData = {
  isLoopLinkage: boolean;
  isRerouted: boolean;
  /** How far below the chip's middle (above, when negative) the edge meets its target. */
  entryOffset: number;
} & Record<string, unknown>;
type PreviewFlowEdge = FlowEdge<PreviewEdgeData, 'preview'>;

const { nodeHeight: NODE_HEIGHT, pillHeight: PILL_HEIGHT } = WORKFLOW_PREVIEW_METRICS;
/** A chip's title sits inside 10px of padding and a border of up to 2px on each side. */
const CHIP_TITLE_INSET = 24;
/** The smallest a chip's title may appear on screen. */
const MIN_TITLE_PX = 10;
/** Chip titles are the `md` size: 0.75rem. */
const TITLE_REM = 0.75;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 2;
const FIT_PADDING = 0.04;
/** A simplified graph shorter than this share of the stage spreads its rows out, up to the stretch limit. */
const ROW_SPREAD_SHARE = 0.6;
const MAX_ROW_STRETCH = 2.5;
const MINIMAP_STYLE = { height: 80, margin: 0, position: 'static', width: 128 } as const;
/** The gap left before the graph's start when it opens at the readable zoom rather than fitted. */
const START_INSET = 24;
/** Edges into one chip meet it this far apart, ordered as their sources sit, within a band this tall. */
const ENTRY_STEP = 6;
const ENTRY_SPAN = 24;
/** Between a control and the nodes fitted beside it. */
const CONTROL_GAP = 8;
const CARD_WIDTH = 300;
const CARD_GAP = 12;
const CARD_MARGIN = 8;
/**
 * The stage's controls sit over the graph; an open node card sits over them, in the same layer, so its chips and Close
 * are never under a control (it is placed clear of them where it can be).
 */
const OVERLAY_Z = 5;
const CARD_Z = 6;
/** The height a card is placed for until it has been measured. */
const CARD_ESTIMATED_HEIGHT = 300;
const ZOOM_STEP_DURATION = 150;
const REVEAL_DURATION = 200;

const EmphasisContext = createContext<EmphasisStore | null>(null);

const useEmphasis = (id: string): Emphasis | undefined => {
  const store = use(EmphasisContext);

  if (!store) {
    throw new Error('Preview graph parts render inside WorkflowDocumentGraph.');
  }

  return store.useValue(id);
};

/** The lowest zoom at which chip titles stay at least `MIN_TITLE_PX` on screen, at the page's root font size. */
const getReadableZoom = () => {
  const rootPx = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

  return Math.min(1, MIN_TITLE_PX / (rootPx * TITLE_REM));
};

const prefersReducedMotion = () => globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
const motion = (duration: number) => (prefersReducedMotion() ? 0 : duration);

/**
 * A node's flow box: its chip. Edges meet the chip's middle, so the pills it stacks above it sit outside the box
 * xyflow measures and draws to.
 */
const getChipBox = (layout: WorkflowPreviewLayout, id: string) => {
  const { x, y } = layout.positions[id] ?? { x: 0, y: 0 };

  return { height: NODE_HEIGHT, width: layout.nodeWidth, x, y };
};

/** What a node takes up on the stage: its chip and the pills above it. */
const getNodeBox = (layout: WorkflowPreviewLayout, id: string) => {
  const chip = getChipBox(layout, id);
  const pillsHeight = (layout.pills[id]?.length ?? 0) * PILL_HEIGHT;

  return { ...chip, height: chip.height + pillsHeight, y: chip.y - pillsHeight };
};

const getLayoutBounds = (layout: WorkflowPreviewLayout) => {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const id of layout.nodeIds) {
    const box = getNodeBox(layout, id);

    minX = Math.min(minX, box.x);
    minY = Math.min(minY, box.y);
    maxX = Math.max(maxX, box.x + box.width);
    maxY = Math.max(maxY, box.y + box.height);
  }

  return layout.nodeIds.length === 0
    ? { height: 0, width: 0, x: 0, y: 0 }
    : { height: maxY - minY, width: maxX - minX, x: minX, y: minY };
};

/** The stage's bands the controls cover: the legend along the top, zoom and the node count along the bottom. */
interface StageArea {
  width: number;
  height: number;
  top: number;
  bottom: number;
}

/**
 * The stage, less the bands its controls cover (`[data-preview-overlay]` above or below its middle) and a gap, so a
 * graph fitted or started in what is left never sits under a control.
 */
const measureStageArea = (stage: HTMLElement): StageArea => {
  const box = stage.getBoundingClientRect();
  let top = 0;
  let bottom = 0;

  for (const control of stage.querySelectorAll('[data-preview-overlay]')) {
    const rect = control.getBoundingClientRect();

    if (rect.height === 0) {
      continue;
    }

    if (rect.top + rect.height / 2 < box.top + box.height / 2) {
      top = Math.max(top, rect.bottom - box.top + CONTROL_GAP);
    } else {
      bottom = Math.max(bottom, box.bottom - rect.top + CONTROL_GAP);
    }
  }

  return { bottom, height: stage.clientHeight, top, width: stage.clientWidth };
};

/** The whole layout, fitted into the stage's open area. */
const getFittedViewport = (bounds: ReturnType<typeof getLayoutBounds>, area: StageArea) => {
  const open = Math.max(1, area.height - area.top - area.bottom);
  const fitted = getViewportForBounds(bounds, area.width, open, MIN_ZOOM, 1, FIT_PADDING);

  return { ...fitted, y: fitted.y + area.top };
};

/**
 * Fitted when that keeps titles readable; otherwise the readable zoom, starting at the graph's left edge (and its top,
 * when it is also too tall), with the minimap for the rest. Either way clear of the controls.
 */
const getOpeningViewport = (
  layout: WorkflowPreviewLayout,
  area: StageArea
): { isClipped: boolean; viewport: Viewport } => {
  const bounds = getLayoutBounds(layout);
  const fitted = getFittedViewport(bounds, area);
  const readable = getReadableZoom();

  if (fitted.zoom >= readable) {
    return { isClipped: false, viewport: fitted };
  }

  const zoom = readable;
  const scaledHeight = bounds.height * zoom;
  const open = area.height - area.top - area.bottom;

  return {
    isClipped: true,
    viewport: {
      x: START_INSET - bounds.x * zoom,
      y:
        scaledHeight + START_INSET * 2 <= open
          ? area.top + (open - scaledHeight) / 2 - bounds.y * zoom
          : area.top + START_INSET - bounds.y * zoom,
      zoom,
    },
  };
};

const handleStyle = {
  background: 'transparent',
  border: 'none',
  height: 1,
  minHeight: 1,
  minWidth: 1,
  width: 1,
} as const;
const proOptions = { hideAttribution: true } as const;
const flowStyle = { background: 'transparent' } as const;
const NO_PILLS: readonly WorkflowPreviewPill[] = [];

const CONTAINER_CSS: SystemStyleObject = {
  ...flowThemeCss,
  '& .react-flow__node': { cursor: 'pointer' },
  '& .react-flow__node:focus-visible': { outline: 'none' },
  '& .react-flow__node:focus-visible [data-preview-chip]': {
    outlineColor: 'accent.solid',
    outlineOffset: '2px',
    outlineStyle: 'solid',
    outlineWidth: '2px',
  },
  '& [data-preview-node]': { transitionDuration: 'fast', transitionProperty: 'opacity' },
  '& [data-preview-edge]': { transitionDuration: 'fast', transitionProperty: 'opacity, stroke' },
  '& [data-preview-edge][data-emphasis]': { stroke: 'accent.solid', strokeWidth: '2' },
  '&[data-dimmed] [data-preview-node]:not([data-emphasis])': { opacity: 0.38 },
  '&[data-dimmed] [data-preview-edge]:not([data-emphasis])': { opacity: 0.25 },
  '@media (prefers-reduced-motion: reduce)': {
    '& [data-preview-node], & [data-preview-edge]': { transitionDuration: '0ms' },
  },
};

const CHIP_EMPHASIS: Record<Emphasis, SystemStyleObject> = {
  current: { borderColor: 'accent.solid', boxShadow: '0 0 0 2px {colors.accent.solid}' },
  match: { borderColor: 'accent.solid' },
  related: { borderColor: 'accent.emphasized' },
  selected: { borderColor: 'accent.solid', boxShadow: '0 0 0 1px {colors.accent.solid}' },
};

const MarkerIcon = ({ marker }: { marker: 'exposed' | 'needsModel' | 'result' }) =>
  marker === 'exposed' ? (
    <Icon as={PencilLineIcon} aria-hidden boxSize="2.5" color="accent.fg" data-marker={marker} flexShrink={0} />
  ) : marker === 'needsModel' ? (
    <Icon as={TriangleAlertIcon} aria-hidden boxSize="2.5" color="fg.warning" data-marker={marker} flexShrink={0} />
  ) : (
    <Icon as={FlagIcon} aria-hidden boxSize="2.5" color="fg.muted" data-marker={marker} flexShrink={0} />
  );

const PillLabel = ({ pill }: { pill: WorkflowPreviewPill }) => {
  const { t } = useTranslation();

  return (
    <>
      {pill.label}
      {pill.value ? (
        <chakra.span color="fg" ms="1">
          {formatWorkflowPreviewValue(pill.value, t)}
        </chakra.span>
      ) : null}
    </>
  );
};

const PreviewChip = memo(function PreviewChip({ id, data }: NodeProps<ChipNode>) {
  const emphasis = useEmphasis(id);
  const { needsModel, node, pills, summary } = data;

  // The flow node is the chip, so edges meet its middle; the pills stack above it, outside the box xyflow measures.
  return (
    <Box data-emphasis={emphasis} data-preview-node={id} h="full" position="relative" w="full">
      {pills.length > 0 ? (
        <Stack bottom="100%" gap="1" insetStart="0" pb="1" position="absolute" w="full">
          {pills.map((pill) => (
            <Box
              key={pill.nodeId}
              alignSelf="flex-start"
              bg="accent.subtle"
              borderColor="accent.muted"
              borderWidth="1px"
              color="accent.fg"
              data-preview-pill={pill.nodeId}
              fontSize="sm"
              h="5.5"
              lineHeight="1.25rem"
              maxW="full"
              px="2"
              rounded="full"
              truncate
            >
              <PillLabel pill={pill} />
            </Box>
          ))}
        </Stack>
      ) : null}
      <Stack
        bg={node.isUnknown ? 'bg.error' : 'bg.panel'}
        borderColor={node.isUnknown ? 'border.error' : node.isResult ? 'fg.muted' : 'border.emphasized'}
        borderWidth={node.isResult ? '2px' : '1px'}
        css={emphasis ? CHIP_EMPHASIS[emphasis] : undefined}
        data-preview-chip=""
        gap="0.5"
        h={`${NODE_HEIGHT}px`}
        justify="center"
        position="relative"
        px="2.5"
        rounded="lg"
        w="full"
      >
        <Handle isConnectable={false} position={Position.Left} style={handleStyle} type="target" />
        <Handle isConnectable={false} position={Position.Right} style={handleStyle} type="source" />
        <Text
          color={node.isUnknown ? 'fg.error' : 'fg'}
          fontFamily={node.title === node.type ? 'mono' : undefined}
          fontSize="md"
          fontWeight="600"
          lineHeight="1.2"
          truncate
        >
          {node.title}
        </Text>
        {node.isUnknown || summary || node.isExposed || needsModel || node.isResult ? (
          <HStack color={node.isUnknown ? 'fg.error' : 'fg.muted'} fontSize="xs" gap="1" lineHeight="1.2" minW="0">
            {node.isExposed ? <MarkerIcon marker="exposed" /> : null}
            {needsModel ? <MarkerIcon marker="needsModel" /> : null}
            {node.isResult ? <MarkerIcon marker="result" /> : null}
            <Text fontFamily={node.isUnknown && node.title !== node.type ? 'mono' : undefined} truncate>
              {summary}
            </Text>
          </HStack>
        ) : null}
      </Stack>
    </Box>
  );
});

const PreviewEdge = memo(function PreviewEdge({
  id,
  data,
  sourcePosition,
  sourceX,
  sourceY,
  targetPosition,
  targetX,
  targetY,
}: EdgeProps<PreviewFlowEdge>) {
  const emphasis = useEmphasis(id);
  const [path] = getBezierPath({
    sourcePosition,
    sourceX,
    sourceY,
    targetPosition,
    targetX,
    targetY: targetY + (data?.entryOffset ?? 0),
  });
  const isLoopLinkage = data?.isLoopLinkage === true;

  return (
    <path
      className="react-flow__edge-path"
      d={path}
      data-emphasis={emphasis}
      data-preview-edge={id}
      data-rerouted={data?.isRerouted ? '' : undefined}
      fill="none"
      stroke={isLoopLinkage ? LOOP_LINKAGE_STROKE : undefined}
      strokeDasharray={isLoopLinkage ? '6 4' : data?.isRerouted ? '4 4' : undefined}
    />
  );
});

const nodeTypes: NodeTypes = { chip: PreviewChip };
const edgeTypes: EdgeTypes = { preview: PreviewEdge };

const ZoomControls = ({ onFit }: { onFit: () => void }) => {
  const { t } = useTranslation();
  const { zoomIn, zoomOut } = useReactFlow();
  const zoom = useStore((state) => state.transform[2]);
  const handleZoomIn = useCallback(() => void zoomIn({ duration: motion(ZOOM_STEP_DURATION) }), [zoomIn]);
  const handleZoomOut = useCallback(() => void zoomOut({ duration: motion(ZOOM_STEP_DURATION) }), [zoomOut]);

  return (
    <Box bottom="3" data-preview-overlay="" insetStart="3" position="absolute" zIndex={OVERLAY_Z}>
      <HStack
        bg="bg.panel"
        borderColor="border.subtle"
        borderWidth="1px"
        gap="0.5"
        p="0.5"
        role="group"
        aria-label={t('workflowLibrary.preview.zoomControls')}
        rounded="md"
        shadow="sm"
      >
        <Tooltip content={t('workflowLibrary.preview.zoomOut')}>
          <IconButton
            aria-label={t('workflowLibrary.preview.zoomOut')}
            disabled={zoom <= MIN_ZOOM}
            size="xs"
            variant="ghost"
            onClick={handleZoomOut}
          >
            <MinusIcon />
          </IconButton>
        </Tooltip>
        <Text fontSize="md" fontVariantNumeric="tabular-nums" textAlign="center" w="10">
          {t('workflowLibrary.preview.zoomPercent', { percent: Math.round(zoom * 100) })}
        </Text>
        <Tooltip content={t('workflowLibrary.preview.zoomIn')}>
          <IconButton
            aria-label={t('workflowLibrary.preview.zoomIn')}
            disabled={zoom >= MAX_ZOOM}
            size="xs"
            variant="ghost"
            onClick={handleZoomIn}
          >
            <PlusIcon />
          </IconButton>
        </Tooltip>
        <Button size="xs" variant="ghost" onClick={onFit}>
          {t('workflowLibrary.preview.fit')}
        </Button>
      </HStack>
    </Box>
  );
};

const Legend = ({ exposed, needsModel, result }: { exposed: boolean; needsModel: boolean; result: boolean }) => {
  const { t } = useTranslation();

  if (!exposed && !needsModel && !result) {
    return null;
  }

  return (
    <Box data-preview-overlay="" insetStart="3" position="absolute" top="3" zIndex={OVERLAY_Z}>
      <HStack
        aria-label={t('workflowLibrary.preview.legend')}
        as="ul"
        color="fg.muted"
        data-preview-legend=""
        fontSize="sm"
        gap="3.5"
        listStyleType="none"
        role="list"
      >
        {exposed ? (
          <HStack as="li" gap="1.5">
            <MarkerIcon marker="exposed" />
            {t('workflowLibrary.preview.youSetThis')}
          </HStack>
        ) : null}
        {needsModel ? (
          <HStack as="li" gap="1.5">
            <MarkerIcon marker="needsModel" />
            {t('workflowLibrary.preview.needsModel')}
          </HStack>
        ) : null}
        {result ? (
          <HStack as="li" gap="1.5">
            <MarkerIcon marker="result" />
            {t('workflowLibrary.preview.result')}
          </HStack>
        ) : null}
      </HStack>
    </Box>
  );
};

/** A From or To chip: selects that neighbour and brings it into view. */
const NeighbourChip = ({
  nodeId,
  title,
  onShowNode,
}: {
  nodeId: string;
  title: string;
  onShowNode: (nodeId: string) => void;
}) => {
  const handleClick = useCallback(() => onShowNode(nodeId), [nodeId, onShowNode]);

  return (
    <Button h="5" maxW="full" px="2" rounded="full" size="xs" variant="outline" onClick={handleClick}>
      <Text as="span" truncate>
        {title}
      </Text>
    </Button>
  );
};

const LAST_FIELD_ROW = { borderBottomWidth: '0' } as const;
const NO_RECTS: readonly never[] = [];

interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

const overlapArea = (a: Rect, b: Rect) =>
  Math.max(0, Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left)) *
  Math.max(0, Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top));

/**
 * Covering its own node, even a strip of it (a card clamped into the stage), costs far more than covering the rest: the
 * card describes that node.
 */
const OWN_NODE_WEIGHT = 50;

/**
 * Where a node's card goes: beside the node, right or left, else below or above it, whichever covers the least of the
 * drawn nodes (its own node most of all) and of the stage's controls, kept inside the stage.
 */
const placeCard = (
  layout: WorkflowPreviewLayout,
  nodeId: string,
  [x, y, zoom]: readonly [number, number, number],
  width: number,
  height: number,
  measuredHeight: number | null,
  controls: readonly Rect[]
) => {
  const toScreen = (box: ReturnType<typeof getNodeBox>) => ({
    height: box.height * zoom,
    left: box.x * zoom + x,
    top: box.y * zoom + y,
    width: box.width * zoom,
  });
  // Beside the chip or below it; above it goes above its pills too, which the card must not cover either.
  const chip = toScreen(getChipBox(layout, nodeId));
  const node = toScreen(getNodeBox(layout, nodeId));
  const cardHeight = Math.min(measuredHeight ?? CARD_ESTIMATED_HEIGHT, height - CARD_MARGIN * 2);
  const clamp = (left: number, top: number) => ({
    height: cardHeight,
    left: Math.max(CARD_MARGIN, Math.min(left, width - CARD_MARGIN - CARD_WIDTH)),
    top: Math.max(CARD_MARGIN, Math.min(top, height - CARD_MARGIN - cardHeight)),
    width: CARD_WIDTH,
  });
  const candidates = [
    clamp(chip.left + chip.width + CARD_GAP, chip.top),
    clamp(chip.left - CARD_GAP - CARD_WIDTH, chip.top),
    clamp(chip.left, chip.top + chip.height + CARD_GAP),
    clamp(chip.left, node.top - CARD_GAP - cardHeight),
  ];
  const others = layout.nodeIds.filter((id) => id !== nodeId).map((id) => toScreen(getNodeBox(layout, id)));
  const cost = (card: (typeof candidates)[number]) =>
    overlapArea(card, node) * OWN_NODE_WEIGHT +
    controls.reduce((sum, control) => sum + overlapArea(card, control) * 2, 0) +
    others.reduce((sum, other) => sum + overlapArea(card, other), 0);

  return candidates.reduce((best, candidate) => (cost(candidate) < cost(best) ? candidate : best));
};

const NodeCard = ({
  fieldSources,
  layout,
  model,
  needsModel,
  node,
  onClose,
  onShowNode,
  ref,
}: {
  fieldSources: WorkflowPreviewFieldSources | undefined;
  layout: WorkflowPreviewLayout;
  model: WorkflowPreviewModel;
  needsModel: boolean;
  /** A node the layout draws. */
  node: WorkflowPreviewNode;
  onClose: () => void;
  onShowNode: (nodeId: string) => void;
  ref: RefObject<HTMLDivElement | null>;
}) => {
  const { t } = useTranslation();
  const titleId = useId();
  const transform = useStore((state) => state.transform);
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);
  const nodeId = node.id;
  const neighbours = layout.neighbours[nodeId] ?? { sources: [], targets: [] };
  // Placed for its real height and the controls' boxes once measured (before paint), so a long note does not run over
  // the node and the card keeps clear of the controls where it can. A live source's new values measure again.
  const [measured, setMeasured] = useState<{ controls: readonly Rect[]; height: number; nodeId: string } | null>(null);
  const isMeasured = measured?.nodeId === nodeId;
  const { left, top } = placeCard(
    layout,
    nodeId,
    transform,
    width,
    height,
    isMeasured ? measured.height : null,
    isMeasured ? measured.controls : NO_RECTS
  );

  useLayoutEffect(() => {
    const card = ref.current;
    const stage = card?.parentElement;

    if (!card || !stage) {
      return;
    }

    const origin = stage.getBoundingClientRect();
    const controls = [...stage.querySelectorAll('[data-preview-overlay]')].map((control) => {
      const box = control.getBoundingClientRect();

      return { height: box.height, left: box.left - origin.left, top: box.top - origin.top, width: box.width };
    });

    setMeasured({ controls, height: card.scrollHeight, nodeId: node.id });
  }, [node, ref]);

  const neighbourChips = (ids: readonly string[]) => (
    <HStack flexWrap="wrap" gap="1">
      {ids.map((id) => (
        <NeighbourChip key={id} nodeId={id} title={model.nodes.get(id)?.title ?? id} onShowNode={onShowNode} />
      ))}
    </HStack>
  );

  return (
    <Stack
      ref={ref}
      aria-labelledby={titleId}
      bg="bg.panel"
      borderColor="border.emphasized"
      borderWidth="1px"
      className="nowheel nopan nodrag"
      data-preview-node-card={nodeId}
      gap="2.5"
      left={`${left}px`}
      maxH={`${height - top - CARD_MARGIN}px`}
      overflowY="auto"
      p="3"
      position="absolute"
      role="dialog"
      rounded="lg"
      shadow="lg"
      tabIndex={-1}
      top={`${top}px`}
      w={`${CARD_WIDTH}px`}
      zIndex={CARD_Z}
    >
      <HStack align="flex-start" gap="2">
        <Stack flex="1" gap="0.5" minW="0">
          <Text fontSize="lg" fontWeight="600" id={titleId} overflowWrap="anywhere">
            {node.title}
          </Text>
          <Text color="fg.muted" fontFamily="mono" fontSize="xs" overflowWrap="anywhere">
            {node.type}
          </Text>
        </Stack>
        <Tooltip content={t('common.close')}>
          <IconButton aria-label={t('common.close')} size="xs" variant="ghost" onClick={onClose}>
            <XIcon />
          </IconButton>
        </Tooltip>
      </HStack>
      {node.isUnknown ? (
        <Text color="fg.error" fontSize="md">
          {t('workflowLibrary.preview.unknownNode')}
        </Text>
      ) : null}
      {needsModel ? (
        <HStack color="fg.warning" fontSize="md" gap="1.5">
          <MarkerIcon marker="needsModel" />
          {t('workflowLibrary.preview.needsModelNote')}
        </HStack>
      ) : null}
      {node.notes ? (
        <Text color="fg.muted" fontSize="md" overflowWrap="anywhere" whiteSpace="pre-wrap">
          {node.notes}
        </Text>
      ) : null}
      {node.fields.length > 0 ? (
        <Stack as="dl" gap="0">
          {node.fields.map((field) => {
            const source = fieldSources?.describe(nodeId, field.name) ?? null;

            return (
              <HStack
                key={field.name}
                align="flex-start"
                borderBottomWidth="1px"
                borderColor="border.subtle"
                fontSize="md"
                gap="2.5"
                justify="space-between"
                py="1"
                _last={LAST_FIELD_ROW}
              >
                <HStack as="dt" color={field.isExposed ? 'fg' : 'fg.muted'} gap="1.5" minW="0">
                  {field.isExposed ? <MarkerIcon marker="exposed" /> : null}
                  <Text truncate>{field.label}</Text>
                  {field.isExposed ? <Text srOnly>{t('workflowLibrary.preview.youSetThisInline')}</Text> : null}
                </HStack>
                <Stack as="dd" align="flex-end" gap="0" maxW="55%" minW="0">
                  <Text maxW="full" textAlign="end" truncate>
                    {formatWorkflowPreviewValue(field.value, t)}
                  </Text>
                  {source && fieldSources ? (
                    <Button
                      color="accent.fg"
                      data-preview-field-source=""
                      fontSize="sm"
                      fontWeight="400"
                      h="auto"
                      maxW="full"
                      minW="0"
                      px="0"
                      size="xs"
                      variant="plain"
                      onClick={fieldSources.onSelect}
                    >
                      <Text as="span" truncate>
                        {source}
                      </Text>
                    </Button>
                  ) : null}
                </Stack>
              </HStack>
            );
          })}
        </Stack>
      ) : null}
      {neighbours.sources.length > 0 ? (
        <Stack gap="1">
          <Text color="fg.muted" fontSize="sm">
            {t('workflowLibrary.preview.from')}
          </Text>
          {neighbourChips(neighbours.sources)}
        </Stack>
      ) : null}
      {neighbours.targets.length > 0 ? (
        <Stack gap="1">
          <Text color="fg.muted" fontSize="sm">
            {t('workflowLibrary.preview.to')}
          </Text>
          {neighbourChips(neighbours.targets)}
        </Stack>
      ) : null}
    </Stack>
  );
};

const StyledMiniMap = chakra(MiniMap);

/**
 * The chip's second line: what the node's type says about its values, else where its model stands, else its first
 * exposed input with a short value. An unknown node shows its type id: as its title when unlabelled, else here.
 */
const getChipSummary = (node: WorkflowPreviewNode, needsModel: boolean, t: TFunction): string | null => {
  if (node.isUnknown) {
    return node.title === node.type ? t('workflowLibrary.preview.nodePackMissing') : node.type;
  }

  const typed = getWorkflowNodeSummary(node.values, t);

  if (typed) {
    return typed;
  }

  if (needsModel) {
    return t('workflowLibrary.preview.modelNotInstalled');
  }

  if (node.modelRequirements.some((requirement) => requirement.kind === 'slot')) {
    return t('workflowLibrary.preview.modelToPick');
  }

  return node.exposedSummary
    ? `${node.exposedSummary.label} ${formatWorkflowPreviewValue(node.exposedSummary.value, t)}`
    : null;
};

/** The layout with its rows spread `stretch` times further apart, keeping each chip's height and its pills. */
const stretchRows = (layout: WorkflowPreviewLayout, stretch: number): WorkflowPreviewLayout =>
  stretch === 1
    ? layout
    : {
        ...layout,
        positions: Object.fromEntries(
          Object.entries(layout.positions).map(([id, position]) => [id, { x: position.x, y: position.y * stretch }])
        ),
      };

/**
 * How far to spread a simplified graph's rows: a wide, short layered graph fitted to the stage would leave most of its
 * height empty and its edges bunched, so rows spread until the graph takes `ROW_SPREAD_SHARE` of it.
 */
const getRowStretch = (layout: WorkflowPreviewLayout, zoom: number, stageHeight: number) => {
  const tops = layout.nodeIds.map((id) => layout.positions[id]!.y);
  const highest = Math.min(...tops);
  const span = Math.max(...tops) - highest;

  // Spread, the chips span `span * stretch`, below the pills over the highest one and above the lowest one's height.
  return span <= 0
    ? 1
    : Math.min(MAX_ROW_STRETCH, Math.max(1, ((stageHeight * ROW_SPREAD_SHARE) / zoom - NODE_HEIGHT - highest) / span));
};

const getNodeName = (
  node: WorkflowPreviewNode,
  needsModel: boolean,
  pills: readonly WorkflowPreviewPill[],
  t: TFunction
) =>
  [
    node.title,
    node.isUnknown ? t('workflowLibrary.preview.unknownNodeMarker') : null,
    node.isExposed ? t('workflowLibrary.preview.youSetThisInline') : null,
    needsModel ? t('workflowLibrary.preview.needsModelInline') : null,
    node.isResult ? t('workflowLibrary.preview.resultInline') : null,
    pills.length > 0
      ? t('workflowLibrary.preview.withHelpers', {
          helpers: pills
            .map((pill) => (pill.value ? `${pill.label} ${formatWorkflowPreviewValue(pill.value, t)}` : pill.label))
            .join(', '),
        })
      : null,
  ]
    .filter((part) => part !== null)
    .join(', ');

/** `next`, with each item that equals its predecessor of the same id replaced by that predecessor. */
const keepUnchanged = <Item extends { id: string }>(
  previous: readonly Item[],
  next: Item[],
  isSame: (a: Item, b: Item) => boolean
): Item[] => {
  const byId = new Map(previous.map((item) => [item.id, item]));

  return next.map((item) => {
    const prior = byId.get(item.id);

    return prior && isSame(prior, item) ? prior : item;
  });
};

/** The same items, in the same order. */
const isSameItems = <Item,>(a: readonly Item[], b: readonly Item[]) =>
  a.length === b.length && a.every((item, index) => item === b[index]);

const isSameChip = (a: ChipNode, b: ChipNode) =>
  a.ariaLabel === b.ariaLabel &&
  a.position.x === b.position.x &&
  a.position.y === b.position.y &&
  a.width === b.width &&
  a.height === b.height &&
  a.data.node === b.data.node &&
  a.data.pills === b.data.pills &&
  a.data.needsModel === b.data.needsModel &&
  a.data.summary === b.data.summary;

const isSameEdge = (a: PreviewFlowEdge, b: PreviewFlowEdge) =>
  a.ariaLabel === b.ariaLabel &&
  a.source === b.source &&
  a.target === b.target &&
  a.data?.isLoopLinkage === b.data?.isLoopLinkage &&
  a.data?.isRerouted === b.data?.isRerouted &&
  a.data?.entryOffset === b.data?.entryOffset;

/**
 * Where each edge meets its target, as an offset from the chip's middle: a chip fed by several nodes takes them a few
 * px apart, top to bottom as their sources sit, rather than at one point, so they arrive without crossing there.
 */
const getEntryOffsets = (layout: WorkflowPreviewLayout): Map<string, number> => {
  const byTarget = new Map<string, WorkflowPreviewLayout['edges'][number][]>();

  for (const edge of layout.edges) {
    byTarget.set(edge.target, [...(byTarget.get(edge.target) ?? []), edge]);
  }

  const offsets = new Map<string, number>();
  const sourceY = (edge: WorkflowPreviewLayout['edges'][number]) => layout.positions[edge.source]?.y ?? 0;

  for (const entering of byTarget.values()) {
    const step = entering.length < 2 ? 0 : Math.min(ENTRY_STEP, ENTRY_SPAN / (entering.length - 1));

    [...entering]
      .sort((a, b) => sourceY(a) - sourceY(b))
      .forEach((edge, index) => offsets.set(edge.id, (index - (entering.length - 1) / 2) * step));
  }

  return offsets;
};

/** The widest of the titles, set as a chip sets them, in px. */
const measureTitlesWidth = (titles: readonly string[], stage: HTMLElement) => {
  const context = document.createElement('canvas').getContext('2d');

  if (!context) {
    return 0;
  }

  const rootPx = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

  context.font = `600 ${TITLE_REM * rootPx}px ${getComputedStyle(stage).fontFamily}`;

  return Math.max(0, ...titles.map((title) => context.measureText(title).width));
};

const GraphView = ({
  fieldSources,
  initialNodeId = null,
  focusInitialCard = true,
  model: sourceModel,
  needsModel,
  ref,
  relayout,
  onCardChange,
  onMatchCountChange,
}: WorkflowDocumentGraphProps) => {
  const { t } = useTranslation();
  /**
   * The drawing's chip width, chosen as it opens (`handleInit`), when the stage's size is known; the drawing is that
   * model laid out at that width, again for each new value of a live source.
   */
  const [chipWidth, setChipWidth] = useState(sourceModel.all.nodeWidth);
  const [laidOut, setLaidOut] = useState({ model: sourceModel, source: sourceModel, width: chipWidth });
  let model = laidOut.model;
  if (laidOut.source !== sourceModel || laidOut.width !== chipWidth) {
    model =
      chipWidth === sourceModel.all.nodeWidth
        ? sourceModel
        : relayout(getWorkflowPreviewMetrics(chipWidth), laidOut.width === chipWidth ? laidOut.model : null);
    setLaidOut({ model, source: sourceModel, width: chipWidth });
  }
  // A node asked for that this document does not have opens nothing.
  const startNodeId = initialNodeId !== null && model.nodes.has(initialNodeId) ? initialNodeId : null;
  const themeId = useWorkflowPreferencesSelector((preferences) => preferences.themeId);
  const flow = useReactFlow();
  const [emphasis] = useState(createKeyedTransientStore<string, Emphasis>);
  // Opens simplified, unless the node asked for is a folded helper or hidden plumbing: then everything shows.
  const [isSimplified, setIsSimplified] = useState(
    model.simplified !== null && (startNodeId === null || model.simplified.nodeIds.includes(startNodeId))
  );
  const [isClipped, setIsClipped] = useState(false);
  const [rowStretch, setRowStretch] = useState(1);
  const [cardNodeId, setCardNodeIdState] = useState<string | null>(startNodeId);
  // Called from handlers only: the host hears every change of card, as it happens.
  const setCardNodeId = useCallback(
    (nodeId: string | null) => {
      setCardNodeIdState(nodeId);
      onCardChange?.(nodeId);
    },
    [onCardChange]
  );
  const [findState, setFindState] = useState<{ cursor: number; matches: readonly string[]; query: string }>({
    cursor: -1,
    matches: [],
    query: '',
  });
  const containerRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const baseLayout = (isSimplified ? model.simplified : null) ?? model.all;
  const layout = useMemo(() => stretchRows(baseLayout, rowStretch), [baseLayout, rowStretch]);

  const builtNodes = useMemo(
    () =>
      layout.nodeIds.map((id): ChipNode => {
        const node = model.nodes.get(id)!;
        const pills = layout.pills[id] ?? NO_PILLS;
        const nodeNeedsModel = needsModel.has(id);

        return {
          ariaLabel: getNodeName(node, nodeNeedsModel, pills, t),
          ariaRole: 'button',
          connectable: false,
          data: { needsModel: nodeNeedsModel, node, pills, summary: getChipSummary(node, nodeNeedsModel, t) },
          draggable: false,
          height: NODE_HEIGHT,
          id,
          position: layout.positions[id]!,
          selectable: false,
          type: 'chip',
          width: layout.nodeWidth,
        };
      }),
    [layout, model, needsModel, t]
  );
  const builtEdges = useMemo(() => {
    const entryOffsets = getEntryOffsets(layout);

    return layout.edges.map((edge): PreviewFlowEdge => ({
      ariaLabel: t('workflowLibrary.preview.edgeLabel', {
        source: model.nodes.get(edge.source)!.title,
        target: model.nodes.get(edge.target)!.title,
      }),
      data: {
        entryOffset: entryOffsets.get(edge.id) ?? 0,
        isLoopLinkage: edge.isLoopLinkage,
        isRerouted: edge.isRerouted,
      },
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'preview',
    }));
  }, [layout, model, t]);
  /**
   * The chips and edges handed to the flow keep their identity while nothing they show changed, so a live source's
   * recompile re-renders only the chips whose values, pills or markers did (the model keeps unchanged nodes' identity).
   */
  const [drawn, setDrawn] = useState({ edges: builtEdges, nodes: builtNodes });
  let { edges, nodes } = drawn;
  const keptNodes = keepUnchanged(drawn.nodes, builtNodes, isSameChip);
  const keptEdges = keepUnchanged(drawn.edges, builtEdges, isSameEdge);
  if (!isSameItems(keptNodes, drawn.nodes) || !isSameItems(keptEdges, drawn.edges)) {
    nodes = keptNodes;
    edges = keptEdges;
    setDrawn({ edges, nodes });
  }
  const ariaLabelConfig = useMemo(
    () => ({
      'minimap.ariaLabel': t('workflowLibrary.preview.minimap'),
      'node.a11yDescription.default': t('workflowLibrary.preview.nodeHint'),
      'node.a11yDescription.keyboardDisabled': t('workflowLibrary.preview.nodeHint'),
    }),
    [t]
  );

  /** Writes one emphasis map over the previous one, notifying only the nodes and edges that change. */
  const applyEmphasis = useCallback(
    (next: ReadonlyMap<string, Emphasis>) => {
      for (const [key] of emphasis.entries()) {
        if (!next.has(key)) {
          emphasis.delete(key);
        }
      }

      for (const [key, value] of next) {
        emphasis.set(key, value);
      }
    },
    [emphasis]
  );
  const emphasiseSelection = useCallback(
    (nodeId: string, shown: WorkflowPreviewLayout) => {
      const next = new Map<string, Emphasis>([[nodeId, 'selected']]);

      for (const edge of shown.edges) {
        if (edge.source === nodeId || edge.target === nodeId) {
          next.set(edge.id, 'related');
          next.set(edge.source === nodeId ? edge.target : edge.source, 'related');
        }
      }

      next.set(nodeId, 'selected');
      applyEmphasis(next);
    },
    [applyEmphasis]
  );
  const emphasiseMatches = useCallback(
    (matches: readonly string[], cursor: number) =>
      applyEmphasis(new Map(matches.map((id, index) => [id, index === cursor ? 'current' : 'match'] as const))),
    [applyEmphasis]
  );

  const centreOn = useCallback(
    (nodeId: string, shown: WorkflowPreviewLayout) => {
      const box = getNodeBox(shown, nodeId);

      void flow.setCenter(box.x + box.width / 2, box.y + box.height / 2, {
        duration: motion(REVEAL_DURATION),
        zoom: Math.max(flow.getZoom(), getReadableZoom()),
      });
    },
    [flow]
  );

  const openCard = useCallback(
    (nodeId: string) => {
      flushSync(() => setCardNodeId(nodeId));
      emphasiseSelection(nodeId, layout);
      cardRef.current?.focus({ preventScroll: true });
    },
    [emphasiseSelection, layout, setCardNodeId]
  );
  const closeCard = useCallback(
    (returnFocus: boolean) => {
      const nodeId = cardNodeId;

      if (nodeId === null) {
        return false;
      }

      flushSync(() => setCardNodeId(null));
      emphasiseMatches(findState.matches, findState.cursor);

      if (returnFocus) {
        containerRef.current
          ?.querySelector<HTMLElement>(`.react-flow__node[data-id="${CSS.escape(nodeId)}"]`)
          ?.focus({ preventScroll: true });
      }

      return true;
    },
    [cardNodeId, emphasiseMatches, findState, setCardNodeId]
  );
  const showNode = useCallback(
    (nodeId: string) => {
      centreOn(nodeId, layout);
      openCard(nodeId);
    },
    [centreOn, layout, openCard]
  );

  /**
   * Opens a layout at its fitted or readable zoom, spreading a simplified graph's rows to the stage's height first, and
   * returns the layout as drawn.
   */
  const applyOpeningViewport = useCallback(
    (base: WorkflowPreviewLayout, instance: Pick<ReactFlowInstance, 'setViewport'>, isFullView: boolean) => {
      const element = containerRef.current;

      if (!element) {
        return base;
      }

      const area = measureStageArea(element);
      const unstretched = getOpeningViewport(base, area);
      const stretch =
        isFullView || unstretched.isClipped
          ? 1
          : getRowStretch(base, unstretched.viewport.zoom, area.height - area.top - area.bottom);
      const shown = stretchRows(base, stretch);
      const opening = stretch === 1 ? unstretched : getOpeningViewport(shown, area);

      setRowStretch(stretch);
      setIsClipped(opening.isClipped);
      void instance.setViewport(opening.viewport);

      return shown;
    },
    []
  );
  /**
   * The chip width for this drawing: wide enough for its longest title where the stage leaves the room (the graph
   * still fitted at a legible zoom), one width for every chip so columns stay aligned.
   */
  const chooseChipWidth = useCallback(
    (stage: HTMLElement) => {
      const area = measureStageArea(stage);
      const layoutOf = (laid: WorkflowPreviewModel) => (isSimplified ? laid.simplified : null) ?? laid.all;
      const titles = layoutOf(model).nodeIds.map((id) => model.nodes.get(id)!.title);
      const readable = getReadableZoom();

      return chooseWorkflowPreviewChipWidth(
        measureTitlesWidth(titles, stage) + CHIP_TITLE_INSET,
        (width) =>
          getFittedViewport(getLayoutBounds(layoutOf(relayout(getWorkflowPreviewMetrics(width), null))), area).zoom >=
          readable
      );
    },
    [isSimplified, model, relayout]
  );
  const handleInit = useCallback(
    (instance: ReactFlowInstance<ChipNode, PreviewFlowEdge>) => {
      const stage = containerRef.current;
      const width = stage ? chooseChipWidth(stage) : chipWidth;
      const opening = width === chipWidth ? model : relayout(getWorkflowPreviewMetrics(width), null);
      const openingBase = (isSimplified ? opening.simplified : null) ?? opening.all;

      setChipWidth(width);
      const shown = applyOpeningViewport(openingBase, instance, openingBase === opening.all);

      if (startNodeId !== null && shown.positions[startNodeId]) {
        const box = getNodeBox(shown, startNodeId);

        emphasiseSelection(startNodeId, shown);
        onCardChange?.(startNodeId);
        void instance.setCenter(box.x + box.width / 2, box.y + box.height / 2, {
          zoom: Math.max(instance.getZoom(), getReadableZoom()),
        });

        const active = document.activeElement;

        if (focusInitialCard || active === null || active === document.body) {
          cardRef.current?.focus({ preventScroll: true });
        }
      }
    },
    [
      applyOpeningViewport,
      chipWidth,
      chooseChipWidth,
      emphasiseSelection,
      focusInitialCard,
      isSimplified,
      model,
      onCardChange,
      relayout,
      startNodeId,
    ]
  );
  const toggleSimplified = useCallback(() => {
    const next = !isSimplified;
    const shown = (next ? model.simplified : null) ?? model.all;
    const matches = findWorkflowPreviewNodes(model, shown, findState.query);

    setIsSimplified(next);
    setCardNodeId(null);
    setFindState({ cursor: -1, matches, query: findState.query });
    emphasiseMatches(matches, -1);
    onMatchCountChange?.(matches.length);
    applyOpeningViewport(shown, flow, shown === model.all);
  }, [
    applyOpeningViewport,
    emphasiseMatches,
    findState.query,
    flow,
    isSimplified,
    model,
    onMatchCountChange,
    setCardNodeId,
  ]);
  // Fitted, the whole graph is in view: the minimap has nothing left to show.
  const handleFit = useCallback(() => {
    const element = containerRef.current;

    if (!element) {
      return;
    }

    setIsClipped(false);
    void flow.setViewport(getFittedViewport(getLayoutBounds(layout), measureStageArea(element)), {
      duration: motion(REVEAL_DURATION),
    });
  }, [flow, layout]);

  useImperativeHandle(
    ref,
    () => ({
      closeNodeCard: () => closeCard(true),
      find: (query) => {
        const matches = findWorkflowPreviewNodes(model, layout, query);

        setCardNodeId(null);
        setFindState({ cursor: -1, matches, query });
        emphasiseMatches(matches, -1);
        onMatchCountChange?.(matches.length);
      },
      revealNextMatch: () => {
        if (findState.matches.length === 0) {
          return;
        }

        const cursor = (findState.cursor + 1) % findState.matches.length;

        setFindState({ ...findState, cursor });
        emphasiseMatches(findState.matches, cursor);
        centreOn(findState.matches[cursor]!, layout);
      },
    }),
    [centreOn, closeCard, emphasiseMatches, findState, layout, model, onMatchCountChange, setCardNodeId]
  );

  const handleNodeClick = useCallback<NodeMouseHandler<ChipNode>>((_event, node) => openCard(node.id), [openCard]);
  const handlePaneClick = useCallback(() => void closeCard(false), [closeCard]);
  const closeCardFromButton = useCallback(() => void closeCard(true), [closeCard]);
  // The node wrappers are xyflow's focusable buttons; Enter and Space on one open its card.
  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const target = event.target as HTMLElement;

      if ((event.key === 'Enter' || event.key === ' ') && target.classList.contains('react-flow__node')) {
        event.preventDefault();

        if (target.dataset.id) {
          openCard(target.dataset.id);
        }
      }
    },
    [openCard]
  );

  const cardNode = cardNodeId !== null && layout.positions[cardNodeId] ? model.nodes.get(cardNodeId) : undefined;
  const shownNodes = layout.nodeIds.map((id) => model.nodes.get(id)!);
  const foldedCount = model.nodes.size - layout.nodeIds.length;

  return (
    <EmphasisContext value={emphasis}>
      <Box
        ref={containerRef}
        bg="bg.inset"
        css={CONTAINER_CSS}
        data-dimmed={cardNodeId !== null || findState.matches.length > 0 ? '' : undefined}
        h="full"
        position="relative"
        w="full"
        onKeyDown={handleKeyDown}
      >
        {/* The stage's controls come before the nodes, so the keyboard reaches them without passing every node. */}
        <Legend
          exposed={shownNodes.some((node) => node.isExposed)}
          needsModel={shownNodes.some((node) => needsModel.has(node.id))}
          result={shownNodes.some((node) => node.isResult)}
        />
        <ZoomControls onFit={handleFit} />
        <Box bottom="3" data-preview-overlay="" insetEnd="3" position="absolute" zIndex={OVERLAY_Z}>
          <Stack align="flex-end" gap="2">
            {isClipped ? (
              <StyledMiniMap
                ariaLabel={t('workflowLibrary.preview.minimap')}
                borderColor="border.subtle"
                borderWidth="1px"
                overflow="hidden"
                pannable
                rounded="md"
                shadow="sm"
                // Laid out in this panel, above the node count, rather than as a corner panel of its own.
                style={MINIMAP_STYLE}
                zoomable
              />
            ) : null}
            <HStack
              bg="bg.panel"
              borderColor="border.subtle"
              borderWidth="1px"
              data-preview-node-count=""
              gap="2.5"
              minH="8"
              ps="3"
              pe={model.simplified ? '1' : '3'}
              py="0.5"
              rounded="md"
              shadow="sm"
            >
              <Text color="fg.muted" fontSize="md">
                {!model.simplified
                  ? t('workflowLibrary.nodeCount', { count: model.nodes.size })
                  : isSimplified
                    ? `${t('workflowLibrary.preview.showingNodes', {
                        count: model.nodes.size,
                        shown: layout.nodeIds.length,
                      })} · ${t('workflowLibrary.preview.helpersFolded', { count: foldedCount })}`
                    : t('workflowLibrary.preview.showingAll', { count: model.nodes.size })}
              </Text>
              {model.simplified ? (
                <Button size="xs" variant="outline" onClick={toggleSimplified}>
                  {isSimplified
                    ? t('workflowLibrary.preview.showAll', { count: model.nodes.size })
                    : t('workflowLibrary.preview.simplify')}
                </Button>
              ) : null}
            </HStack>
          </Stack>
        </Box>
        <ReactFlow
          ariaLabelConfig={ariaLabelConfig}
          colorMode={getFlowColorMode(themeId)}
          deleteKeyCode={null}
          edges={edges}
          edgesFocusable={false}
          edgeTypes={edgeTypes}
          elementsSelectable={false}
          maxZoom={MAX_ZOOM}
          minZoom={MIN_ZOOM}
          multiSelectionKeyCode={null}
          nodes={nodes}
          nodesConnectable={false}
          nodesDraggable={false}
          nodesFocusable
          nodeTypes={nodeTypes}
          panActivationKeyCode={null}
          proOptions={proOptions}
          selectionKeyCode={null}
          style={flowStyle}
          zoomOnDoubleClick={false}
          onInit={handleInit}
          onNodeClick={handleNodeClick}
          onPaneClick={handlePaneClick}
        >
          <Background
            bgColor="var(--xy-background-color)"
            color="var(--wb-flow-grid)"
            gap={24}
            size={1.5}
            variant={BackgroundVariant.Dots}
          />
        </ReactFlow>
        {/* Over the controls, in their layer: the graph's own layer would paint it under them. */}
        {cardNode ? (
          <NodeCard
            ref={cardRef}
            fieldSources={fieldSources}
            layout={layout}
            model={model}
            needsModel={needsModel.has(cardNode.id)}
            node={cardNode}
            onClose={closeCardFromButton}
            onShowNode={showNode}
          />
        ) : null}
      </Box>
    </EmphasisContext>
  );
};

/** One provider per graph, so the controls and the card read this graph's viewport. */
export const WorkflowDocumentGraph = (props: WorkflowDocumentGraphProps) => (
  <ReactFlowProvider>
    <GraphView {...props} />
  </ReactFlowProvider>
);
