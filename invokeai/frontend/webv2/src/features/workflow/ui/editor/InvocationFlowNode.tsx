/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { FieldInputTemplate, FieldOutputTemplate, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState as NodeExecutionState } from '@features/workflow/ui/contracts';
import type { TFunction } from 'i18next';

import { Box, Checkbox, Field, Flex, HStack, Icon, IconButton, Input, Stack, Text } from '@chakra-ui/react';
import { getWorkflowFieldSeedMode, isSeedInputField } from '@features/workflow/graph';
import { FieldDescriptionPopover } from '@features/workflow/ui/fields/FieldDescriptionPopover';
import { WorkflowFieldInput, WorkflowFieldSnapshot } from '@features/workflow/ui/fields/WorkflowFieldInput';
import {
  getWorkflowNodeAppearance,
  getWorkflowNodeBodyProps,
  getWorkflowNodeHandleStyle,
  getWorkflowNodeHeaderProps,
  getWorkflowNodeShellProps,
  WORKFLOW_NODE_DENSITY,
  WORKFLOW_NODE_FIELD_CONTROL_ATTRIBUTE,
  WORKFLOW_NODE_FIELD_ROW_ATTRIBUTE,
  WorkflowNodeHeaderButton,
  WorkflowNodeInfoButton,
  WorkflowNodeStatusButton,
  useIsWorkflowImageExport,
  type WorkflowNodeChromeState,
  type WorkflowNodeStatus,
} from '@features/workflow/ui/nodeChrome';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  useOpenNodesManager,
  useWorkflowCapabilitiesSelector,
  useWorkflowNodeExecutionState,
} from '@features/workflow/ui/WorkflowUiContext';
import {
  areHiDiffusionRatiosAutomatic,
  getHiDiffusionRatioOrderIssue,
  getHiDiffusionRatioTemplate,
  isHiDiffusionNode,
  isHiDiffusionRatioField,
  CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX,
  cloneWorkflowFieldDefault,
  formatOutputFieldValue,
  getEffectiveWorkflowFieldDescription,
  getFieldTypeLabel,
  getNodeUpdateStatus,
  getOutputFieldNamesByScope,
  getWorkflowBatchGroupId,
  isWorkflowBatchNodeType,
  getOutputFieldRows,
  getWorkflowFieldIssue,
  isDirectInputField,
  isExposableField,
  isWorkflowFieldValueDefault,
  type WorkflowNodeUpdateStatus,
} from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Handle, Position, useStore, type NodeProps } from '@xyflow/react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  CpuIcon,
  FlaskConicalIcon,
  HammerIcon,
  LinkIcon,
  MinusIcon,
  PlusIcon,
  RotateCcwIcon,
  TriangleAlertIcon,
  ZapIcon,
  type LucideIcon,
} from 'lucide-react';
import {
  Component,
  lazy,
  memo,
  Suspense,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationFlowNode as InvocationFlowNodeType, InvocationNodeTemplateView } from './flowAdapters';
import type * as NodePanelsModule from './NodePanels';

import { getHandleTypeTooltip } from './handleTooltip';
import { getExecutionStatusText, getFieldIssueMessage, type InvocationNodeFieldIssue } from './nodeStatusText';

const NODE_WIDTH = '18rem';
/** Row padding-x in px; inline handles sit in the label rows, so they pull back out past it. */
const ROW_PADDING_X = 12;
/** A field port's centre, on its row's label line. */
const PORT_TOP = `calc(${WORKFLOW_NODE_DENSITY.labelLine} / 2)`;
/** Below this viewport zoom, field content renders as skeleton bars (ComfyUI-style) for performance/readability. */
export const CONTENT_VISIBILITY_ZOOM = 0.4;

/** True while the viewport is zoomed out far enough that field content is unreadable noise. */
const useIsZoomedOut = (): boolean => {
  const isWorkflowImageExport = useIsWorkflowImageExport();
  const isZoomedOut = useStore((state) => state.transform[2] < CONTENT_VISIBILITY_ZOOM);

  return isZoomedOut && !isWorkflowImageExport;
};

/** The node-level loading hint is only useful while a selected child signature is being fetched. */
export const shouldShowCallSavedWorkflowLoadingHint = (node: WorkflowInvocationNode): boolean =>
  node.data.type === 'call_saved_workflow' && node.data.callSavedWorkflowStatus === 'loading';

/** A callable child can be valid without exposing any fields; explain the empty body instead of showing a blank node. */
export const shouldShowCallSavedWorkflowNoExposedFieldsHint = (node: WorkflowInvocationNode): boolean => {
  if (node.data.type !== 'call_saved_workflow' || node.data.callSavedWorkflowStatus !== 'ready') {
    return false;
  }

  const workflowId = node.data.inputs.workflow_id?.value;
  const hasSelectedWorkflow = typeof workflowId === 'string' && workflowId.trim() !== '';
  const hasDynamicFields =
    Object.keys(node.data.dynamicInputTemplates ?? {}).length > 0 ||
    Object.keys(node.data.inputs).some((name) => name.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX));

  return hasSelectedWorkflow && !hasDynamicFields;
};

/** Static placeholder bar standing in for text/controls at far zoom. No animation — there may be hundreds. */
const SkeletonBar = ({ h = '2', w }: { h?: string; w?: string }) => <Box bg="bg.emphasized" h={h} rounded="sm" w={w} />;

/** A field that keeps its node from running, named as the node shows it. */
const HIDIFFUSION_RATIO_FIELDS = ['hidiffusion_t1_ratio', 'hidiffusion_t2_ratio'] as const;
/** Invoke's validation reports the ratio order on T2 (buildGraph); the status lists it once, there. */
const HIDIFFUSION_RATIO_REPORTED_FIELD = 'hidiffusion_t2_ratio';

const EMPTY_FIELD_ISSUES: InvocationNodeFieldIssue[] = [];

/** Every field that keeps the node from running, in row order; the node, its rows and its status panel share it. */
const getInvocationNodeFieldIssues = (
  node: WorkflowInvocationNode,
  inputTemplates: readonly FieldInputTemplate[],
  connectedFieldNames: ReadonlySet<string>
): InvocationNodeFieldIssue[] => {
  let issues: InvocationNodeFieldIssue[] | null = null;
  const ratioIssue = inputTemplates.some((template) => isHiDiffusionRatioField(template.name))
    ? getHiDiffusionRatioOrderIssue(node, connectedFieldNames)
    : null;

  for (const template of inputTemplates) {
    const instance = node.data.inputs[template.name];
    const label = instance?.label || template.title;

    // A ratio order problem stands in for the ratios' own checks, as it always has on these rows.
    if (ratioIssue && isHiDiffusionRatioField(template.name)) {
      if (template.name === HIDIFFUSION_RATIO_REPORTED_FIELD) {
        (issues ??= []).push({
          issue: { kind: 'invalidValue', reason: ratioIssue },
          label,
          markedFieldNames: HIDIFFUSION_RATIO_FIELDS,
          template,
        });
      }
      continue;
    }

    const issue = getWorkflowFieldIssue({
      isConnected: connectedFieldNames.has(template.name),
      template,
      value: instance?.value,
    });

    if (issue) {
      (issues ??= []).push({ issue, label, markedFieldNames: [template.name], template });
    }
  }

  return issues ?? EMPTY_FIELD_ISSUES;
};

const invalidByFlowData = new WeakMap<InvocationFlowNodeType['data'], boolean>();

/**
 * Whether a node's fields or type keep it from running, as its chrome shows it. Cached per flow-node data, which the
 * flow model rebuilds only when the node's document, template or connections change, so the minimap can ask on every
 * render.
 */
export const isInvocationFlowNodeInvalid = (data: InvocationFlowNodeType['data']): boolean => {
  const cached = invalidByFlowData.get(data);

  if (cached !== undefined) {
    return cached;
  }

  const invalid = data.template
    ? getInvocationNodeFieldIssues(
        data.documentNode,
        data.template.inputTemplates,
        new Set(data.connectedTargetHandles)
      ).length > 0
    : data.templateStatus === 'loaded';
  invalidByFlowData.set(data, invalid);

  return invalid;
};

/**
 * One place turns what is known about a node into its chrome and its status button. The chrome facts stay
 * independent so they compose; the status button shows the one that most needs attention: a live run, then a failed
 * one, then what blocks the next run, then a version change, then a completed run.
 */
export const getInvocationNodeState = ({
  execution,
  hasFieldIssues,
  isMissingTemplate = false,
  updateStatus,
}: {
  execution: NodeExecutionState | null;
  hasFieldIssues: boolean;
  isMissingTemplate?: boolean;
  /** Null for a node without a template. */
  updateStatus: WorkflowNodeUpdateStatus | null;
}): { chrome: Omit<WorkflowNodeChromeState, 'selected'>; status: WorkflowNodeStatus | null } => {
  const executionStatus = execution?.status ?? null;
  const isOutdated = updateStatus !== null && updateStatus !== 'current';

  return {
    chrome: {
      failed: executionStatus === 'failed',
      invalid: hasFieldIssues || isMissingTemplate,
      outdated: isOutdated,
      running: executionStatus === 'running',
    },
    status:
      executionStatus === 'running' || executionStatus === 'failed'
        ? executionStatus
        : hasFieldIssues
          ? 'invalid'
          : updateStatus === 'updatable'
            ? 'updatable'
            : isOutdated
              ? 'versionMismatch'
              : executionStatus,
  };
};

const NodeShell = ({
  chrome,
  children,
  selected,
}: {
  chrome?: Omit<WorkflowNodeChromeState, 'selected'>;
  children: React.ReactNode;
  selected: boolean;
}) => (
  <Box
    data-is-selected={selected}
    data-workflow-node-shell="true"
    position="relative"
    w={NODE_WIDTH}
    {...getWorkflowNodeShellProps({ ...chrome, selected })}
  >
    {children}
  </Box>
);

// The panel bodies load on first open; hovering or focusing a header button starts the load, so the panel is usually
// ready by the time its tooltip opens. A failed load is forgotten, so the next attempt imports again.
let loadedNodePanels: typeof NodePanelsModule | null = null;
let nodePanelsModule: Promise<typeof NodePanelsModule> | null = null;
const loadNodePanels = (): Promise<typeof NodePanelsModule> =>
  (nodePanelsModule ??= import('./NodePanels').then(
    (module) => (loadedNodePanels = module),
    (error: unknown) => {
      nodePanelsModule = null;
      throw error;
    }
  ));
const preloadNodePanels = () => void loadNodePanels().catch(() => {});

type RenderNodePanel = (module: typeof NodePanelsModule) => ReactNode;

/** A panel whose body failed to load says what its button says rather than taking the editor down. */
class NodePanelBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

/**
 * A header panel's body. Once the module has loaded it renders directly. Before that, each mount owns its own lazy
 * load (tooltips mount their content per open), so a load that failed is tried again on the next open, where a
 * module-level `lazy` would keep its first failure for the page's lifetime. While loading, and if it fails, the panel
 * shows the button's own label, so the state stays readable.
 */
const DeferredNodePanel = ({ label, render }: { label: string; render: RenderNodePanel }) => {
  const [LazyPanel] = useState(() =>
    loadedNodePanels
      ? null
      : lazy(() =>
          loadNodePanels().then((module) => ({
            default: ({ renderPanel }: { renderPanel: RenderNodePanel }) => renderPanel(module),
          }))
        )
  );
  const fallback = <Text>{label}</Text>;

  return (
    <NodePanelBoundary fallback={fallback}>
      {LazyPanel ? (
        <Suspense fallback={fallback}>
          <LazyPanel renderPanel={render} />
        </Suspense>
      ) : (
        render(loadedNodePanels!)
      )}
    </NodePanelBoundary>
  );
};

/** The header's one status slot, hidden while there is nothing to report. */
const NodeStatusButton = ({
  execution,
  isPlain = false,
  issues,
  node,
  status,
  template,
}: {
  execution: NodeExecutionState | null;
  /** A plain mark with no panel, for compact nodes; selecting the node opens it with its full header. */
  isPlain?: boolean;
  issues: readonly InvocationNodeFieldIssue[];
  node: WorkflowInvocationNode;
  status: WorkflowNodeStatus | null;
  template: InvocationNodeTemplateView['template'];
}) => {
  const { t } = useTranslation();

  if (status === null) {
    return null;
  }

  const updateText = getUpdateStatusText(t, node, template);
  const summary =
    status === 'invalid'
      ? issues.length === 1
        ? getFieldIssueMessage(t, issues[0]!)
        : t('nodes.status.issueCount', { count: issues.length })
      : status === 'updatable' || status === 'versionMismatch'
        ? updateText
        : getExecutionStatusText(t, execution!);

  const label = t('nodes.status.label', { status: summary });

  return (
    <WorkflowNodeStatusButton
      content={
        isPlain ? null : (
          <DeferredNodePanel
            label={label}
            render={(module) => (
              <module.NodeStatusPanel execution={execution} issues={issues} node={node} updateText={updateText} />
            )}
          />
        )
      }
      label={label}
      onPreload={preloadNodePanels}
      progress={execution?.status === 'running' ? execution.progress : null}
      status={status}
    />
  );
};

/** The editor a label double-click opens: Enter or blur commits, Escape keeps the current name. */
/**
 * The rename input a double-click opens, for node titles and field labels alike. It lies over its label on the same
 * line, a little wider and taller than it, in the larger input text and on its own surface (so a tinted header does
 * not tint it). The label keeps its place underneath, so nothing around it moves when the input opens or closes.
 */
const INLINE_RENAME_FIELD_PROPS = {
  left: '-6px',
  minW: '10rem',
  position: 'absolute',
  top: '50%',
  transform: 'translateY(-50%)',
  w: 'calc(100% + 12px)',
  zIndex: 1,
} as const;
const INLINE_RENAME_INPUT_PROPS = { bg: 'bg', color: 'fg', fontSize: 'lg', fontWeight: '400', px: '1.5' } as const;

const INLINE_RENAME_HIDDEN_LABEL_CSS = { '& > :first-child': { visibility: 'hidden' } } as const;

/** Holds a label and, while it is being renamed, the input over it. */
const InlineRename = ({ children, editor }: { children: ReactNode; editor: ReactNode }) => (
  <Box
    alignItems="center"
    display="flex"
    minW="0"
    position="relative"
    css={editor ? INLINE_RENAME_HIDDEN_LABEL_CSS : undefined}
  >
    {children}
    {editor}
  </Box>
);

/** The editor a label double-click opens: Enter or blur commits, Escape keeps the current name. */
const InlineLabelInput = ({
  ariaLabel,
  initialValue,
  onCommit,
  onEnd,
}: {
  ariaLabel: string;
  initialValue: string;
  onCommit: (value: string) => void;
  onEnd: () => void;
}) => {
  const [draft, setDraft] = useState(initialValue);
  const isCancelled = useRef(false);

  // Its own field: inside a value row it would otherwise take that control's id and invalid state.
  return (
    <Field.Root {...INLINE_RENAME_FIELD_PROPS}>
      <Input
        autoFocus
        aria-label={ariaLabel}
        className="nodrag"
        {...INLINE_RENAME_INPUT_PROPS}
        value={draft}
        onBlur={() => {
          if (!isCancelled.current && draft.trim() !== initialValue) {
            onCommit(draft.trim());
          }

          onEnd();
        }}
        onChange={(event: ChangeEvent<HTMLInputElement>) => setDraft(event.currentTarget.value)}
        onFocus={(event) => event.currentTarget.select()}
        onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
          if (event.key === 'Escape') {
            isCancelled.current = true;
          }

          if (event.key === 'Enter' || event.key === 'Escape') {
            event.stopPropagation();
            // Hand focus back to the node so editor shortcuts (undo) keep working; the blur commits.
            const node = event.currentTarget.closest<HTMLElement>('.react-flow__node');

            if (node) {
              node.focus({ preventScroll: true });
            } else {
              event.currentTarget.blur();
            }
          }
        }}
      />
    </Field.Root>
  );
};

const NodeTitle = ({
  isWorkflowImageExport,
  node,
  title,
}: {
  isWorkflowImageExport: boolean;
  node: WorkflowInvocationNode;
  title: string;
}) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const [isEditing, setIsEditing] = useState(false);

  if (isWorkflowImageExport) {
    return (
      <MiddleTruncate
        data-workflow-export-node-title="true"
        data-workflow-export-static-node-content="true"
        fontWeight="600"
        minW="0"
        text={title}
      />
    );
  }

  return (
    <InlineRename
      editor={
        isEditing ? (
          <InlineLabelInput
            ariaLabel={t('nodes.nodeLabel')}
            // Start from the displayed title, including the template fallback for unset labels.
            initialValue={title}
            onCommit={(label) => editGraph({ label, nodeId: node.id, type: 'setNodeLabel' })}
            onEnd={() => setIsEditing(false)}
          />
        ) : null
      }
    >
      <MiddleTruncate
        fontWeight="600"
        minW="0"
        text={title}
        // Keeps the full title MiddleTruncate would show, for a title cut short.
        title={t('nodes.nodeTitleHint', { title })}
        onDoubleClick={() => setIsEditing(true)}
      />
    </InlineRename>
  );
};

const InputFieldTooltip = ({
  description,
  isConnected,
  isExposed,
  issue,
  label,
  template,
}: {
  description: string;
  isConnected: boolean;
  isExposed: boolean;
  issue: InvocationNodeFieldIssue | null;
  label: string;
  template: FieldInputTemplate;
}) => {
  const { t } = useTranslation();

  return (
    <Stack gap="0.5" maxW="18rem">
      <Text fontWeight="600">{label}</Text>
      <Text color="fg.subtle">
        {t('nodes.fieldInfo', { name: template.name, type: getFieldTypeLabel(template.type) })} ·{' '}
        {t(template.required ? 'nodes.required' : 'nodes.optional')} · {t(`nodes.inputModes.${template.input}`)}
        {isConnected ? ` · ${t('nodes.providedByConnection')}` : null}
        {isExposed ? ` · ${t('nodes.inForm')}` : null}
      </Text>
      {issue ? <Text color="fg.error">{getFieldIssueMessage(t, issue)}</Text> : null}
      {description ? <Text>{description}</Text> : null}
      <Text color="fg.subtle">{t('nodes.renameFieldHint')}</Text>
    </Stack>
  );
};

const OutputFieldTooltip = ({
  includeTitle = true,
  template,
}: {
  includeTitle?: boolean;
  template: FieldOutputTemplate;
}) => {
  const { t } = useTranslation();

  return (
    <Stack gap="0.5" maxW="18rem">
      {includeTitle ? <Text fontWeight="700">{template.title}</Text> : null}
      <Text color="fg.subtle">
        {t('nodes.fieldInfo', { name: template.name, type: getFieldTypeLabel(template.type) })} · {t('nodes.output')}
      </Text>
      {template.description ? <Text>{template.description}</Text> : null}
    </Stack>
  );
};

/** One line naming the node's version against its template's, for the tooltip and the inspector. */
const getUpdateStatusText = (
  t: ReturnType<typeof useTranslation>['t'],
  node: WorkflowInvocationNode,
  template: InvocationNodeTemplateView['template']
): string | null => {
  const status = getNodeUpdateStatus(node, template);
  const versions = { from: node.data.version, to: template.version };

  return status === 'current'
    ? null
    : status === 'updatable'
      ? t('nodes.nodeUpdateAvailable', versions)
      : status === 'newer'
        ? t('nodes.nodeNewerThanBackend', versions)
        : t('nodes.nodeVersionIncompatible', versions);
};

/** Legacy's group tints, so a zipped group reads as one colour across the canvas. */
const BATCH_GROUP_COLORS: Record<string, string> = {
  'Group 1': 'green.fg',
  'Group 2': 'blue.fg',
  'Group 3': 'purple.fg',
  'Group 4': 'red.fg',
  'Group 5': 'yellow.fg',
};

/** A batch node's group beside its title: zipped groups share a colour, an ungrouped node says so. */
const BatchGroupSuffix = ({ node }: { node: WorkflowInvocationNode }) => {
  const { t } = useTranslation();

  if (!isWorkflowBatchNodeType(node.data.type)) {
    return null;
  }

  const groupId = getWorkflowBatchGroupId(node);

  return (
    <Text color={BATCH_GROUP_COLORS[groupId] ?? 'fg.subtle'} flexShrink={0} fontSize="xs" fontWeight="600">
      ({groupId === 'None' ? t('nodes.noBatchGroup') : groupId})
    </Text>
  );
};

const CLASSIFICATION_GLYPHS: Record<string, { color: string; icon: LucideIcon }> = {
  beta: { color: 'yellow.fg', icon: HammerIcon },
  internal: { color: 'purple.fg', icon: CpuIcon },
  prototype: { color: 'red.fg', icon: FlaskConicalIcon },
  special: { color: 'green.fg', icon: ZapIcon },
};

const CLASSIFICATION_TOOLTIP_POSITIONING = { placement: 'top-start' } as const;

/** Legacy's mark left of the title for a node that is not stable, explained on hover. */
const NodeClassificationIcon = ({ classification, isPlain = false }: { classification: string; isPlain?: boolean }) => {
  const { t } = useTranslation();
  const glyph = CLASSIFICATION_GLYPHS[classification];

  if (!glyph) {
    return null;
  }

  const description = t(`nodes.classificationDescriptions.${classification}`);

  const icon = (
    <Icon aria-label={description} as={glyph.icon} boxSize="3.5" color={glyph.color} flexShrink={0} role="img" />
  );

  return isPlain ? (
    icon
  ) : (
    <Tooltip content={description} positioning={CLASSIFICATION_TOOLTIP_POSITIONING} showArrow>
      {icon}
    </Tooltip>
  );
};

const NodeInfoButton = ({
  node,
  template,
}: {
  node: WorkflowInvocationNode;
  template: InvocationNodeTemplateView['template'];
}) => {
  const { t } = useTranslation();
  const label = t('nodes.showNodeDetails', { title: node.data.label || template.title });

  return (
    <WorkflowNodeInfoButton
      content={
        <DeferredNodePanel
          label={label}
          render={(module) => <module.NodeInfoPanel node={node} template={template} />}
        />
      }
      label={label}
      onPreload={preloadNodePanels}
    />
  );
};

const NodeFooter = ({ canUseCache, node }: { canUseCache: boolean; node: WorkflowInvocationNode }) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();

  return (
    <Flex
      align="center"
      bg="bg.subtle"
      borderBottomRadius="lg"
      borderColor="border.subtle"
      borderTopWidth="1px"
      className="nodrag"
      gap="3"
      justify="space-between"
      minH="8"
      px="2.5"
      py={WORKFLOW_NODE_DENSITY.rowPaddingY}
    >
      <HStack gap="4">
        {canUseCache ? (
          <Checkbox.Root
            checked={node.data.useCache}
            colorPalette="accent"
            size="sm"
            onCheckedChange={(event) =>
              editGraph({ nodeId: node.id, type: 'setNodeUseCache', useCache: event.checked === true })
            }
          >
            <Checkbox.HiddenInput />
            <Checkbox.Control />
            <Checkbox.Label fontSize="xs">{t('nodes.useCache')}</Checkbox.Label>
          </Checkbox.Root>
        ) : null}
        <Checkbox.Root
          checked={!node.data.isIntermediate}
          colorPalette="accent"
          size="sm"
          onCheckedChange={(event) =>
            editGraph({ isIntermediate: event.checked !== true, nodeId: node.id, type: 'setNodeIsIntermediate' })
          }
        >
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label fontSize="xs">{t('nodes.saveToGallery')}</Checkbox.Label>
        </Checkbox.Root>
      </HStack>
    </Flex>
  );
};

/** The upstream node owns the seed now; the local mode waits for a disconnect. A leaf so only this row subscribes to i18n. */
const ProvidedByConnectionNote = () => {
  const { t } = useTranslation();

  return (
    <Text color="fg.subtle" fontSize="xs" mt="0.5">
      {t('nodes.providedByConnection')}
    </Text>
  );
};

/**
 * A node's value controls sit in a well of the canvas colour, a step darker than the node body, so a value reads as
 * the thing to edit. The colour is defined once, as `--wb-control-well` on the row, which also carries
 * `data-control-well`, and every surface that paints a control's background reads it: the value surfaces below (text
 * and number inputs, text areas, select, combobox and listbox pickers such as the model picker, not switches, plain
 * buttons or media drop zones), the text area's faded bottom (`ResizableTextarea`) and the export snapshot's value
 * box. Borders, including the error border, are untouched.
 * Scoped to the node's rows, so the form view and the form builder's cards keep their own surfaces.
 */
const NODE_FIELD_WELL_VARIABLE = { '--wb-control-well': '{colors.bg}' } as const;
const NODE_FIELD_WELL_CSS = {
  '& :is(input:not([type="checkbox"], [type="radio"], [type="range"], [type="file"]), textarea, [data-scope="select"][data-part="trigger"], [data-scope="combobox"][data-part="control"], button[aria-haspopup="listbox"])':
    { bg: 'var(--wb-control-well)' },
} as const;

/** Row actions stay in the tab order; CSS shows them while the row is hovered, holds focus, or has a popover open. */
const INPUT_ROW_CSS = {
  ...NODE_FIELD_WELL_VARIABLE,
  '& [data-node-field-actions]': {
    opacity: 0,
    transition: 'opacity var(--wb-motion-duration-fast) ease',
  },
  '&:hover [data-node-field-actions], &:focus-within [data-node-field-actions], & [data-node-field-actions]:has([data-state="open"])':
    { opacity: 1 },
} as const;

/** Legacy's row actions, in its order: description, reset, add to or remove from the form. */
const InputFieldActions = ({
  fieldName,
  isExposed,
  label,
  node,
  showsControl,
  template,
}: {
  fieldName: string;
  isExposed: boolean;
  label: string;
  node: WorkflowInvocationNode;
  showsControl: boolean;
  template: FieldInputTemplate;
}) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const instance = node.data.inputs[fieldName];
  const formLabel = isExposed ? t('nodes.removeFieldFromForm', { label }) : t('nodes.addFieldToForm', { label });
  const isDefault = showsControl && isWorkflowFieldValueDefault(template, instance?.value);

  return (
    <HStack data-node-field-actions="true" flexShrink={0} gap="0" ml="auto">
      <FieldDescriptionPopover
        description={instance?.description}
        fieldName={fieldName}
        nodeId={node.id}
        size="xs"
        templateDescription={template.description}
      />
      {showsControl ? (
        // As legacy: present but disabled at the default. A disabled button shows no tooltip, so none is mounted.
        <Tooltip content={t('nodes.resetToDefault')} disabled={isDefault}>
          <IconButton
            aria-label={t('nodes.resetFieldToDefault', { label })}
            className="nodrag"
            color="fg.subtle"
            disabled={isDefault}
            size="xs"
            variant="ghost"
            onClick={() =>
              editGraph({
                fieldName,
                nodeId: node.id,
                type: 'setFieldValue',
                value: cloneWorkflowFieldDefault(template),
              })
            }
          >
            <Icon as={RotateCcwIcon} boxSize="3" />
          </IconButton>
        </Tooltip>
      ) : null}
      {showsControl && isExposableField(template) ? (
        <Tooltip content={formLabel}>
          <IconButton
            aria-label={formLabel}
            className="nodrag"
            color="fg.subtle"
            size="xs"
            variant="ghost"
            onClick={() =>
              editGraph({
                fieldIdentifier: { fieldName, nodeId: node.id },
                type: isExposed ? 'unexposeField' : 'exposeField',
              })
            }
          >
            <Icon as={isExposed ? MinusIcon : PlusIcon} boxSize="3" />
          </IconButton>
        </Tooltip>
      ) : null}
    </HStack>
  );
};

const InputFieldRow = ({
  connectedFieldNames,
  isConnected,
  isExposed,
  isSkeleton,
  issue,
  node,
  template,
}: {
  connectedFieldNames: ReadonlySet<string>;
  isConnected: boolean;
  isExposed: boolean;
  /** The field's entry in the node's issue list, null when it is fine. */
  issue: InvocationNodeFieldIssue | null;
  isSkeleton: boolean;
  node: WorkflowInvocationNode;
  template: FieldInputTemplate;
}) => {
  const isWorkflowImageExport = useIsWorkflowImageExport();
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const instance = node.data.inputs[template.name];
  const showsControl = !isConnected && isDirectInputField(template);
  const label = instance?.label || template.title;
  const description = getEffectiveWorkflowFieldDescription(instance, template);
  const ratiosDisabled =
    isHiDiffusionNode(node.data.type) && isHiDiffusionRatioField(template.name) && areHiDiffusionRatiosAutomatic(node);
  const isInvalid = issue !== null;
  const showInvalid = isInvalid && !isWorkflowImageExport;
  const showsInForm = isExposed && !isWorkflowImageExport;
  const handleTooltip = getHandleTypeTooltip(template.type, t('nodes.any'), t);
  const [isEditingLabel, setIsEditingLabel] = useState(false);
  // Legacy's label: semibold where the row holds a value, muted where a connection feeds it; red when the field
  // blocks a run, blue with a link mark when the field is in the form.
  const fieldTitle = (
    <Text
      alignItems="center"
      color={showInvalid ? 'fg.error' : showsInForm ? 'accent.fg' : showsControl ? 'fg' : 'fg.muted'}
      data-node-input-field-title="true"
      display="flex"
      fontSize="sm"
      fontWeight={showsControl ? '600' : '400'}
      gap="1"
      lineHeight={WORKFLOW_NODE_DENSITY.labelLine}
      minW="0"
      onDoubleClick={isWorkflowImageExport ? undefined : () => setIsEditingLabel(true)}
    >
      <Text as="span" minW="0" truncate>
        {label}
        {template.required ? (
          <Text as="span" color="fg.error">
            {' *'}
          </Text>
        ) : null}
      </Text>
      {showsInForm ? <Icon aria-label={t('nodes.inForm')} as={LinkIcon} boxSize="3" flexShrink={0} role="img" /> : null}
    </Text>
  );

  if (isSkeleton) {
    return (
      <Box px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
        <HStack gap="1.5" h={WORKFLOW_NODE_DENSITY.labelLine} position="relative">
          {template.input !== 'direct' ? (
            <Tooltip content={handleTooltip} showArrow>
              <Handle
                id={template.name}
                position={Position.Left}
                style={{ ...getWorkflowNodeHandleStyle(template.type, 'left'), left: -ROW_PADDING_X, top: PORT_TOP }}
                type="target"
              />
            </Tooltip>
          ) : null}
          <SkeletonBar w="55%" />
        </HStack>
        {showsControl ? <SkeletonBar h="6" w="full" /> : null}
      </Box>
    );
  }

  return (
    <Box
      css={INPUT_ROW_CSS}
      {...{ [WORKFLOW_NODE_FIELD_ROW_ATTRIBUTE]: '' }}
      data-control-well=""
      data-workflow-export-field-content={isWorkflowImageExport ? 'true' : undefined}
      px={WORKFLOW_NODE_DENSITY.rowPaddingX}
      py={WORKFLOW_NODE_DENSITY.rowPaddingY}
      w="full"
    >
      <Field.Root gap="0" invalid={isInvalid && !isWorkflowImageExport} minW="0" w="full">
        {/* The handle lives inside the label row so it stays centered on the
            label even when the value control below grows the row. */}
        <HStack
          gap="1.5"
          h={WORKFLOW_NODE_DENSITY.labelLine}
          justify="space-between"
          minW="0"
          position="relative"
          w="full"
        >
          {template.input !== 'direct' ? (
            <Tooltip content={handleTooltip} positioning={{ placement: 'right-end' }} showArrow>
              <Handle
                id={template.name}
                position={Position.Left}
                style={{ ...getWorkflowNodeHandleStyle(template.type, 'left'), left: -ROW_PADDING_X, top: PORT_TOP }}
                type="target"
              />
            </Tooltip>
          ) : null}
          {isWorkflowImageExport ? (
            fieldTitle
          ) : (
            <InlineRename
              editor={
                isEditingLabel ? (
                  <InlineLabelInput
                    ariaLabel={t('nodes.fieldLabel')}
                    initialValue={label}
                    // The template title is the unset label, so renaming back to it follows template changes again.
                    onCommit={(next) =>
                      editGraph({
                        fieldName: template.name,
                        label: next === template.title ? '' : next,
                        nodeId: node.id,
                        type: 'setFieldLabel',
                      })
                    }
                    onEnd={() => setIsEditingLabel(false)}
                  />
                ) : null
              }
            >
              <Tooltip
                disabled={isEditingLabel}
                positioning={{ placement: 'top-start' }}
                content={
                  <InputFieldTooltip
                    description={description}
                    isConnected={isConnected}
                    isExposed={isExposed}
                    issue={issue}
                    label={label}
                    template={template}
                  />
                }
              >
                {fieldTitle}
              </Tooltip>
            </InlineRename>
          )}
          {!isWorkflowImageExport ? (
            <InputFieldActions
              fieldName={template.name}
              isExposed={isExposed}
              label={label}
              node={node}
              showsControl={showsControl}
              template={template}
            />
          ) : null}
        </HStack>
        {showsControl ? (
          isWorkflowImageExport ? (
            <WorkflowFieldSnapshot
              seedMode={getWorkflowFieldSeedMode(instance)}
              template={template}
              value={instance?.value}
            />
          ) : (
            <Box css={NODE_FIELD_WELL_CSS} mt="0.5" w="full" {...{ [WORKFLOW_NODE_FIELD_CONTROL_ATTRIBUTE]: '' }}>
              <WorkflowFieldInput
                disabled={ratiosDisabled}
                id={`${node.id}-${template.name}-value`}
                invalid={isInvalid}
                nodeId={node.id}
                seedMode={getWorkflowFieldSeedMode(instance)}
                template={getHiDiffusionRatioTemplate(node, template, connectedFieldNames)}
                value={
                  isHiDiffusionNode(node.data.type) && isHiDiffusionRatioField(template.name)
                    ? (instance?.value ?? 0)
                    : instance?.value
                }
                onChange={(value) =>
                  editGraph({ fieldName: template.name, nodeId: node.id, type: 'setFieldValue', value })
                }
                onSeedModeChange={(seedMode) =>
                  editGraph({ fieldName: template.name, nodeId: node.id, seedMode, type: 'setFieldSeedMode' })
                }
              />
            </Box>
          )
        ) : !isWorkflowImageExport && isConnected && isSeedInputField(template) ? (
          <ProvidedByConnectionNote />
        ) : null}
        {/* The reason is read with the control; on screen it lives in the status panel and the label's hover. */}
        {showInvalid && issue ? <Field.ErrorText srOnly>{getFieldIssueMessage(t, issue)}</Field.ErrorText> : null}
      </Field.Root>
    </Box>
  );
};

const OutputFieldRow = ({
  isSkeleton,
  latestResult,
  template,
}: {
  isSkeleton: boolean;
  latestResult: unknown;
  template: FieldOutputTemplate;
}) => {
  const isWorkflowImageExport = useIsWorkflowImageExport();
  const { t } = useTranslation();
  const handleTooltip = getHandleTypeTooltip(template.type, t('nodes.any'), t);
  const value =
    !isWorkflowImageExport && latestResult !== undefined ? formatOutputFieldValue(latestResult, template.name) : null;

  return (
    <Box px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
      <Flex
        align="center"
        h={isWorkflowImageExport ? 'auto' : WORKFLOW_NODE_DENSITY.labelLine}
        justify="flex-end"
        minH={isWorkflowImageExport ? WORKFLOW_NODE_DENSITY.labelLine : undefined}
        position="relative"
      >
        <Tooltip content={handleTooltip} positioning={{ placement: 'left-start' }} showArrow>
          <Handle
            id={template.name}
            position={Position.Right}
            style={{ ...getWorkflowNodeHandleStyle(template.type, 'right'), right: -ROW_PADDING_X, top: PORT_TOP }}
            type="source"
          />
        </Tooltip>
        {isSkeleton ? (
          <Flex justify="flex-end" w="full">
            <SkeletonBar w="40%" />
          </Flex>
        ) : isWorkflowImageExport ? (
          <Box flexShrink={0} maxW="full" textAlign="end">
            <Text
              color="fg.muted"
              data-workflow-export-output-title="true"
              fontSize="xs"
              lineHeight={WORKFLOW_NODE_DENSITY.labelLine}
              maxW="full"
              overflowWrap="anywhere"
              textAlign="end"
              whiteSpace="normal"
            >
              {template.title}
            </Text>
          </Box>
        ) : (
          <HStack gap="1.5" justify="flex-end" minW="0" w="full">
            {value ? (
              <Tooltip content={value.full} positioning={{ placement: 'top-end' }}>
                <Text
                  color="fg.subtle"
                  fontSize="xs"
                  lineHeight={WORKFLOW_NODE_DENSITY.labelLine}
                  minW="0"
                  overflow="hidden"
                  textOverflow="ellipsis"
                  whiteSpace="nowrap"
                >
                  {value.short}
                </Text>
              </Tooltip>
            ) : null}
            <Box flexShrink={0} maxW={value ? '60%' : 'full'} textAlign="end">
              <Tooltip content={<OutputFieldTooltip template={template} />} positioning={{ placement: 'top-end' }}>
                <MiddleTruncate
                  as="span"
                  color="fg.muted"
                  data-node-output-field-title="true"
                  fontSize="xs"
                  justifyContent="flex-end"
                  lineHeight={WORKFLOW_NODE_DENSITY.labelLine}
                  maxW="full"
                  text={template.title}
                />
              </Tooltip>
            </Box>
          </HStack>
        )}
      </Flex>
    </Box>
  );
};

const OutputScopeHeader = ({ scope }: { scope: 'iteration' | 'final' }) => {
  const { t } = useTranslation();

  return (
    <Box px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
      <Text color="fg.subtle" fontSize="xs" fontWeight="600" textAlign="end">
        {scope === 'iteration' ? t('nodes.iterationOutputs') : t('nodes.finalOutputs')}
      </Text>
    </Box>
  );
};

/** Keeps every handle mounted (invisible) so edges stay attached when rows are not rendered. */
const HiddenHandles = ({
  inputTemplates,
  outputTemplates,
}: {
  inputTemplates: FieldInputTemplate[];
  outputTemplates: FieldOutputTemplate[];
}) => (
  <Box position="relative" h="0">
    {outputTemplates.map((outputTemplate) => (
      <Handle
        key={outputTemplate.name}
        id={outputTemplate.name}
        position={Position.Right}
        style={{ ...getWorkflowNodeHandleStyle(outputTemplate.type, 'right'), opacity: 0, right: 0, top: -14 }}
        type="source"
      />
    ))}
    {inputTemplates.map((inputTemplate) =>
      inputTemplate.input !== 'direct' ? (
        <Handle
          key={inputTemplate.name}
          id={inputTemplate.name}
          position={Position.Left}
          style={{ ...getWorkflowNodeHandleStyle(inputTemplate.type, 'left'), left: 0, opacity: 0, top: -14 }}
          type="target"
        />
      ) : null
    )}
  </Box>
);

const MERGED_PORT_SIZE = 14;
/**
 * A field handle behind a merged port: present so its edges attach there, but neither seen nor connectable. It takes
 * the port's size, so an edge leaves the port's outer side as it leaves an open node's dot.
 */
const COLLAPSED_FIELD_HANDLE_STYLE = {
  height: MERGED_PORT_SIZE,
  minHeight: 0,
  minWidth: 0,
  visibility: 'hidden',
  width: MERGED_PORT_SIZE,
} as const;

/** Legacy's merged port: one per side of a collapsed node, a rounded square in the neutral port colour. */
const MergedPort = ({ side }: { side: 'left' | 'right' }) => (
  <Box
    aria-hidden
    bg="border.emphasized"
    boxShadow="0 0 0 1.5px var(--xy-background-color, {colors.bg})"
    boxSize={`${MERGED_PORT_SIZE}px`}
    data-collapsed-port={side}
    pointerEvents="none"
    position="absolute"
    rounded="3px"
    top="50%"
    transform={side === 'left' ? 'translate(-50%, -50%)' : 'translate(50%, -50%)'}
    {...(side === 'left' ? { left: '0' } : { right: '0' })}
  />
);

/**
 * A collapsed node shows one merged port per side, as legacy did. Every field keeps its handle, hidden at that port,
 * so the document's edges stay as they are and draw from it; expanding draws them from their own rows again. The
 * hidden handles cannot connect: a new connection needs the node open, as in legacy.
 */
const CollapsedPorts = ({
  connected,
  inputTemplates,
  outputTemplates,
}: {
  /** In a large graph's compact mode, only the handles with edges mount, as `CompactHiddenHandles` does. */
  connected?: { sources: readonly string[]; targets: readonly string[] };
  inputTemplates: FieldInputTemplate[];
  outputTemplates: FieldOutputTemplate[];
}) => {
  const targetTemplates = inputTemplates.filter((inputTemplate) => inputTemplate.input !== 'direct');
  const targetHandles = connected
    ? targetTemplates.filter((inputTemplate) => connected.targets.includes(inputTemplate.name))
    : targetTemplates;
  const sourceHandles = connected
    ? outputTemplates.filter((outputTemplate) => connected.sources.includes(outputTemplate.name))
    : outputTemplates;

  return (
    <>
      {targetTemplates.length > 0 ? <MergedPort side="left" /> : null}
      {targetHandles.map((inputTemplate) => (
        <Handle
          key={inputTemplate.name}
          id={inputTemplate.name}
          isConnectable={false}
          position={Position.Left}
          style={COLLAPSED_FIELD_HANDLE_STYLE}
          type="target"
        />
      ))}
      {outputTemplates.length > 0 ? <MergedPort side="right" /> : null}
      {sourceHandles.map((outputTemplate) => (
        <Handle
          key={outputTemplate.name}
          id={outputTemplate.name}
          isConnectable={false}
          position={Position.Right}
          style={COLLAPSED_FIELD_HANDLE_STYLE}
          type="source"
        />
      ))}
    </>
  );
};

const CompactHiddenHandles = ({
  connectedSourceHandles,
  connectedTargetHandles,
  inputTemplates,
  outputTemplates,
}: {
  connectedSourceHandles: string[];
  connectedTargetHandles: string[];
  inputTemplates: FieldInputTemplate[];
  outputTemplates: FieldOutputTemplate[];
}) => {
  const connectedSources = new Set(connectedSourceHandles);
  const connectedTargets = new Set(connectedTargetHandles);

  return (
    <HiddenHandles
      inputTemplates={inputTemplates.filter((inputTemplate) => connectedTargets.has(inputTemplate.name))}
      outputTemplates={outputTemplates.filter((outputTemplate) => connectedSources.has(outputTemplate.name))}
    />
  );
};

const CompactNodeBody = ({ inputCount, outputCount }: { inputCount: number; outputCount: number }) => {
  const { t } = useTranslation();

  return (
    <Flex
      align="center"
      color="fg.muted"
      fontSize="xs"
      gap="2"
      px={WORKFLOW_NODE_DENSITY.rowPaddingX}
      {...getWorkflowNodeBodyProps()}
    >
      <Text>{t('nodes.inputCount', { count: inputCount })}</Text>
      <Text>·</Text>
      <Text>{t('nodes.outputCount', { count: outputCount })}</Text>
      <Text ms="auto">{t('nodes.selectForFields')}</Text>
    </Flex>
  );
};

/** What a node without a template says: still loading, definitions unavailable, or a type this backend lacks. */
const getTemplatelessNodeMessage = (
  t: TFunction,
  { documentNode, templateStatus }: InvocationFlowNodeType['data'],
  isCompact: boolean
): string =>
  templateStatus === 'loading'
    ? t('nodes.loadingNode')
    : templateStatus === 'error'
      ? t('nodes.nodeDefinitionsUnavailable')
      : isCompact
        ? t('nodes.unknownNodeType')
        : t('nodes.unknownNodeTypeOnBackend', { type: documentNode.data.type });

/**
 * Where a node type this backend lacks can come from: the Nodes Manager installs node packs, for an account that may;
 * anyone else is told who can.
 */
const MissingNodeTypeAction = () => {
  const { t } = useTranslation();
  const canManageNodes = useWorkflowCapabilitiesSelector((capabilities) => capabilities.canManageNodes);
  const openNodesManager = useOpenNodesManager();

  return canManageNodes ? (
    <Button alignSelf="flex-start" className="nodrag" mt="1.5" size="xs" variant="outline" onClick={openNodesManager}>
      {t('nodes.openNodesManager')}
    </Button>
  ) : (
    <Text color="fg.subtle" fontSize="xs">
      {t('nodes.unknownNodeTypeAskAdmin')}
    </Text>
  );
};

const CompactInvocationNode = ({ data, selected }: NodeProps<InvocationFlowNodeType>) => {
  const { t } = useTranslation();
  const node = data.documentNode;
  const templateView = data.template;
  const execution = useWorkflowNodeExecutionState(node.id);
  const inputTemplates = templateView?.inputTemplates ?? [];
  const outputTemplates = templateView?.outputTemplates ?? [];
  const title = node.data.label || templateView?.template.title || node.data.type;
  const fieldIssues = templateView
    ? getInvocationNodeFieldIssues(node, inputTemplates, new Set(data.connectedTargetHandles))
    : EMPTY_FIELD_ISSUES;
  const { chrome, status } = getInvocationNodeState({
    execution,
    hasFieldIssues: fieldIssues.length > 0,
    isMissingTemplate: !templateView && data.templateStatus === 'loaded',
    updateStatus: templateView ? getNodeUpdateStatus(node, templateView.template) : null,
  });
  const { tone } = getWorkflowNodeAppearance({ ...chrome, selected: false });
  const isCollapsed = templateView !== null && !node.data.isOpen;

  // Compact nodes stay cheap in a large graph: the status is a plain mark named for what it means, with no tooltip.
  return (
    <NodeShell chrome={chrome} selected={selected ?? false}>
      <Flex
        {...getWorkflowNodeHeaderProps({
          endsWithButton: templateView !== null && status !== null,
          roundedBottom: isCollapsed,
          tone,
        })}
      >
        {templateView ? <NodeClassificationIcon classification={templateView.template.classification} isPlain /> : null}
        <MiddleTruncate fontWeight="600" minW="0" text={title} />
        <BatchGroupSuffix node={node} />
        <Box flex="1" />
        {templateView ? (
          <NodeStatusButton
            execution={execution}
            isPlain
            issues={fieldIssues}
            node={node}
            status={status}
            template={templateView.template}
          />
        ) : null}
      </Flex>
      {isCollapsed ? (
        <CollapsedPorts
          connected={{ sources: data.connectedSourceHandles, targets: data.connectedTargetHandles }}
          inputTemplates={inputTemplates}
          outputTemplates={outputTemplates}
        />
      ) : (
        <>
          {templateView ? (
            <CompactNodeBody inputCount={inputTemplates.length} outputCount={outputTemplates.length} />
          ) : (
            <Text color="fg.subtle" fontSize="xs" px="3" {...getWorkflowNodeBodyProps()} py="2">
              {getTemplatelessNodeMessage(t, data, true)}
            </Text>
          )}
          <CompactHiddenHandles
            connectedSourceHandles={data.connectedSourceHandles}
            connectedTargetHandles={data.connectedTargetHandles}
            inputTemplates={inputTemplates}
            outputTemplates={outputTemplates}
          />
        </>
      )}
    </NodeShell>
  );
};

const ExpandedInvocationNode = ({ data, selected }: NodeProps<InvocationFlowNodeType>) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const isWorkflowImageExport = useIsWorkflowImageExport();
  const isZoomedOut = useIsZoomedOut();
  const node = data.documentNode;
  const templateView = data.template;
  const execution = useWorkflowNodeExecutionState(node.id);

  if (!templateView) {
    const isMissing = !isWorkflowImageExport && data.templateStatus === 'loaded';

    return (
      <NodeShell chrome={{ invalid: isMissing }} selected={selected ?? false}>
        <HStack gap="1.5" minW={isWorkflowImageExport ? '0' : undefined} p="3">
          {isMissing ? <Icon as={TriangleAlertIcon} boxSize="3.5" color="fg.error" /> : null}
          <Stack flex={isWorkflowImageExport ? '1' : undefined} gap="0" minW={isWorkflowImageExport ? '0' : undefined}>
            {isWorkflowImageExport ? (
              <MiddleTruncate
                data-workflow-export-node-title="true"
                data-workflow-export-static-node-content="true"
                fontWeight="600"
                minW="0"
                text={node.data.label || node.data.type}
              />
            ) : (
              <Text fontWeight="600">{node.data.label || node.data.type}</Text>
            )}
            {!isWorkflowImageExport ? (
              <Text color="fg.subtle" fontSize="xs">
                {getTemplatelessNodeMessage(t, data, false)}
              </Text>
            ) : null}
            {isMissing ? <MissingNodeTypeAction /> : null}
          </Stack>
        </HStack>
      </NodeShell>
    );
  }

  const template = templateView.template;
  const connectedFieldNames = new Set(data.connectedTargetHandles);
  const exposedFieldNames = new Set(data.exposedFieldNames);
  const inputTemplates = templateView.inputTemplates;
  const outputTemplates = templateView.outputTemplates;
  const outputTemplatesByName = new Map(outputTemplates.map((template) => [template.name, template]));
  const outputRows = getOutputFieldRows(getOutputFieldNamesByScope(outputTemplates));
  const isOpen = node.data.isOpen || isWorkflowImageExport;
  const fieldIssues = getInvocationNodeFieldIssues(node, inputTemplates, connectedFieldNames);
  const issuesByField = new Map(
    fieldIssues.flatMap((fieldIssue) => fieldIssue.markedFieldNames.map((name) => [name, fieldIssue] as const))
  );
  // The static export draws the document only: no run, validity or version state.
  const { chrome, status } = isWorkflowImageExport
    ? { chrome: undefined, status: null }
    : getInvocationNodeState({
        execution,
        hasFieldIssues: fieldIssues.length > 0,
        updateStatus: getNodeUpdateStatus(node, template),
      });
  const { tone } = getWorkflowNodeAppearance({ ...chrome, selected: false });
  const isCompact = data.isCompact && !selected && !isWorkflowImageExport;
  const withFooter = !isWorkflowImageExport && !isZoomedOut && templateView.isExecutable && templateView.hasImageOutput;
  const resultExecution = isWorkflowImageExport ? null : execution;
  const latestResult = resultExecution?.latestOutput;
  const title = node.data.label || template.title;

  return (
    <NodeShell chrome={chrome} selected={selected ?? false}>
      {/* Buttons at either end of the header sit at the same inset from the border. */}
      <Flex
        {...getWorkflowNodeHeaderProps({
          // Far zoom drops the status and info buttons, leaving the title at the end.
          endsWithButton: !isWorkflowImageExport && !isZoomedOut,
          roundedBottom: !isOpen,
          startsWithButton: !isWorkflowImageExport,
          tone,
        })}
        gap="0.5"
      >
        {!isWorkflowImageExport ? (
          <WorkflowNodeHeaderButton
            aria-label={isOpen ? t('nodes.collapseNode') : t('nodes.expandNode')}
            className="nodrag"
            content={isOpen ? t('nodes.collapseNode') : t('nodes.expandNode')}
            onClick={() => editGraph({ isOpen: !isOpen, nodeId: node.id, type: 'setNodeIsOpen' })}
          >
            <Icon as={isOpen ? ChevronDownIcon : ChevronRightIcon} boxSize="3.5" />
          </WorkflowNodeHeaderButton>
        ) : null}
        {!isWorkflowImageExport && !isZoomedOut ? (
          <NodeClassificationIcon classification={template.classification} />
        ) : null}
        {isZoomedOut ? (
          <MiddleTruncate fontWeight="600" minW="0" text={title} />
        ) : (
          <>
            <NodeTitle isWorkflowImageExport={isWorkflowImageExport} node={node} title={title} />
            <BatchGroupSuffix node={node} />
            <Box flex="1" />
            {!isWorkflowImageExport ? (
              <>
                <NodeStatusButton
                  execution={execution}
                  issues={fieldIssues}
                  node={node}
                  status={status}
                  template={template}
                />
                <NodeInfoButton node={node} template={template} />
              </>
            ) : null}
          </>
        )}
      </Flex>
      {isOpen && isCompact ? (
        <>
          <CompactNodeBody inputCount={inputTemplates.length} outputCount={outputTemplates.length} />
          <HiddenHandles inputTemplates={inputTemplates} outputTemplates={outputTemplates} />
        </>
      ) : isOpen ? (
        <Box {...getWorkflowNodeBodyProps({ roundedBottom: !withFooter })}>
          {outputRows.map((row) =>
            row.type === 'header' ? (
              <OutputScopeHeader key={`output-scope-${row.scope}`} scope={row.scope} />
            ) : (
              <OutputFieldRow
                key={row.fieldName}
                isSkeleton={isZoomedOut}
                latestResult={latestResult}
                template={outputTemplatesByName.get(row.fieldName) as FieldOutputTemplate}
              />
            )
          )}
          {inputTemplates.map((inputTemplate) => (
            <InputFieldRow
              key={inputTemplate.name}
              connectedFieldNames={connectedFieldNames}
              issue={issuesByField.get(inputTemplate.name) ?? null}
              isConnected={connectedFieldNames.has(inputTemplate.name)}
              isExposed={exposedFieldNames.has(inputTemplate.name)}
              isSkeleton={isZoomedOut}
              node={node}
              template={inputTemplate}
            />
          ))}
          {!isWorkflowImageExport && shouldShowCallSavedWorkflowLoadingHint(node) ? (
            <Text color="fg.subtle" fontSize="xs" px={WORKFLOW_NODE_DENSITY.rowPaddingX} py="1">
              {t('nodes.savedWorkflowDetailLoading')}
            </Text>
          ) : !isWorkflowImageExport && shouldShowCallSavedWorkflowNoExposedFieldsHint(node) ? (
            <Text color="fg.subtle" fontSize="xs" px={WORKFLOW_NODE_DENSITY.rowPaddingX} py="1">
              {t('nodes.savedWorkflowNoExposedFields')}
            </Text>
          ) : null}
        </Box>
      ) : (
        <CollapsedPorts inputTemplates={inputTemplates} outputTemplates={outputTemplates} />
      )}
      {isOpen && withFooter ? <NodeFooter canUseCache={data.canUseCache} node={node} /> : null}
    </NodeShell>
  );
};

const InvocationFlowNodeComponent = (props: NodeProps<InvocationFlowNodeType>) => {
  const isWorkflowImageExport = useIsWorkflowImageExport();

  return props.data.isCompact && !props.selected && !isWorkflowImageExport ? (
    <CompactInvocationNode {...props} />
  ) : (
    <ExpandedInvocationNode {...props} />
  );
};

export const InvocationFlowNode = memo(InvocationFlowNodeComponent);
