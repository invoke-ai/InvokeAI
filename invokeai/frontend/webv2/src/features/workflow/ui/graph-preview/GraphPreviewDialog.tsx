import type { InvocationTemplates, ProjectGraphState } from '@features/workflow/core/types';
import type { WorkflowPreviewMetrics, WorkflowPreviewModel } from '@features/workflow/core/workflowPreview';
import type {
  GraphPreviewInvoke,
  GraphPreviewSourceState,
  WorkflowInvocationSourceId,
} from '@features/workflow/ui/contracts';

import { Box, DataList, HStack, Icon, Portal, Stack, type SystemStyleObject, Text } from '@chakra-ui/react';
import { localizeForLoopValidationReason } from '@features/workflow/core/forLoops';
import { buildWorkflowPreviewModel, compiledGraphToPreviewDocument } from '@features/workflow/core/workflowPreview';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useNodesNeedingModel } from '@features/workflow/ui/modelRequirementStatus';
import { useWorkflowGraphPreview } from '@features/workflow/ui/WorkflowUiContext';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Scrollable } from '@platform/ui/Scrollable';
import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { CheckIcon, InfoIcon, TriangleAlertIcon } from 'lucide-react';
import { Suspense, useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowDocumentGraphHandle, WorkflowPreviewFieldSources } from './WorkflowDocumentGraph';
import type { WorkflowPreviewView } from './WorkflowPreviewStage';

import { getGraphJsonText } from './graphJson';
import { GraphPreviewMoreMenu, useGraphAsWorkflow } from './GraphPreviewMoreMenu';
import { WorkflowDocumentGraph } from './WorkflowDocumentGraph';
import { WorkflowPreviewInputs } from './WorkflowPreviewInputs';
import {
  getWorkflowPreviewViewTabs,
  useWorkflowPreviewFind,
  WorkflowPreviewBar,
  WorkflowPreviewFindField,
  WorkflowPreviewJson,
  WorkflowPreviewStageLoading,
  WorkflowPreviewStageMessage,
  WORKFLOW_PREVIEW_DIALOG_SIZE,
} from './WorkflowPreviewStage';

// Loaded through this module so the snapshot and the library's graph share the dialog's chunk; a second entry splits
// the flow out of it.
export { GraphPreviewSnapshot } from './GraphPreviewSnapshot';
export { WorkflowDocumentGraph } from './WorkflowDocumentGraph';

interface GraphPreviewDialogProps {
  graphId: string;
  isOpen: boolean;
  source: GraphPreviewSourceState;
  sourceId?: WorkflowInvocationSourceId;
  /** e.g. "Generate": the title, unless the source's document has a name of its own. */
  sourceLabel: string;
  /** The host's Invoke for this source, in the footer; none for a source that is not run from here (a review). */
  invoke?: GraphPreviewInvoke;
  onOpenChange: (isOpen: boolean) => void;
  /** Conditional hosts must unmount after close completion so exit animation can run. */
  onExitComplete?: () => void;
}

/** Stage and rail side by side; a narrow dialog puts the rail under the stage. */
const BODY_CSS: SystemStyleObject = {
  display: 'grid',
  flex: '1',
  gap: '3',
  py: '1',
  gridTemplateColumns: 'minmax(0, 1fr)',
  gridTemplateRows: 'minmax(0, 1fr) auto',
  minH: '0',
  // As wide as the library's detail rail.
  md: { gridTemplateColumns: 'minmax(0, 1fr) 18rem', gridTemplateRows: 'minmax(0, 1fr)' },
};
const RAIL_CSS: SystemStyleObject = { maxH: '10rem', md: { maxH: 'none' } };
const STAGE_LOADING = <WorkflowPreviewStageLoading />;

/** "Generate → Seed" names the setting "Seed": the label a set field takes in the drawing. */
const getSettingName = (provenance: string) => provenance.split('→').at(-1)!.trim() || provenance;

/** What the drawing was built from, and what was built. */
interface ModelState {
  graph: GraphPreviewSourceState['graph'];
  sourceDocument: ProjectGraphState | undefined;
  templates: InvocationTemplates | null;
  isPresent: boolean;
  document: ProjectGraphState | null;
  model: WorkflowPreviewModel | null;
}

const NO_MODEL: ModelState = {
  document: null,
  graph: null,
  isPresent: false,
  model: null,
  sourceDocument: undefined,
  templates: null,
};

interface Drawing {
  key: number;
  structureKey: string | null;
  /** The node whose card was open when the structure changed, which the new drawing opens again. */
  carriedNodeId: string | null;
}

/**
 * A compiled graph, or the workflow document it compiles from, drawn as the library previews a workflow: the same
 * graph, node card, find, Inputs view (for a document) and JSON view, with the source's facts and notices in a rail
 * beside it. A live source redraws in place as it recompiles: a value change redraws only the chips it changes; a
 * structure change draws afresh, keeping an open card whose node survives. Nothing is built while it is closed.
 */
export const GraphPreviewDialog = ({
  graphId,
  isOpen,
  source,
  sourceId,
  sourceLabel,
  invoke,
  onOpenChange,
  onExitComplete,
}: GraphPreviewDialogProps) => {
  const { t } = useTranslation();
  const graphPreview = useWorkflowGraphPreview();
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const viewTabsId = useId();
  const graphRef = useRef<WorkflowDocumentGraphHandle>(null);
  const [view, setView] = useState<WorkflowPreviewView>('nodes');
  // Open, or still playing its close: until the exit completes the content shows what it showed.
  const [isPresent, setIsPresent] = useState(isOpen);
  if (isOpen && !isPresent) {
    setIsPresent(true);
  }

  const { document: sourceDocument, getProvenance, graph, positionHints, resolvedInputOverrides } = source;
  const templates = templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null;
  /**
   * One model path: a compiled graph is drawn as a document whose form is what the person set, named by the setting.
   * Rebuilt when the graph or document itself changes (not the source object around them: a host may hand a fresh one
   * each render); its provenance, overrides and position hints come with each compile. Each rebuild hands the last
   * model on, so what a recompile did not change keeps its identity. Nothing is built while the dialog is closed.
   */
  const [modelState, setModelState] = useState<ModelState>(NO_MODEL);
  let built = modelState;
  if (
    modelState.graph !== graph ||
    modelState.sourceDocument !== sourceDocument ||
    modelState.isPresent !== isPresent ||
    (isPresent && modelState.templates !== templates)
  ) {
    const document = !isPresent
      ? null
      : (sourceDocument ??
        (graph
          ? compiledGraphToPreviewDocument(graph, {
              getSetLabel: getProvenance
                ? (nodeId, fieldName) => {
                    const provenance = getProvenance(nodeId, fieldName);

                    return provenance ? getSettingName(provenance.label) : null;
                  }
                : undefined,
              overrides: resolvedInputOverrides,
              positionHints,
            })
          : null));

    built = {
      document,
      graph,
      isPresent,
      model: document && templates ? buildWorkflowPreviewModel(document, templates, undefined, modelState.model) : null,
      sourceDocument,
      templates,
    };
    setModelState(built);
  }
  const { document, model } = built;
  // The drawing at a wider chip, where the graph's stage has the room.
  const relayout = useCallback(
    (metrics: WorkflowPreviewMetrics, previous: WorkflowPreviewModel | null) =>
      buildWorkflowPreviewModel(document!, templates!, metrics, previous),
    [document, templates]
  );
  const needsModel = useNodesNeedingModel(model);
  // The open card, as the graph reports it, so a structure change can carry it into the new drawing.
  const [cardNodeId, setCardNodeId] = useState<string | null>(null);
  const structureKey = model?.structureKey ?? null;
  const [drawing, setDrawing] = useState<Drawing>({ carriedNodeId: null, key: 0, structureKey });
  if (drawing.structureKey !== structureKey) {
    setDrawing({
      carriedNodeId: cardNodeId !== null && model?.nodes.has(cardNodeId) ? cardNodeId : null,
      key: drawing.key + 1,
      structureKey,
    });
  }
  const find = useWorkflowPreviewFind(graphRef, drawing.key);
  // A node a notice or an input asked to show opens selected in its drawing; asking again opens it again.
  const [shownNode, setShownNode] = useState<{ drawing: number; nodeId: string; request: number } | null>(null);
  const requestedNodeId = shownNode?.drawing === drawing.key ? shownNode.nodeId : null;

  const closeDialog = useCallback(() => onOpenChange(false), [onOpenChange]);
  const handleOpenChange = useCallback((event: { open: boolean }) => onOpenChange(event.open), [onOpenChange]);
  const { reset: resetFind } = find;
  // Reopened, the preview starts on the graph again, unfiltered, no card open.
  const handleExitComplete = useCallback(() => {
    setIsPresent(false);
    setView('nodes');
    setShownNode(null);
    setCardNodeId(null);
    resetFind();
    onExitComplete?.();
  }, [onExitComplete, resetFind, setCardNodeId]);
  const handleEscapeKeyDown = useCallback((event: KeyboardEvent) => {
    // One keypress can arrive twice (StrictMode registers the open dialog's layer twice); only the first acts.
    if (event.defaultPrevented) {
      return;
    }

    if (graphRef.current?.closeNodeCard()) {
      event.preventDefault();
    }
  }, []);
  const fieldSources = useMemo<WorkflowPreviewFieldSources | undefined>(
    () =>
      getProvenance
        ? {
            describe: (nodeId, fieldName) => getProvenance(nodeId, fieldName)?.label ?? null,
            onSelect: () => {
              graphPreview.focusSource(sourceId);
              onOpenChange(false);
            },
          }
        : undefined,
    [getProvenance, graphPreview, onOpenChange, sourceId]
  );

  const { canConvert, editInEditor } = useGraphAsWorkflow({
    graph,
    onClose: closeDialog,
    sourceDocument,
    sourceLabel,
  });

  const selectView = useCallback(
    (next: WorkflowPreviewView) => {
      setShownNode(null);
      setCardNodeId(null);
      resetFind();
      setView(next);
    },
    [resetFind, setCardNodeId]
  );
  const showNode = useCallback(
    (nodeId: string) => {
      resetFind();
      setView('nodes');
      setShownNode((current) => ({ drawing: drawing.key, nodeId, request: (current?.request ?? 0) + 1 }));
    },
    [drawing.key, resetFind]
  );
  const getJsonText = useCallback(() => (graph ? getGraphJsonText(graph) : ''), [graph]);

  // Only a document has a form to show. The Graph view always shows: the compiled graph, or why there is none.
  const hasInputs = sourceDocument !== undefined;
  const viewTabs = useMemo(
    () => getWorkflowPreviewViewTabs(model, { hasGraph: true, hasInputs }, t),
    [hasInputs, model, t]
  );
  const shownView = view === 'inputs' && !hasInputs ? 'nodes' : view;
  const blockedReason =
    source.invalidReasons.length > 0
      ? `${t('graphPreview.invalidTitle')} ${source.invalidReasons
          .map((reason) => localizeForLoopValidationReason(reason, t))
          .join(' ')}`
      : null;
  // A document still draws when it does not compile; a compiled source has nothing to draw.
  const isBlocked = blockedReason !== null && !sourceDocument;
  const showsNodes = shownView === 'nodes' && !isBlocked && model !== null && model.nodes.size > 0;
  // The run's facts: the source's summary, then where results go.
  const facts = [
    ...source.summaryRows,
    ...(source.destinationLabel === null
      ? []
      : [{ id: 'destination', label: t('graphPreview.destination'), value: source.destinationLabel }]),
  ];
  // The rail says what the graph alone does not: the run's facts, notices, and why a drawn document does not compile.
  const hasRail = facts.length > 0 || source.notices.length > 0 || (blockedReason !== null && !isBlocked);
  // The footer decides the run: whether it can, Invoke, and for a compiled source the way to edit it as a workflow.
  const canEditAsWorkflow = invoke !== undefined && !sourceDocument && sourceId !== 'workflow' && graph !== null;
  const subtitle = source.isLive ? t('graphPreview.liveHint') : null;
  const title = sourceDocument?.name.trim() || sourceLabel;
  const initialNodeId = requestedNodeId ?? drawing.carriedNodeId;

  const stageBody = (() => {
    if (shownView === 'graph') {
      return graph ? (
        <Suspense fallback={STAGE_LOADING}>
          <WorkflowPreviewJson
            copyFailedLabel={t('graphPreview.copyFailed')}
            copyLabel={t('graphPreview.copyGraphJson')}
            getText={getJsonText}
            label={t('workflowLibrary.preview.graphJson')}
          />
        </Suspense>
      ) : blockedReason !== null ? (
        <WorkflowPreviewStageMessage isAlert>{blockedReason}</WorkflowPreviewStageMessage>
      ) : (
        <WorkflowPreviewStageMessage>{t('graphPreview.noCompiledGraph', { graphId })}</WorkflowPreviewStageMessage>
      );
    }

    if (isBlocked) {
      return <WorkflowPreviewStageMessage isAlert>{blockedReason}</WorkflowPreviewStageMessage>;
    }

    if (!sourceDocument && !graph) {
      return (
        <WorkflowPreviewStageMessage>{t('graphPreview.noCompiledGraph', { graphId })}</WorkflowPreviewStageMessage>
      );
    }

    if (!model) {
      return templatesSnapshot.status === 'error' ? (
        <WorkflowPreviewStageMessage isAlert>
          {t('workflowLibrary.preview.unreadable', { message: templatesSnapshot.error ?? t('common.unknownError') })}
        </WorkflowPreviewStageMessage>
      ) : (
        STAGE_LOADING
      );
    }

    if (model.nodes.size === 0) {
      return <WorkflowPreviewStageMessage>{t('workflowLibrary.preview.empty')}</WorkflowPreviewStageMessage>;
    }

    return shownView === 'inputs' ? (
      <WorkflowPreviewInputs model={model} onShowNode={showNode} />
    ) : (
      <WorkflowDocumentGraph
        key={`${drawing.key}:${shownNode?.request ?? 0}`}
        ref={graphRef}
        fieldSources={fieldSources}
        focusInitialCard={requestedNodeId !== null}
        initialNodeId={initialNodeId}
        model={model}
        needsModel={needsModel}
        relayout={relayout}
        onCardChange={setCardNodeId}
        onMatchCountChange={find.onMatchCountChange}
      />
    );
  })();

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      size="xl"
      unmountOnExit
      onEscapeKeyDown={handleEscapeKeyDown}
      onExitComplete={handleExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content {...WORKFLOW_PREVIEW_DIALOG_SIZE}>
            <Dialog.Header alignItems="flex-start" flexDirection="row" gap="3" justifyContent="space-between">
              <Stack gap="0.5" minW="0">
                <Dialog.Title truncate>{title}</Dialog.Title>
                {subtitle ? <Dialog.Description>{subtitle}</Dialog.Description> : null}
              </Stack>
              <Dialog.CloseTrigger asChild>
                <CloseButton insetEnd="auto" position="static" top="auto" />
              </Dialog.CloseTrigger>
            </Dialog.Header>
            <Dialog.Body css={BODY_CSS}>
              <Stack
                borderColor="border.subtle"
                borderWidth="1px"
                data-graph-preview-stage=""
                gap="0"
                gridColumn={hasRail ? undefined : '1 / -1'}
                gridRow={hasRail ? undefined : '1 / -1'}
                minH="0"
                minW="0"
                overflow="hidden"
                rounded="md"
              >
                {/* Tabs at the start and More at the end in every view; only find comes and goes between them. */}
                <WorkflowPreviewBar>
                  <SegmentTabs
                    activeId={shownView}
                    ariaLabel={t('workflowLibrary.preview.view')}
                    idBase={viewTabsId}
                    isCompact
                    tabs={viewTabs}
                    onSelect={selectView}
                  />
                  <Box flex="1" />
                  {showsNodes ? <WorkflowPreviewFindField find={find} /> : null}
                  <GraphPreviewMoreMenu
                    graph={graph}
                    sourceDocument={sourceDocument}
                    sourceId={sourceId}
                    sourceLabel={sourceLabel}
                    onClose={closeDialog}
                  />
                </WorkflowPreviewBar>
                <Box
                  aria-labelledby={segmentTabsTabId(viewTabsId, shownView)}
                  flex="1"
                  id={segmentTabsPanelId(viewTabsId)}
                  minH="0"
                  position="relative"
                  role="tabpanel"
                >
                  {stageBody}
                </Box>
              </Stack>
              {hasRail ? (
                <Scrollable css={RAIL_CSS} data-graph-preview-rail="" label={t('graphPreview.aboutThisGraph')} minH="0">
                  <Stack gap="4" pb="1">
                    {facts.length > 0 ? (
                      <DataList.Root data-graph-preview-facts="" gap="1.5" orientation="horizontal">
                        {facts.map((fact) => (
                          <DataList.Item key={fact.id} alignItems="center" gap="2" minW="0">
                            <DataList.ItemLabel color="fg.muted" flex="0 0 5.5rem" fontSize="xs" minW="0">
                              {fact.label}
                            </DataList.ItemLabel>
                            <DataList.ItemValue flex="1" fontSize="xs" minW="0" overflowWrap="anywhere">
                              {fact.value}
                            </DataList.ItemValue>
                          </DataList.Item>
                        ))}
                      </DataList.Root>
                    ) : null}
                    {blockedReason !== null && !isBlocked ? (
                      <HStack align="flex-start" color="fg.warning" data-graph-preview-blocked="" gap="2" role="alert">
                        <Icon aria-hidden as={TriangleAlertIcon} boxSize="3.5" flexShrink={0} mt="0.5" />
                        <Text fontSize="md">{blockedReason}</Text>
                      </HStack>
                    ) : null}
                    {source.notices.map((notice) => (
                      <Notice key={notice.id} message={notice.message} nodeId={notice.nodeId} onShowNode={showNode} />
                    ))}
                  </Stack>
                </Scrollable>
              ) : null}
            </Dialog.Body>
            {invoke ? (
              <Dialog.Footer alignItems="center" data-graph-preview-footer="" gap="3" justifyContent="space-between">
                <ReadinessLine isReady={invoke.isReady} />
                <HStack flexShrink={0} gap="2">
                  {canEditAsWorkflow ? (
                    <Button
                      data-graph-preview-edit-as-workflow=""
                      disabled={!canConvert}
                      variant="outline"
                      onClick={editInEditor}
                    >
                      {t('graphPreview.editAsWorkflow')}
                    </Button>
                  ) : null}
                  {invoke.control}
                </HStack>
              </Dialog.Footer>
            ) : null}
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

/** Something about the source the graph alone does not say (a seed drawn each run), with a jump to its node. */
const Notice = ({
  message,
  nodeId,
  onShowNode,
}: {
  message: string;
  nodeId?: string;
  onShowNode: (nodeId: string) => void;
}) => {
  const { t } = useTranslation();
  const handleShowNode = useCallback(() => {
    if (nodeId) {
      onShowNode(nodeId);
    }
  }, [nodeId, onShowNode]);

  return (
    <HStack align="flex-start" data-graph-preview-notice="" gap="2">
      <Icon aria-hidden as={InfoIcon} boxSize="3.5" color="fg.muted" flexShrink={0} mt="0.5" />
      <Stack align="flex-start" gap="0.5" minW="0">
        <Text color="fg.muted" fontSize="md">
          {message}
        </Text>
        {nodeId ? (
          <Button
            color="accent.fg"
            fontSize="sm"
            fontWeight="400"
            h="auto"
            px="0"
            size="xs"
            variant="plain"
            onClick={handleShowNode}
          >
            {t('graphPreview.showNode')}
          </Button>
        ) : null}
      </Stack>
    </HStack>
  );
};

/**
 * The footer's readiness line: whether Invoke can run (its control's card says why not), a polite live region of its
 * own so a recompile says nothing until the answer changes.
 */
const ReadinessLine = ({ isReady }: { isReady: boolean }) => {
  const { t } = useTranslation();

  return (
    <HStack data-graph-preview-readiness="" flex="1" gap="2" minW="0">
      <Icon
        aria-hidden
        as={isReady ? CheckIcon : TriangleAlertIcon}
        boxSize="3.5"
        color={isReady ? 'fg.success' : 'fg.warning'}
        flexShrink={0}
      />
      <Text
        aria-live="polite"
        data-graph-preview-status=""
        fontSize="md"
        fontWeight="600"
        minW="0"
        role="status"
        truncate
      >
        {isReady ? t('graphPreview.readyToInvoke') : t('graphPreview.cannotInvoke')}
      </Text>
    </HStack>
  );
};
