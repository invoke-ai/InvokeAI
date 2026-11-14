import type { ProjectGraphState } from '@features/workflow/core/types';
import type { WorkflowDocumentGraphHandle } from '@features/workflow/ui/graph-preview/WorkflowDocumentGraph';

import { Box, Menu, Portal, Stack, type SystemStyleObject } from '@chakra-ui/react';
import { localizeForLoopValidationReason } from '@features/workflow/core/forLoops';
import { hasMultipleWorkflowReturnNodes } from '@features/workflow/core/workflowJson';
import {
  buildWorkflowPreviewModel,
  type WorkflowPreviewMetrics,
  type WorkflowPreviewModel,
} from '@features/workflow/core/workflowPreview';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { getGraphJsonText } from '@features/workflow/ui/graph-preview/graphJson';
import { WorkflowPreviewInputs } from '@features/workflow/ui/graph-preview/WorkflowPreviewInputs';
import {
  getWorkflowPreviewViewTabs,
  WorkflowPreviewBar,
  WorkflowPreviewFindField,
  WorkflowPreviewStageLoading,
  WorkflowPreviewStageMessage,
  type WorkflowPreviewView,
  useWorkflowPreviewFind,
  WorkflowPreviewJson,
} from '@features/workflow/ui/graph-preview/WorkflowPreviewStage';
import { useNodesNeedingModel } from '@features/workflow/ui/modelRequirementStatus';
import { copyWorkflowJson, downloadWorkflowJson } from '@features/workflow/ui/workflowTransfer';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { canCopyText } from '@platform/browser/clipboard';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ChevronLeftIcon, CopyIcon, DownloadIcon, EllipsisIcon } from 'lucide-react';
import { type Ref, Suspense, useCallback, useId, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { buildLibraryGraphPreviewSource, DeferredWorkflowDocumentGraph } from './libraryPreviewSource';
import { keepMenuOpenWhenSiblingMenuCloses } from './WorkflowLibraryCard';
import { WORKFLOW_LIBRARY_NARROW, WORKFLOW_LIBRARY_WIDE } from './WorkflowLibraryRail';

/**
 * Preview mode's stage: it takes the places rail's and the collection's place in the dialog while the detail rail stays
 * where it is (and steps through the view's results). The bar leaves (back to the place), switches the view, finds a
 * node and exports the saved workflow; the body draws the selected workflow's document. The collection stays laid out underneath, hidden, so leaving
 * returns to it exactly as it was.
 */

export interface WorkflowLibraryPreviewStageHandle {
  /** Escape in preview mode: closes an open node card, else leaves the preview. */
  handleEscape: () => void;
}

export type WorkflowLibraryPreviewStatus =
  | { kind: 'pending' }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; document: ProjectGraphState };

export interface WorkflowLibraryPreviewStageProps {
  ref: Ref<WorkflowLibraryPreviewStageHandle>;
  /** The place the preview leaves back to, for the back button. */
  placeLabel: string;
  /** The previewed workflow; null when the view has nothing selected. */
  workflowId: string | null;
  name: string;
  status: WorkflowLibraryPreviewStatus;
  /** Leaves preview mode; receives the workflow shown last, whose card or row then takes focus. */
  onLeave: (workflowId: string | null) => void;
  /** The view, which the library keeps while it is open. */
  view: WorkflowPreviewView;
  onViewChange: (view: WorkflowPreviewView) => void;
}

/**
 * Wide: over the places rail and the centre column, from under the header to the dialog's foot, reaching across the
 * column gap to the detail rail's border. Narrow: the panes grid lays it over the collection.
 */
const STAGE_CSS: SystemStyleObject = {
  bg: 'bg',
  minH: '0',
  minW: '0',
  zIndex: 1,
  [WORKFLOW_LIBRARY_WIDE]: { gridColumn: 'rail-start / toolbar-end', gridRow: '1 / -1', me: '-3' },
};
// A narrow dialog keeps the bar to leaving, the view and export; finding a node needs the room.
const FIND_CSS: SystemStyleObject = { [WORKFLOW_LIBRARY_NARROW]: { display: 'none' } };
// The stage takes focus as preview mode starts, so a reader hears its name; it draws no ring of its own.
const NO_OUTLINE = { outline: 'none' } as const;

export const WorkflowLibraryPreviewStage = ({
  name,
  placeLabel,
  ref,
  status,
  view,
  workflowId,
  onLeave,
  onViewChange,
}: WorkflowLibraryPreviewStageProps) => {
  const { t } = useTranslation();
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const notify = useWorkflowNotifications();
  const exportMenuIds = useTooltipTriggerIds();
  const viewTabsId = useId();
  // The node an input named, opened selected when the view switches to the graph; it belongs to its workflow.
  const [shownNode, setShownNode] = useState<{ nodeId: string; workflowId: string | null } | null>(null);
  const initialNodeId = shownNode?.workflowId === workflowId ? shownNode.nodeId : null;
  const stageRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<WorkflowDocumentGraphHandle>(null);

  const document = status.kind === 'ready' ? status.document : null;
  const templates = templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null;
  const model = useMemo(
    () => (document && templates ? buildWorkflowPreviewModel(document, templates) : null),
    [document, templates]
  );
  /**
   * Each document the stage draws is a new drawing: a workflow replaced under the same id (updated from its template)
   * must not inherit the last one's selected node, find matches, simplified state or viewport.
   */
  const [drawing, setDrawing] = useState({ key: 0, model });
  if (drawing.model !== model) {
    setDrawing({ key: drawing.key + 1, model });
  }
  const needsModel = useNodesNeedingModel(model);
  // The drawing at a wider chip, where the stage has the room.
  const relayout = useCallback(
    (metrics: WorkflowPreviewMetrics, previous: WorkflowPreviewModel | null) =>
      buildWorkflowPreviewModel(document!, templates!, metrics, previous),
    [document, templates]
  );
  const find = useWorkflowPreviewFind(graphRef, drawing.key);
  const menuPositioning = useMemo(
    () => ({
      boundary: () => stageRef.current?.closest<HTMLElement>('[data-scope="dialog"][data-part="content"]') ?? [],
    }),
    []
  );

  const leave = useCallback(() => onLeave(workflowId), [onLeave, workflowId]);
  useImperativeHandle(
    ref,
    () => ({
      handleEscape: () => {
        if (!graphRef.current?.closeNodeCard()) {
          leave();
        }
      },
    }),
    [leave]
  );

  const exportDocument = useCallback(
    (action: 'copy' | 'download') => {
      if (!document) {
        return;
      }

      // The editor's own export refuses the same documents: a workflow return node must be unique to round-trip.
      if (hasMultipleWorkflowReturnNodes(document)) {
        notify.error(
          action === 'copy' ? t('widgets.workflow.copyJsonFailed') : t('projects.exportFailed'),
          t('workflowLibrary.multipleWorkflowReturnNodesForTransfer')
        );
        return;
      }

      if (action === 'download') {
        downloadWorkflowJson(document);
        return;
      }

      copyWorkflowJson(document)
        .then(() => notify.success(t('widgets.workflow.copyJsonSuccess')))
        .catch(() => notify.error(t('widgets.workflow.copyJsonFailed')));
    },
    [document, notify, t]
  );
  const copyJson = useCallback(() => exportDocument('copy'), [exportDocument]);
  // Each switch draws the graph afresh, so the find field starts over with it.
  const { reset: resetFind } = find;
  const selectView = useCallback(
    (next: WorkflowPreviewView) => {
      setShownNode(null);
      resetFind();
      onViewChange(next);
    },
    [onViewChange, resetFind]
  );
  const showNode = useCallback(
    (nodeId: string) => {
      selectView('nodes');
      setShownNode({ nodeId, workflowId });
    },
    [selectView, workflowId]
  );
  const viewTabs = useMemo(() => getWorkflowPreviewViewTabs(model, { hasGraph: true, hasInputs: true }, t), [model, t]);
  // The Graph view compiles the document as it would run, only while that view shows (once per document).
  const compiled = useMemo(
    () => (view === 'graph' && document && templates ? buildLibraryGraphPreviewSource(document, templates) : null),
    [document, templates, view]
  );
  const compiledGraph = compiled?.graph ?? null;
  const getJsonText = useCallback(() => (compiledGraph ? getGraphJsonText(compiledGraph) : ''), [compiledGraph]);
  const downloadJson = useCallback(() => exportDocument('download'), [exportDocument]);

  const showsNodes = view === 'nodes' && model !== null && model.nodes.size > 0;

  return (
    <Stack
      ref={stageRef}
      aria-label={t('workflowLibrary.preview.region', { name })}
      as="section"
      css={STAGE_CSS}
      data-preview-stage={workflowId ?? ''}
      gap="0"
      tabIndex={-1}
      _focusVisible={NO_OUTLINE}
    >
      <WorkflowPreviewBar>
        <Button
          aria-label={t('workflowLibrary.preview.backTo', { place: placeLabel })}
          data-preview-back=""
          gap="1.5"
          px="2"
          variant="ghost"
          onClick={leave}
        >
          <ChevronLeftIcon />
          {placeLabel}
        </Button>
        <SegmentTabs
          activeId={view}
          ariaLabel={t('workflowLibrary.preview.view')}
          idBase={viewTabsId}
          isCompact
          tabs={viewTabs}
          onSelect={selectView}
        />
        <Box flex="1" />
        {showsNodes ? <WorkflowPreviewFindField css={FIND_CSS} find={find} /> : null}
        <Menu.Root
          ids={exportMenuIds}
          positioning={menuPositioning}
          onRequestDismiss={keepMenuOpenWhenSiblingMenuCloses}
        >
          {/* Inside a dialog the tooltip must sit inside the menu trigger: wrapped the other way round, the menu
              never takes focus and the first pointer move onto it closes it. */}
          <Menu.Trigger asChild>
            <Tooltip content={t('workflowLibrary.preview.exportMenu')} ids={exportMenuIds}>
              <IconButton
                aria-label={t('workflowLibrary.preview.exportMenu')}
                data-preview-export=""
                disabled={!document}
                size="sm"
                variant="outline"
              >
                <EllipsisIcon />
              </IconButton>
            </Tooltip>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="min(13rem, var(--available-width))" maxW="var(--available-width)">
                <MenuActionItem
                  icon={CopyIcon}
                  label={t('workflowLibrary.preview.copyJson')}
                  unavailableReason={canCopyText() ? undefined : t('common.copyUnavailable')}
                  value="copy-workflow-json"
                  onSelect={copyJson}
                />
                <MenuActionItem
                  icon={DownloadIcon}
                  label={t('workflowLibrary.preview.downloadJson')}
                  value="download-workflow-json"
                  onSelect={downloadJson}
                />
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </WorkflowPreviewBar>
      <Box
        aria-labelledby={segmentTabsTabId(viewTabsId, view)}
        data-preview-stage-body=""
        flex="1"
        id={segmentTabsPanelId(viewTabsId)}
        minH="0"
        position="relative"
        role="tabpanel"
      >
        {status.kind === 'error' ? (
          <WorkflowPreviewStageMessage isAlert>
            {t('workflowLibrary.preview.unreadable', { message: status.message })}
          </WorkflowPreviewStageMessage>
        ) : workflowId === null ? (
          <WorkflowPreviewStageMessage>{t('workflowLibrary.preview.nothingSelected')}</WorkflowPreviewStageMessage>
        ) : compiled ? (
          compiledGraph ? (
            <Suspense fallback={STAGE_LOADING}>
              <WorkflowPreviewJson
                copyFailedLabel={t('graphPreview.copyFailed')}
                copyLabel={t('graphPreview.copyGraphJson')}
                getText={getJsonText}
                label={t('workflowLibrary.preview.graphJson')}
              />
            </Suspense>
          ) : (
            // Why it would not run, in words: the Nodes and Inputs views still show the document.
            <WorkflowPreviewStageMessage isAlert>
              {t('graphPreview.invalidTitle')}{' '}
              {compiled.invalidReasons.map((reason) => localizeForLoopValidationReason(reason, t)).join(' ')}
            </WorkflowPreviewStageMessage>
          )
        ) : model?.nodes.size === 0 ? (
          <WorkflowPreviewStageMessage>{t('workflowLibrary.preview.empty')}</WorkflowPreviewStageMessage>
        ) : model && view === 'inputs' ? (
          <WorkflowPreviewInputs model={model} onShowNode={showNode} />
        ) : model ? (
          <Suspense fallback={STAGE_LOADING}>
            <DeferredWorkflowDocumentGraph
              key={`${drawing.key}:${initialNodeId ?? ''}`}
              ref={graphRef}
              initialNodeId={initialNodeId}
              model={model}
              needsModel={needsModel}
              relayout={relayout}
              onMatchCountChange={find.onMatchCountChange}
            />
          </Suspense>
        ) : templatesSnapshot.status === 'error' ? (
          <WorkflowPreviewStageMessage isAlert>
            {t('workflowLibrary.preview.unreadable', {
              message: templatesSnapshot.error ?? t('common.unknownError'),
            })}
          </WorkflowPreviewStageMessage>
        ) : (
          <WorkflowPreviewStageLoading />
        )}
      </Box>
    </Stack>
  );
};

const STAGE_LOADING = <WorkflowPreviewStageLoading />;
