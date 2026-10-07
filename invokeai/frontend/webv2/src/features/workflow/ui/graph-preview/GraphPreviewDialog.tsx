import type { GraphPreviewSourceState, WorkflowInvocationSourceId } from '@features/workflow/ui/contracts';
import type { ReactFlowInstance } from '@xyflow/react';
import type { ReactNode } from 'react';

import { Box, Center, Dialog, Icon, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { localizeForLoopValidationReason } from '@features/workflow/core/forLoops';
import { useWorkflowGraphPreview } from '@features/workflow/ui/WorkflowUiContext';
import { Button, JsonPreview, SegmentTabs, segmentTabsPanelId, segmentTabsTabId, toaster } from '@platform/ui';
import { CheckIcon, ChevronUpIcon, CopyIcon, TriangleAlertIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GraphPreviewFlow } from './GraphPreviewFlow';
import { GraphPreviewOpenAsMenu } from './GraphPreviewOpenAsMenu';
import { GraphPreviewSidePanel } from './GraphPreviewSidePanel';

// Loaded through this module so the snapshot shares the dialog's chunk; a second entry splits the flow out of it.
export { GraphPreviewSnapshot } from './GraphPreviewSnapshot';

interface GraphPreviewDialogProps {
  graphId: string;
  isOpen: boolean;
  source: GraphPreviewSourceState;
  sourceId?: WorkflowInvocationSourceId;
  /** e.g. "Generate" — used in the header subtitle and the JSON preview label. */
  sourceLabel: string;
  /** Hides the footer Invoke button only — Copy JSON and the Open-as menu stay. For sources with no invocation route (e.g. a library entry, previewed before it's ever opened into a project). */
  hideInvoke?: boolean;
  onOpenChange: (isOpen: boolean) => void;
  /** Conditional hosts must unmount after close completion so exit animation can run. */
  onExitComplete?: () => void;
}

type PreviewMode = 'graph' | 'list' | 'json';

const modeItems = [
  { labelKey: 'graphPreview.graph', value: 'graph' },
  { labelKey: 'graphPreview.list', value: 'list' },
  { labelKey: 'common.json', value: 'json' },
] as const satisfies readonly { labelKey: string; value: PreviewMode }[];

const COPY_RESET_DELAY_MS = 1500;
const SELECT_AND_REVEAL_FIT_VIEW_OPTIONS = { duration: 150, maxZoom: 1 } as const;

// List mode brings the virtualized list; load it when chosen so the workflow route's startup does not carry it.
const GraphPreviewList = lazy(() =>
  import('./GraphPreviewList').then((module) => ({ default: module.GraphPreviewList }))
);
const ListFallback = () => {
  const { t } = useTranslation();

  return (
    <Center aria-label={t('common.loading')} h="full" role="status">
      <Spinner color="fg.subtle" size="lg" />
    </Center>
  );
};
const LIST_FALLBACK = <ListFallback />;

const PreviewPane = ({ children }: { children: ReactNode }) => (
  <Box flex="1" h="full" minH="0" minW="0" w="full" rounded="md" borderWidth={1} overflow="hidden">
    {children}
  </Box>
);

/** The strip above the graph when compilation is blocked — notices about a *valid* graph (e.g. randomized seed) live inline in the summary panel instead. */
const InvalidBanner = ({ children }: { children: ReactNode }) => (
  <Box
    alignItems="center"
    bg="bg.muted"
    color="fg.muted"
    display="flex"
    fontSize="lg"
    gap="2"
    px="3"
    py="2"
    rounded="md"
  >
    <Icon as={TriangleAlertIcon} boxSize="4" flexShrink={0} />
    <Box flex="1" minW="0">
      {children}
    </Box>
  </Box>
);

export const GraphPreviewDialog = ({
  graphId,
  isOpen,
  source,
  sourceId,
  sourceLabel,
  hideInvoke = false,
  onOpenChange,
  onExitComplete,
}: GraphPreviewDialogProps) => {
  const { t } = useTranslation();
  const graphPreview = useWorkflowGraphPreview();
  const modeTabsIdBase = useId();
  const [mode, setMode] = useState<PreviewMode>('graph');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hasCopied, setHasCopied] = useState(false);
  const copyResetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flowInstanceRef = useRef<ReactFlowInstance | null>(null);
  const dialogRoute = graphPreview.getRoute(sourceId);
  const canInvoke = dialogRoute?.canInvoke === true;
  const validationMessage = dialogRoute?.validationMessage
    ? localizeForLoopValidationReason(dialogRoute.validationMessage, t)
    : undefined;
  const hasInvalidReasons = source.invalidReasons.length > 0;

  const graph = source.graph;
  const selectedNode = graph?.nodes.find((node) => node.id === selectedNodeId) ?? null;

  // Reset selection on every close path so reopening starts clean.
  const closeAndReset = useCallback(
    (open: boolean) => {
      if (!open) {
        setSelectedNodeId(null);
      }

      onOpenChange(open);
    },
    [onOpenChange]
  );

  const handleOpenChange = useCallback((event: { open: boolean }) => closeAndReset(event.open), [closeAndReset]);
  const handleModeChange = useCallback((value: PreviewMode) => setMode(value), []);
  const closeDialog = useCallback(() => closeAndReset(false), [closeAndReset]);
  const invokeRoute = useCallback(() => {
    void graphPreview.invoke(sourceId).then((submitted) => {
      if (submitted) {
        closeAndReset(false);
      }
    });
  }, [graphPreview, closeAndReset, sourceId]);

  // Queue reveals for the next flow initialization; the previous mount's instance may already be destroyed.
  const pendingRevealNodeIdRef = useRef<string | null>(null);

  const handleFlowInit = useCallback((instance: ReactFlowInstance) => {
    flowInstanceRef.current = instance;

    const pendingNodeId = pendingRevealNodeIdRef.current;

    if (pendingNodeId !== null) {
      pendingRevealNodeIdRef.current = null;
      void instance.fitView({ ...SELECT_AND_REVEAL_FIT_VIEW_OPTIONS, nodes: [{ id: pendingNodeId }] });
    }
  }, []);
  const handleFlowNodeSelect = useCallback((nodeId: string | null) => setSelectedNodeId(nodeId), []);
  const handleBack = useCallback(() => setSelectedNodeId(null), []);
  const handleProvenanceClick = useCallback(() => {
    graphPreview.focusSource(sourceId);
    closeAndReset(false);
  }, [graphPreview, sourceId, closeAndReset]);

  // Switching from list/JSON starts a flow remount; defer fitView until its fresh instance initializes.
  const selectAndReveal = useCallback(
    (nodeId: string) => {
      setSelectedNodeId(nodeId);

      const instance = flowInstanceRef.current;
      const isFlowMounted = mode === 'graph' && instance !== null;

      if (isFlowMounted) {
        void instance.fitView({ ...SELECT_AND_REVEAL_FIT_VIEW_OPTIONS, nodes: [{ id: nodeId }] });
      } else {
        pendingRevealNodeIdRef.current = nodeId;
      }

      setMode('graph');
    },
    [mode]
  );

  const copyJson = useCallback(() => {
    const json = JSON.stringify(graph?.backendGraph ?? graph, null, 2);

    navigator.clipboard
      .writeText(json)
      .then(() => {
        setHasCopied(true);

        if (copyResetTimerRef.current !== null) {
          clearTimeout(copyResetTimerRef.current);
        }

        // Keep copy-feedback timing in the event handler; a post-unmount state update is ignored.
        copyResetTimerRef.current = setTimeout(() => setHasCopied(false), COPY_RESET_DELAY_MS);
      })
      .catch(() => toaster.create({ title: t('graphPreview.copyFailed'), type: 'error' }));
  }, [graph, t]);

  const modeTabs = useMemo(() => modeItems.map((item) => ({ id: item.value, label: t(item.labelKey) })), [t]);
  const jsonLabel = useMemo(() => t('graphPreview.graphJsonLabel', { title: sourceLabel }), [t, sourceLabel]);
  const subtitle = useMemo(() => {
    const compiledFrom = t('graphPreview.compiledFrom', { source: sourceLabel });
    return source.isLive ? `${compiledFrom} ${t('graphPreview.liveHint')}` : compiledFrom;
  }, [t, sourceLabel, source.isLive]);

  return (
    <Dialog.Root open={isOpen} size="xl" onExitComplete={onExitComplete} onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content h="min(46rem, 85vh)" maxH="85vh" maxW="min(72rem, calc(100vw - 4rem))">
            <Dialog.Header alignItems="center" flexDirection="row" justifyContent="space-between">
              <Stack gap="0.5" minW="0">
                <Dialog.Title>{t('graphPreview.title')}</Dialog.Title>
                <Dialog.Description>{subtitle}</Dialog.Description>
              </Stack>
              <SegmentTabs
                activeId={mode}
                ariaLabel={t('graphPreview.title')}
                idBase={modeTabsIdBase}
                isCompact
                tabs={modeTabs}
                onSelect={handleModeChange}
              />
            </Dialog.Header>
            <Dialog.Body
              aria-labelledby={segmentTabsTabId(modeTabsIdBase, mode)}
              display="flex"
              flex="1"
              flexDirection="column"
              gap="3"
              id={segmentTabsPanelId(modeTabsIdBase)}
              minH="0"
              role="tabpanel"
            >
              {hasInvalidReasons ? (
                <InvalidBanner>
                  <Text>
                    {t('graphPreview.invalidTitle')} {localizeForLoopValidationReason(source.invalidReasons[0], t)}
                  </Text>
                </InvalidBanner>
              ) : null}
              {hasInvalidReasons ? null : (
                <Box display="flex" flex="1" gap="3" minH="0">
                  <PreviewPane>
                    {!graph ? (
                      <Text color="fg.muted" fontSize="lg">
                        {t('graphPreview.noCompiledGraph', { graphId })}
                      </Text>
                    ) : mode === 'json' ? (
                      <JsonPreview h="full" label={jsonLabel} maxH="100%" value={graph} />
                    ) : mode === 'list' ? (
                      // The inset keeps the first row off the pane's rounded border.
                      <Box display="flex" flexDirection="column" h="full" pt="2">
                        <Suspense fallback={LIST_FALLBACK}>
                          <GraphPreviewList graph={graph} selectedNodeId={selectedNodeId} onSelect={selectAndReveal} />
                        </Suspense>
                      </Box>
                    ) : (
                      <GraphPreviewFlow
                        graph={graph}
                        positionHints={source.positionHints}
                        selectedNodeId={selectedNodeId}
                        onInit={handleFlowInit}
                        onNodeSelect={handleFlowNodeSelect}
                      />
                    )}
                  </PreviewPane>
                  {mode === 'graph' ? (
                    <GraphPreviewSidePanel
                      source={source}
                      selectedNode={selectedNode}
                      onBack={handleBack}
                      onProvenanceClick={handleProvenanceClick}
                      onShowNode={selectAndReveal}
                    />
                  ) : null}
                </Box>
              )}
            </Dialog.Body>
            <Dialog.Footer justifyContent="space-between">
              <Box display="flex" gap="2">
                <Button disabled={!graph} variant="outline" onClick={copyJson}>
                  <Icon
                    as={hasCopied ? CheckIcon : CopyIcon}
                    boxSize="3.5"
                    color={hasCopied ? 'green.solid' : undefined}
                  />
                  {hasCopied ? t('graphPreview.copied') : t('graphPreview.copyJson')}
                </Button>
                {graph ? (
                  <GraphPreviewOpenAsMenu
                    graph={graph}
                    sourceId={sourceId}
                    sourceLabel={sourceLabel}
                    onClose={closeDialog}
                  >
                    <Button variant="outline">
                      {t('graphPreview.openAs')}
                      <Icon as={ChevronUpIcon} boxSize="3.5" />
                    </Button>
                  </GraphPreviewOpenAsMenu>
                ) : null}
              </Box>
              <Box display="flex" gap="2">
                {!hideInvoke && dialogRoute ? (
                  <Button
                    aria-disabled={!canInvoke}
                    cursor={canInvoke ? undefined : 'not-allowed'}
                    opacity={canInvoke ? undefined : 0.6}
                    title={validationMessage}
                    onClick={invokeRoute}
                  >
                    {t('graphPreview.invokeRoute', { route: dialogRoute.label })}
                  </Button>
                ) : null}
                <Button variant="ghost" onClick={closeDialog}>
                  {t('common.close')}
                </Button>
              </Box>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
