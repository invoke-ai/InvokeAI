import type { GraphPreviewSourceState, WorkflowInvocationSourceId } from '@features/workflow/ui/contracts';
import type { ReactFlowInstance } from '@xyflow/react';
import type { ReactNode } from 'react';

import { Box, Dialog, Icon, Portal, SegmentGroup, Stack, Text } from '@chakra-ui/react';
import { useWorkflowGraphPreview } from '@features/workflow/ui/WorkflowUiContext';
import { Button, JsonPreview, toaster } from '@platform/ui';
import { CheckIcon, ChevronUpIcon, CopyIcon, TriangleAlertIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GraphPreviewFlow } from './GraphPreviewFlow';
import { GraphPreviewList } from './GraphPreviewList';
import { GraphPreviewOpenAsMenu } from './GraphPreviewOpenAsMenu';
import { GraphPreviewSidePanel } from './GraphPreviewSidePanel';

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
}

type PreviewMode = 'graph' | 'list' | 'json';

const MODE_VALUES: readonly PreviewMode[] = ['graph', 'list', 'json'];
const isPreviewMode = (value: string | null): value is PreviewMode => MODE_VALUES.includes(value as PreviewMode);

const modeItems = [
  { labelKey: 'graphPreview.graph', value: 'graph' },
  { labelKey: 'graphPreview.list', value: 'list' },
  { labelKey: 'common.json', value: 'json' },
] as const;

const COPY_RESET_DELAY_MS = 1500;
const SELECT_AND_REVEAL_FIT_VIEW_OPTIONS = { duration: 150, maxZoom: 1 } as const;

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
    fontSize="sm"
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
}: GraphPreviewDialogProps) => {
  const { t } = useTranslation();
  const graphPreview = useWorkflowGraphPreview();
  const [mode, setMode] = useState<PreviewMode>('graph');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hasCopied, setHasCopied] = useState(false);
  const copyResetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flowInstanceRef = useRef<ReactFlowInstance | null>(null);
  const dialogRoute = graphPreview.getRoute(sourceId);
  const canInvoke = dialogRoute?.canInvoke === true;
  const hasInvalidReasons = source.invalidReasons.length > 0;

  const graph = source.graph;
  const selectedNode = graph?.nodes.find((node) => node.id === selectedNodeId) ?? null;

  // Every path that can close the dialog routes through this, so the
  // selection never survives into the next time it's opened.
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
  const handleModeChange = useCallback((event: { value: string | null }) => {
    if (isPreviewMode(event.value)) {
      setMode(event.value);
    }
  }, []);
  const closeDialog = useCallback(() => closeAndReset(false), [closeAndReset]);
  const invokeRoute = useCallback(() => {
    void graphPreview.invoke(sourceId).then((submitted) => {
      if (submitted) {
        closeAndReset(false);
      }
    });
  }, [graphPreview, closeAndReset, sourceId]);

  // Consumed by `handleFlowInit` the next time the flow mounts. Set by
  // `selectAndReveal` when it fires while the flow isn't the visible pane —
  // `flowInstanceRef` still points at the *previous* mount's (destroyed)
  // instance until then, so fitting it immediately would no-op on a dead
  // instance instead of the one that's about to render.
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

  // Selects a node and brings it into view. List rows and the notice banner's
  // "show node" link both switch to graph mode so the selection is always
  // visible after they fire — but if the flow isn't the mounted pane right
  // now (`mode` is still 'list'/'json' at call time), setting `mode` below
  // only *starts* the remount; the fresh instance doesn't exist until
  // `handleFlowInit` runs next render. Stash the id for that instead of
  // fitting a stale/dead instance.
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

        // Deliberately not a `useEffect` cleanup (unlike `JsonPreview`'s
        // equivalent timer) — the no-useEffect rule keeps this in the
        // handler, and a `setHasCopied` that fires after unmount is a no-op
        // in React 18+, not a leak.
        copyResetTimerRef.current = setTimeout(() => setHasCopied(false), COPY_RESET_DELAY_MS);
      })
      .catch(() => toaster.create({ title: t('graphPreview.copyFailed'), type: 'error' }));
  }, [graph, t]);

  const jsonLabel = useMemo(() => t('graphPreview.graphJsonLabel', { title: sourceLabel }), [t, sourceLabel]);
  const subtitle = useMemo(() => {
    const compiledFrom = t('graphPreview.compiledFrom', { source: sourceLabel });
    return source.isLive ? `${compiledFrom} ${t('graphPreview.liveHint')}` : compiledFrom;
  }, [t, sourceLabel, source.isLive]);

  return (
    <Dialog.Root open={isOpen} placement="center" size="xl" onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content h="80vh" maxH="80vh" maxW="min(72rem, calc(100vw - 4rem))">
            <Dialog.Header alignItems="center" flexDirection="row" justifyContent="space-between">
              <Stack gap="0.5" minW="0">
                <Dialog.Title>{t('graphPreview.title')}</Dialog.Title>
                <Text color="fg.muted" fontSize="xs">
                  {subtitle}
                </Text>
              </Stack>
              <SegmentGroup.Root size="xs" value={mode} onValueChange={handleModeChange}>
                <SegmentGroup.Indicator />
                {modeItems.map((item) => (
                  <SegmentGroup.Item key={item.value} value={item.value}>
                    <SegmentGroup.ItemHiddenInput />
                    <SegmentGroup.ItemText>{t(item.labelKey)}</SegmentGroup.ItemText>
                  </SegmentGroup.Item>
                ))}
              </SegmentGroup.Root>
            </Dialog.Header>
            <Dialog.Body display="flex" flex="1" flexDirection="column" gap="3" minH="0">
              {hasInvalidReasons ? (
                <InvalidBanner>
                  <Text>
                    {t('graphPreview.invalidTitle')} {source.invalidReasons[0]}
                  </Text>
                </InvalidBanner>
              ) : null}
              {hasInvalidReasons ? null : (
                <Box display="flex" flex="1" gap="3" minH="0">
                  <PreviewPane>
                    {!graph ? (
                      <Text color="fg.muted" fontSize="sm">
                        {t('graphPreview.noCompiledGraph', { graphId })}
                      </Text>
                    ) : mode === 'json' ? (
                      <JsonPreview h="full" label={jsonLabel} maxH="100%" value={graph} />
                    ) : mode === 'list' ? (
                      <GraphPreviewList graph={graph} onSelect={selectAndReveal} />
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
                <Button disabled={!graph} size="xs" variant="outline" onClick={copyJson}>
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
                    <Button size="xs" variant="outline">
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
                    size="xs"
                    title={dialogRoute.validationMessage}
                    onClick={invokeRoute}
                  >
                    {t('graphPreview.invokeRoute', { route: dialogRoute.label })}
                  </Button>
                ) : null}
                <Button size="xs" variant="outline" onClick={closeDialog}>
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
