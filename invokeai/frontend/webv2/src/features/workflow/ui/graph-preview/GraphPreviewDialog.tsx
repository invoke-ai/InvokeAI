import type { GraphPreviewSourceState, WorkflowInvocationSourceId } from '@features/workflow/ui/contracts';
import type { ReactNode } from 'react';

import { Box, Dialog, Icon, Portal, SegmentGroup, Stack, Text } from '@chakra-ui/react';
import { useWorkflowGraphPreview } from '@features/workflow/ui/WorkflowUiContext';
import { Button, JsonPreview, toaster } from '@platform/ui';
import { CheckIcon, CopyIcon, SquareDashedIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GraphPreviewFlow } from './GraphPreviewFlow';
import { GraphPreviewSidePanel } from './GraphPreviewSidePanel';

interface GraphPreviewDialogProps {
  graphId: string;
  isOpen: boolean;
  source: GraphPreviewSourceState;
  sourceId?: WorkflowInvocationSourceId;
  /** e.g. "Generate" — used in the header subtitle and the JSON preview label. */
  sourceLabel: string;
  onOpenChange: (isOpen: boolean) => void;
}

type PreviewMode = 'graph' | 'list' | 'json';

const MODE_VALUES: readonly PreviewMode[] = ['graph', 'list', 'json'];
const isPreviewMode = (value: string | null): value is PreviewMode => MODE_VALUES.includes(value as PreviewMode);

// `list` renders the same pane as `graph` until Task 6 fills in the node list — the
// segment stays visible so the header layout is final now.
const modeItems = [
  { labelKey: 'graphPreview.graph', value: 'graph' },
  { labelKey: 'graphPreview.list', value: 'list' },
  { labelKey: 'common.json', value: 'json' },
] as const;

const COPY_RESET_DELAY_MS = 1500;

const PreviewPane = ({ children }: { children: ReactNode }) => (
  <Box flex="1" h="full" minH="0" minW="0" w="full" rounded="md" borderWidth={1} overflow="hidden">
    {children}
  </Box>
);

const NoticeBanner = ({ bg, color, children }: { bg: string; color: string; children: ReactNode }) => (
  <Box alignItems="center" bg={bg} color={color} display="flex" fontSize="sm" gap="2" px="3" py="2" rounded="md">
    <Icon as={SquareDashedIcon} boxSize="4" flexShrink={0} />
    <Text>{children}</Text>
  </Box>
);

export const GraphPreviewDialog = ({
  graphId,
  isOpen,
  source,
  sourceId,
  sourceLabel,
  onOpenChange,
}: GraphPreviewDialogProps) => {
  const { t } = useTranslation();
  const graphPreview = useWorkflowGraphPreview();
  const [mode, setMode] = useState<PreviewMode>('graph');
  const [hasCopied, setHasCopied] = useState(false);
  const copyResetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dialogRoute = graphPreview.getRoute(sourceId);
  const canInvoke = dialogRoute?.canInvoke === true;
  const hasInvalidReasons = source.invalidReasons.length > 0;

  const handleOpenChange = useCallback((event: { open: boolean }) => onOpenChange(event.open), [onOpenChange]);
  const handleModeChange = useCallback((event: { value: string | null }) => {
    if (isPreviewMode(event.value)) {
      setMode(event.value);
    }
  }, []);
  const closeDialog = useCallback(() => onOpenChange(false), [onOpenChange]);
  const invokeRoute = useCallback(() => {
    void graphPreview.invoke(sourceId).then((submitted) => {
      if (submitted) {
        onOpenChange(false);
      }
    });
  }, [graphPreview, onOpenChange, sourceId]);

  const graph = source.graph;
  const copyJson = useCallback(() => {
    const json = JSON.stringify(graph?.backendGraph ?? graph, null, 2);

    navigator.clipboard
      .writeText(json)
      .then(() => {
        setHasCopied(true);

        if (copyResetTimerRef.current !== null) {
          clearTimeout(copyResetTimerRef.current);
        }

        copyResetTimerRef.current = setTimeout(() => setHasCopied(false), COPY_RESET_DELAY_MS);
      })
      .catch(() => toaster.create({ title: 'Failed to copy JSON', type: 'error' }));
  }, [graph]);

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
              {source.notices.map((notice) => (
                <NoticeBanner key={notice.id} bg="orange.subtle" color="fg">
                  {notice.message}
                </NoticeBanner>
              ))}
              {hasInvalidReasons ? (
                <NoticeBanner bg="bg.muted" color="fg.muted">
                  {t('graphPreview.invalidTitle')} {source.invalidReasons[0]}
                </NoticeBanner>
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
                    ) : (
                      <GraphPreviewFlow graph={graph} positionHints={source.positionHints} />
                    )}
                  </PreviewPane>
                  <GraphPreviewSidePanel source={source} />
                </Box>
              )}
            </Dialog.Body>
            <Dialog.Footer justifyContent="space-between">
              <Button size="xs" variant="outline" onClick={copyJson}>
                <Icon
                  as={hasCopied ? CheckIcon : CopyIcon}
                  boxSize="3.5"
                  color={hasCopied ? 'green.solid' : undefined}
                />
                {hasCopied ? t('graphPreview.copied') : t('graphPreview.copyJson')}
              </Button>
              <Box display="flex" gap="2">
                {dialogRoute ? (
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
