import type { XYPosition } from '@features/workflow/contracts';
import type { WorkflowInvocationSourceId, WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

import { Box, Dialog, Portal, SegmentGroup, Text } from '@chakra-ui/react';
import { useWorkflowGraphPreview } from '@features/workflow/ui/WorkflowUiContext';
import { Button, JsonPreview } from '@platform/ui';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { GraphPreviewFlow } from './GraphPreviewFlow';

interface GraphPreviewDialogProps {
  graph: WorkflowPreviewGraph | null;
  graphId: string;
  isOpen: boolean;
  /** Editor positions keyed by node id, when the graph mirrors an editable document. */
  positionHints?: Record<string, XYPosition>;
  sourceId?: WorkflowInvocationSourceId;
  title: string;
  onOpenChange: (isOpen: boolean) => void;
}

type PreviewMode = 'nodes' | 'json';

const modeItems = [
  { labelKey: 'graphPreview.nodes', value: 'nodes' },
  { labelKey: 'common.json', value: 'json' },
];

const PreviewPane = ({ children }: { children: ReactNode }) => (
  <Box flex="1" h="full" minH="0" minW="0" w="full" rounded="md" borderWidth={1} overflow="hidden">
    {children}
  </Box>
);

export const GraphPreviewDialog = ({
  graph,
  graphId,
  isOpen,
  positionHints,
  sourceId,
  title,
  onOpenChange,
}: GraphPreviewDialogProps) => {
  const { t } = useTranslation();
  const graphPreview = useWorkflowGraphPreview();
  const [mode, setMode] = useState<PreviewMode>('nodes');
  const dialogRoute = graphPreview.getRoute(sourceId);
  const canInvoke = dialogRoute?.canInvoke === true;

  const handleOpenChange = useCallback((event: { open: boolean }) => onOpenChange(event.open), [onOpenChange]);
  const handleModeChange = useCallback(
    (event: { value: string | null }) => setMode(event.value === 'json' ? 'json' : 'nodes'),
    []
  );
  const closeDialog = useCallback(() => onOpenChange(false), [onOpenChange]);
  const invokeRoute = useCallback(() => {
    void graphPreview.invoke(sourceId).then((submitted) => {
      if (submitted) {
        onOpenChange(false);
      }
    });
  }, [graphPreview, onOpenChange, sourceId]);
  const jsonLabel = useMemo(() => t('graphPreview.graphJsonLabel', { title }), [t, title]);

  return (
    <Dialog.Root open={isOpen} size="xl" onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content h="62vh" maxH="62vh">
            <Dialog.Header alignItems="center" flexDirection="row" justifyContent="space-between">
              <Dialog.Title>{t('graphPreview.title', { title })}</Dialog.Title>
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
            <Dialog.Body display="flex" flex="1" flexDirection="column" minH="0">
              {!graph ? (
                <Text color="fg.muted" fontSize="sm">
                  {t('graphPreview.noCompiledGraph', { graphId })}
                </Text>
              ) : mode === 'nodes' ? (
                <PreviewPane>
                  <GraphPreviewFlow graph={graph} positionHints={positionHints} />
                </PreviewPane>
              ) : (
                <PreviewPane>
                  <JsonPreview h="full" label={jsonLabel} maxH="100%" value={graph} />
                </PreviewPane>
              )}
            </Dialog.Body>
            <Dialog.Footer>
              {dialogRoute ? (
                <Button
                  aria-disabled={!canInvoke}
                  cursor={canInvoke ? undefined : 'not-allowed'}
                  opacity={canInvoke ? undefined : 0.6}
                  size="sm"
                  title={dialogRoute.validationMessage}
                  onClick={invokeRoute}
                >
                  {t('graphPreview.invokeRoute', { route: dialogRoute.label })}
                </Button>
              ) : null}
              <Button size="sm" variant="outline" onClick={closeDialog}>
                {t('common.close')}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
