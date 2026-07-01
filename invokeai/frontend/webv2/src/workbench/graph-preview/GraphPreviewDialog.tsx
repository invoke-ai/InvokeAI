import type { GraphContract, GraphId, InvocationSourceId } from '@workbench/types';
import type { XYPosition } from '@workbench/workflows/types';

import { Box, Dialog, Portal, SegmentGroup, Text } from '@chakra-ui/react';
import { Button, JsonPreview } from '@workbench/components/ui';
import {
  createInvocationRouteInputSelector,
  formatRoute,
  isInvocationRouteValid,
  resolveInvocationRoute,
  resolveInvocationRouteInput,
} from '@workbench/invocation';
import { ensureModelsLoaded, useModelsSelector } from '@workbench/models/modelsStore';
import { flushGenerateDrafts } from '@workbench/widgets/generate/generateDraftRegistry';
import { useActiveProjectSelector, useWorkbenchDispatch, useWorkbenchStore } from '@workbench/WorkbenchContext';
import { useInvocationTemplatesSelector } from '@workbench/workflows/templates';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { GraphPreviewFlow } from './GraphPreviewFlow';

interface GraphPreviewDialogProps {
  graph: GraphContract | null;
  graphId: GraphId;
  isOpen: boolean;
  /** Editor positions keyed by node id, when the graph mirrors an editable document. */
  positionHints?: Record<string, XYPosition>;
  sourceId?: InvocationSourceId;
  title: string;
  onOpenChange: (isOpen: boolean) => void;
}

type PreviewMode = 'nodes' | 'json';

const selectInvocationRouteInput = createInvocationRouteInputSelector();
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
  const routeInput = useActiveProjectSelector(selectInvocationRouteInput);
  const dispatch = useWorkbenchDispatch();
  const store = useWorkbenchStore();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const [mode, setMode] = useState<PreviewMode>('nodes');

  useInvocationTemplatesSelector((snapshot) => snapshot.status);

  const dialogRoute = sourceId
    ? resolveInvocationRouteInput(
        routeInput,
        'dialog',
        { ...routeInput.invocation, sourceId, sourceLocked: true },
        availabilityModels
      )
    : null;
  const canInvoke = dialogRoute ? isInvocationRouteValid(dialogRoute) : false;

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const handleOpenChange = useCallback((event: { open: boolean }) => onOpenChange(event.open), [onOpenChange]);
  const handleModeChange = useCallback(
    (event: { value: string | null }) => setMode(event.value === 'json' ? 'json' : 'nodes'),
    []
  );
  const closeDialog = useCallback(() => onOpenChange(false), [onOpenChange]);
  const invokeRoute = useCallback(() => {
    flushGenerateDrafts();
    const postFlushProject = store.getSnapshot().activeProject;
    const postFlushRoute = sourceId
      ? resolveInvocationRoute(
          postFlushProject,
          'dialog',
          { ...postFlushProject.invocation, sourceId, sourceLocked: true },
          availabilityModels
        )
      : null;

    if (!postFlushRoute || !isInvocationRouteValid(postFlushRoute)) {
      return;
    }

    dispatch({
      backendSupportsCancellation: true,
      models: availabilityModels,
      route: postFlushRoute,
      type: 'submitResolvedInvocationSnapshot',
    });
    onOpenChange(false);
  }, [availabilityModels, dispatch, onOpenChange, sourceId, store]);
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
                  {t('graphPreview.invokeRoute', { route: formatRoute(dialogRoute) })}
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
