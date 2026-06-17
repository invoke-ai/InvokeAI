import type { GraphContract, GraphId, InvocationSourceId } from '@workbench/types';
import type { XYPosition } from '@workbench/workflows/types';

import { Box, Dialog, Portal, SegmentGroup, Text } from '@chakra-ui/react';
import { Button, JsonPreview } from '@workbench/components/ui';
import { formatRoute, isInvocationRouteValid, resolveInvocationRoute } from '@workbench/invocation';
import { ensureModelsLoaded, useModelsSnapshot } from '@workbench/models/modelsStore';
import { useActiveProject, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useEffect, useState, type ReactNode } from 'react';

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
  const activeProject = useActiveProject();
  const dispatch = useWorkbenchDispatch();
  const { models, status: modelsStatus } = useModelsSnapshot();
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const [mode, setMode] = useState<PreviewMode>('nodes');
  const dialogRoute = sourceId
    ? resolveInvocationRoute(
        activeProject,
        'dialog',
        { ...activeProject.invocation, sourceId, sourceLocked: true },
        availabilityModels
      )
    : null;
  const canInvoke = dialogRoute ? isInvocationRouteValid(dialogRoute) : false;

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  return (
    <Dialog.Root open={isOpen} size="xl" onOpenChange={(event) => onOpenChange(event.open)}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content h="62vh" maxH="62vh">
            <Dialog.Header alignItems="center" flexDirection="row" justifyContent="space-between">
              <Dialog.Title>{title} Graph Preview</Dialog.Title>
              <SegmentGroup.Root
                size="xs"
                value={mode}
                onValueChange={(event) => setMode(event.value === 'json' ? 'json' : 'nodes')}
              >
                <SegmentGroup.Indicator />
                <SegmentGroup.Items
                  items={[
                    { label: 'Nodes', value: 'nodes' },
                    { label: 'JSON', value: 'json' },
                  ]}
                />
              </SegmentGroup.Root>
            </Dialog.Header>
            <Dialog.Body display="flex" flex="1" flexDirection="column" minH="0">
              {!graph ? (
                <Text color="fg.muted" fontSize="sm">
                  No compiled graph is available for "{graphId}" yet.
                </Text>
              ) : mode === 'nodes' ? (
                <PreviewPane>
                  <GraphPreviewFlow graph={graph} positionHints={positionHints} />
                </PreviewPane>
              ) : (
                <PreviewPane>
                  <JsonPreview h="full" label={`${title} graph JSON`} maxH="100%" value={graph} />
                </PreviewPane>
              )}
            </Dialog.Body>
            <Dialog.Footer>
              {dialogRoute ? (
                <Button
                  disabled={!canInvoke}
                  size="sm"
                  title={dialogRoute.validationMessage}
                  onClick={() => {
                    if (!canInvoke) {
                      return;
                    }

                    dispatch({
                      backendSupportsCancellation: true,
                      models: availabilityModels,
                      route: dialogRoute,
                      type: 'submitResolvedInvocationSnapshot',
                    });
                    onOpenChange(false);
                  }}
                >
                  Invoke {formatRoute(dialogRoute)}
                </Button>
              ) : null}
              <Button size="sm" variant="outline" onClick={() => onOpenChange(false)}>
                Close
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
