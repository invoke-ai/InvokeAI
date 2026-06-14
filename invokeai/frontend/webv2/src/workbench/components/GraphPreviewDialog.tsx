import { Box, Dialog, Portal, SegmentGroup, Text } from '@chakra-ui/react';
import { useState } from 'react';

import { formatRoute, isInvocationRouteValid, resolveInvocationRoute } from '../invocation';
import type { GraphContract, GraphId, InvocationSourceId } from '../types';
import { useWorkbench } from '../WorkbenchContext';
import type { XYPosition } from '../workflows/types';
import { GraphPreviewFlow } from './GraphPreviewFlow';
import { Button } from './ui/Button';
import { JsonPreview } from './ui/JsonPreview';

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

export const GraphPreviewDialog = ({
  graph,
  graphId,
  isOpen,
  positionHints,
  sourceId,
  title,
  onOpenChange,
}: GraphPreviewDialogProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const [mode, setMode] = useState<PreviewMode>('nodes');
  const dialogRoute = sourceId
    ? resolveInvocationRoute(activeProject, 'dialog', { ...activeProject.invocation, sourceId, sourceLocked: true })
    : null;
  const canInvoke = dialogRoute ? isInvocationRouteValid(dialogRoute) : false;

  return (
    <Dialog.Root open={isOpen} size="xl" onOpenChange={(event) => onOpenChange(event.open)}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content h="75vh" maxH="75vh">
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
            <Dialog.Body display="flex" flexDirection="column" minH="0">
              {!graph ? (
                <Text color="fg.muted" fontSize="sm">
                  No compiled graph is available for "{graphId}" yet.
                </Text>
              ) : mode === 'nodes' ? (
                <Box flex="1" minH="0">
                  <GraphPreviewFlow graph={graph} positionHints={positionHints} />
                </Box>
              ) : (
                <JsonPreview label={`${title} graph JSON`} value={graph} />
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
