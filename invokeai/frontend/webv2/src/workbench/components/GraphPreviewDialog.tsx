import { Button, Code, Dialog, Portal, ScrollArea, Stack, Text } from '@chakra-ui/react';

import { formatRoute, isInvocationRouteValid, resolveInvocationRoute } from '../invocation';
import type { GraphContract, GraphId, InvocationSourceId } from '../types';
import { useWorkbench } from '../WorkbenchContext';

interface GraphPreviewDialogProps {
  graph: GraphContract | null;
  graphId: GraphId;
  isOpen: boolean;
  sourceId?: InvocationSourceId;
  title: string;
  onOpenChange: (isOpen: boolean) => void;
}

export const GraphPreviewDialog = ({
  graph,
  graphId,
  isOpen,
  sourceId,
  title,
  onOpenChange,
}: GraphPreviewDialogProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const dialogRoute = sourceId
    ? resolveInvocationRoute(activeProject, 'dialog', { ...activeProject.invocation, sourceId, sourceLocked: true })
    : null;
  const canInvoke = dialogRoute ? isInvocationRouteValid(dialogRoute) : false;

  return (
    <Dialog.Root open={isOpen} onOpenChange={(event) => onOpenChange(event.open)}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content bg="bg.surfaceRaised" borderColor="border.emphasis" borderWidth="1px" color="fg.default">
            <Dialog.Header>
              <Dialog.Title>{title} Graph Preview</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="3">
                <Text color="fg.muted" fontSize="sm">
                  Read-only shell for graph inspection. Full graph preview rendering lands in later phases.
                </Text>
                <ScrollArea.Root maxH="20rem" size="xs" variant="hover">
                  <ScrollArea.Viewport maxH="20rem">
                    <ScrollArea.Content>
                      <Code display="block" p="3" whiteSpace="pre-wrap">
                        {JSON.stringify(graph ?? { id: graphId, status: 'not-created-yet' }, null, 2)}
                      </Code>
                    </ScrollArea.Content>
                  </ScrollArea.Viewport>
                  <ScrollArea.Scrollbar>
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Scrollbar orientation="horizontal">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Corner />
                </ScrollArea.Root>
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              {dialogRoute ? (
                <Button
                  bg="accent.invoke"
                  color="accent.invokeFg"
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
