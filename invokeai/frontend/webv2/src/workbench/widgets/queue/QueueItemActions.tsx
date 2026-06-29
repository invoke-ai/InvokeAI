import type { GenerateWidgetValues } from '@workbench/generation/types';

import { ButtonGroup, Dialog, Icon, Portal } from '@chakra-ui/react';
import { Button, CloseButton, JsonPreview, Tooltip } from '@workbench/components/ui';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { FileTextIcon, UndoIcon, WandSparklesIcon } from 'lucide-react';
import { useCallback, useState } from 'react';

import type { QueueServerItem } from './queueServerApi';

/**
 * Per-item actions for the RECENT details panel.
 *
 * - **Use again** recalls the exact submission snapshot into Generate (only the
 *   submitting client has it; otherwise disabled — recall from the gallery).
 * - **Send to canvas** is deferred (no canvas-import path exists in webv2 yet),
 *   so it reports "coming soon" rather than silently no-op'ing.
 * - **View JSON** opens the raw queue item (incl. its session) in a dialog.
 */
export const QueueItemActions = ({
  item,
  localGenerateValues,
}: {
  item: QueueServerItem;
  localGenerateValues: GenerateWidgetValues | null;
}) => {
  const dispatch = useWorkbenchDispatch();
  const openWidget = useOpenWorkbenchWidget();
  const notify = useNotify();
  const [jsonOpen, setJsonOpen] = useState(false);

  const canRecall = localGenerateValues !== null;

  const onUseAgain = useCallback(() => {
    if (!localGenerateValues) {
      return;
    }

    dispatch({ type: 'setGenerateSettings', values: localGenerateValues });
    openWidget('generate', { preferredRegions: ['left'] });
    notify.success('Settings recalled', 'Loaded these settings into Generate.');
  }, [dispatch, localGenerateValues, notify, openWidget]);

  const onSendToCanvas = useCallback(
    () => notify.info('Send to canvas', 'Sending queue results to the canvas is coming soon.'),
    [notify]
  );

  const openJson = useCallback(() => setJsonOpen(true), []);
  const closeJson = useCallback(() => setJsonOpen(false), []);

  return (
    <>
      {/*<HStack flexWrap="wrap" gap="1.5">
        <Tooltip content={canRecall ? 'Load these settings into Generate' : 'Recall this generation from the gallery'}>
          <Button disabled={!canRecall} size="2xs" variant="surface" onClick={onUseAgain}>
            Recall All
          </Button>
        </Tooltip>
        <Button color="fg.subtle" size="2xs" variant="ghost" onClick={onSendToCanvas}>
          Send to canvas
        </Button>
        <Button size="2xs" variant="surface" onClick={openJson}>
          View JSON
        </Button>
      </HStack>*/}

      <ButtonGroup size="2xs" variant="subtle">
        <Tooltip content={canRecall ? 'Load these settings into Generate' : 'Recall this generation from the gallery'}>
          <Button onClick={onUseAgain} disabled={!canRecall}>
            <Icon as={UndoIcon} boxSize="3" />
            Recall All
          </Button>
        </Tooltip>
        <Button disabled variant="ghost" onClick={onSendToCanvas}>
          <Icon as={WandSparklesIcon} boxSize="3" />
          Send to canvas
        </Button>
        <Button size="2xs" onClick={openJson}>
          <Icon as={FileTextIcon} boxSize="3" />
          View JSON
        </Button>
      </ButtonGroup>

      <Dialog.Root open={jsonOpen} placement="center" scrollBehavior="inside" size="lg" onOpenChange={closeJson}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" color="fg">
              <Dialog.Header>
                <Dialog.Title fontSize="sm" fontWeight="700">
                  Queue item #{item.item_id}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <JsonPreview label={`Queue item ${item.item_id} JSON`} maxH="60vh" value={item} />
              </Dialog.Body>
              <Dialog.CloseTrigger asChild>
                <CloseButton color="fg.muted" size="sm" />
              </Dialog.CloseTrigger>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
  );
};
