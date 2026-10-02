import type { QueueItemReadModel } from '@features/queue/contracts';

import { ButtonGroup, Dialog, Icon, Portal } from '@chakra-ui/react';
import { Button, CloseButton } from '@platform/ui/Button';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { useNotify } from '@workbench/useNotify';
import { FileTextIcon, WandSparklesIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

export const QueueItemActions = ({ item }: { item: QueueItemReadModel }) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [jsonOpen, setJsonOpen] = useState(false);

  const onSendToCanvas = useCallback(
    () => notify.info(t('widgets.queue.sendToCanvas'), t('widgets.queue.sendToCanvasComingSoon')),
    [notify, t]
  );

  const openJson = useCallback(() => setJsonOpen(true), []);
  const closeJson = useCallback(() => setJsonOpen(false), []);

  return (
    <>
      <ButtonGroup flexWrap="wrap" minW="0" rowGap="1" size="2xs" variant="subtle" w="full">
        <Button disabled variant="ghost" onClick={onSendToCanvas}>
          <Icon as={WandSparklesIcon} boxSize="3" />
          {t('widgets.queue.sendToCanvas')}
        </Button>
        <Button onClick={openJson}>
          <Icon as={FileTextIcon} boxSize="3" />
          {t('common.viewJson')}
        </Button>
      </ButtonGroup>

      <Dialog.Root open={jsonOpen} scrollBehavior="inside" size="lg" onOpenChange={closeJson}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title>{t('widgets.queue.itemTitle', { id: item.id })}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <JsonPreview label={t('widgets.queue.itemJsonLabel', { id: item.id })} maxH="60vh" value={item} />
              </Dialog.Body>
              <Dialog.CloseTrigger asChild>
                <CloseButton />
              </Dialog.CloseTrigger>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
  );
};
