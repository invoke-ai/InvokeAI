import type { QueueItemReadModel } from '@features/queue/contracts';
import type { QueueItemDetailReadModel } from '@features/queue/react';

import { ButtonGroup, Icon, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { useQueueItemDetail } from '@features/queue/react';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { JsonPreview } from '@platform/ui/JsonPreview';
import { Tabs } from '@platform/ui/Tabs';
import { useNotify } from '@workbench/useNotify';
import { FileTextIcon, WandSparklesIcon } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type DetailView = 'graph' | 'item' | 'results' | 'workflow';

const getDetailViews = (detail: QueueItemDetailReadModel): DetailView[] => [
  'item',
  'graph',
  'results',
  ...(detail.workflow ? (['workflow'] as const) : []),
];

const getDetailValue = (item: QueueItemReadModel, detail: QueueItemDetailReadModel, view: DetailView): unknown => {
  switch (view) {
    case 'graph':
      return detail.graph;
    case 'results':
      return detail.results;
    case 'workflow':
      return detail.workflow;
    default:
      return { ...item, errorTraceback: detail.errorTraceback, priority: detail.priority };
  }
};

/** The saved execution data, read when the dialog opens so a failed or outputless job can still be inspected. */
const QueueItemDetailBody = ({ item, open }: { item: QueueItemReadModel; open: boolean }) => {
  const { t } = useTranslation();
  const { detail, error, loadState, retry, retrying } = useQueueItemDetail(item.id, open);
  const [view, setView] = useState<DetailView>('item');
  const onViewChange = useCallback((details: { value: string }) => setView(details.value as DetailView), []);
  // A successful retry replaces the focused Retry button with the views; focus follows onto the selected one.
  const focusAfterRetryRef = useRef(false);
  const onRetry = useCallback(() => {
    focusAfterRetryRef.current = true;
    retry();
  }, [retry]);
  const focusRestoredViews = useCallback((node: HTMLDivElement | null) => {
    if (node && focusAfterRetryRef.current) {
      focusAfterRetryRef.current = false;
      node.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')?.focus();
    }
  }, []);

  if (loadState === 'error') {
    return (
      <Stack align="start" gap="2">
        <Stack gap="2" role="alert">
          <Text color="fg.error" fontSize="sm">
            {t('widgets.queue.detailLoadFailed')}
          </Text>
          {error ? (
            <Text color="fg.muted" fontSize="sm">
              {error}
            </Text>
          ) : null}
        </Stack>
        <Button loading={retrying} size="sm" variant="surface" onClick={onRetry}>
          {t('common.retry')}
        </Button>
      </Stack>
    );
  }

  if (!detail) {
    return (
      <Stack align="center" aria-busy="true" minH="24" justify="center" role="status">
        <Spinner size="md" />
        <Text color="fg.muted" fontSize="sm">
          {t('widgets.queue.detailLoading')}
        </Text>
      </Stack>
    );
  }

  const views = getDetailViews(detail);
  const activeView = views.includes(view) ? view : 'item';

  return (
    <Tabs.Root lazyMount size="lg" value={activeView} variant="line" onValueChange={onViewChange}>
      <Tabs.List ref={focusRestoredViews}>
        {views.map((candidate) => (
          <Tabs.Trigger key={candidate} value={candidate}>
            {t(`widgets.queue.detailViews.${candidate}`)}
          </Tabs.Trigger>
        ))}
      </Tabs.List>
      {views.map((candidate) => (
        <Tabs.Content key={candidate} value={candidate} pt="2">
          <JsonPreview
            copyFailedLabel={t('widgets.queue.copyFailed')}
            copyLabel={t('widgets.queue.copyJson')}
            label={t('widgets.queue.itemJsonLabel', { id: item.id })}
            maxH="55vh"
            value={getDetailValue(item, detail, candidate)}
          />
        </Tabs.Content>
      ))}
    </Tabs.Root>
  );
};

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
      <ButtonGroup flexWrap="wrap" minW="0" rowGap="1" size="sm" variant="subtle">
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
                <QueueItemDetailBody item={item} open={jsonOpen} />
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
