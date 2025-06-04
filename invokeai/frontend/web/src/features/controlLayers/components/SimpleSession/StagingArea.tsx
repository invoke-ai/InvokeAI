/* eslint-disable i18next/no-literal-string */

import { Divider, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { getQueueItemElementId } from 'features/controlLayers/components/SimpleSession/shared';
import { StagingAreaContent } from 'features/controlLayers/components/SimpleSession/StagingAreaContent';
import { StagingAreaHeader } from 'features/controlLayers/components/SimpleSession/StagingAreaHeader';
import { useStagingAreaKeyboardNav } from 'features/controlLayers/components/SimpleSession/use-staging-keyboard-nav';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useListAllQueueItemsQuery } from 'services/api/endpoints/queue';
import type { S } from 'services/api/types';
import { $socket, setProgress } from 'services/events/stores';

const LIST_ALL_OPTIONS = {
  selectFromResult: ({ data }) => {
    if (!data) {
      return { items: EMPTY_ARRAY };
    }
    return { items: data.filter(({ status }) => status !== 'canceled') };
  },
} satisfies Parameters<typeof useListAllQueueItemsQuery>[1];

export const StagingArea = memo(() => {
  const ctx = useCanvasSessionContext();
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const [autoSwitch, setAutoSwitch] = useState(true);
  const { items } = useListAllQueueItemsQuery({ destination: ctx.session.id }, LIST_ALL_OPTIONS);
  const selectedItem = useMemo(() => {
    if (items.length === 0) {
      return null;
    }
    if (selectedItemId === null) {
      return null;
    }
    return items.find(({ item_id }) => item_id === selectedItemId) ?? null;
  }, [items, selectedItemId]);
  const selectedItemIndex = useMemo(() => {
    if (items.length === 0) {
      return null;
    }
    if (selectedItemId === null) {
      return null;
    }
    return items.findIndex(({ item_id }) => item_id === selectedItemId) ?? null;
  }, [items, selectedItemId]);

  const onSelectItemId = useCallback((item_id: number | null) => {
    setSelectedItemId(item_id);
    if (item_id !== null) {
      document.getElementById(getQueueItemElementId(item_id))?.scrollIntoView();
    }
  }, []);

  useStagingAreaKeyboardNav(items, selectedItemId, onSelectItemId);

  useEffect(() => {
    if (items.length === 0) {
      onSelectItemId(null);
      return;
    }
    if (selectedItemId === null && items.length > 0) {
      onSelectItemId(items[0]?.item_id ?? null);
      return;
    }
  }, [items, onSelectItemId, selectedItem, selectedItemId]);

  const socket = useStore($socket);
  useEffect(() => {
    if (!socket) {
      return;
    }

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      if (data.destination !== ctx.session.id) {
        return;
      }
      if (data.status === 'in_progress' && autoSwitch) {
        onSelectItemId(data.item_id);
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [autoSwitch, ctx.$progressData, ctx.session.id, onSelectItemId, socket]);

  useEffect(() => {
    if (!socket) {
      return;
    }
    const onProgress = (data: S['InvocationProgressEvent']) => {
      if (data.destination !== ctx.session.id) {
        return;
      }
      setProgress(ctx.$progressData, data);
    };
    socket.on('invocation_progress', onProgress);

    return () => {
      socket.off('invocation_progress', onProgress);
    };
  }, [ctx.$progressData, ctx.session.id, socket]);

  return (
    <Flex flexDir="column" gap={2} w="full" h="full" minW={0} minH={0}>
      <StagingAreaHeader autoSwitch={autoSwitch} setAutoSwitch={setAutoSwitch} />
      <Divider />
      {items.length > 0 && (
        <StagingAreaContent
          items={items}
          selectedItem={selectedItem}
          selectedItemId={selectedItemId}
          selectedItemIndex={selectedItemIndex}
          onChangeAutoSwitch={setAutoSwitch}
          onSelectItemId={onSelectItemId}
        />
      )}
      {items.length === 0 && (
        <Flex w="full" h="full" alignItems="center" justifyContent="center">
          <Text>No generations</Text>
        </Flex>
      )}
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';
