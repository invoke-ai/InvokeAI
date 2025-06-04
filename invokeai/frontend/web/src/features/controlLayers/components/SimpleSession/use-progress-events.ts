import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useEffect } from 'react';
import type { S } from 'services/api/types';
import { $socket, setProgress } from 'services/events/stores';

export const useProgressEvents = () => {
  const ctx = useCanvasSessionContext();
  const socket = useStore($socket);
  useEffect(() => {
    if (!socket) {
      return;
    }

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      if (data.destination !== ctx.session.id) {
        return;
      }
      if (data.status === 'completed' && ctx.$autoSwitch.get()) {
        ctx.$selectedItemId.set(data.item_id);
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [ctx.$autoSwitch, ctx.$progressData, ctx.$selectedItemId, ctx.session.id, socket]);

  useEffect(() => {
    if (!socket) {
      return;
    }
    const onProgress = (data: S['InvocationProgressEvent']) => {
      if (data.destination !== ctx.session.id) {
        return;
      }
      // TODO: clear progress when done w/ it memory leak
      setProgress(ctx.$progressData, data);
    };
    socket.on('invocation_progress', onProgress);

    return () => {
      socket.off('invocation_progress', onProgress);
    };
  }, [ctx.$progressData, ctx.session.id, socket]);
};
