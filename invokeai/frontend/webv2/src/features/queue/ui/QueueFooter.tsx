import { InvocationCacheFooter } from './InvocationCacheFooter';
import { ModelCacheFooter } from './ModelCacheFooter';
import { useQueueUi } from './QueueUiContext';

/** Both server caches sit under the queue; the invocation cache is only readable by administrators. */
export const QueueFooter = () => {
  const { canManageInvocationCache } = useQueueUi();

  return (
    <>
      {canManageInvocationCache ? <InvocationCacheFooter /> : null}
      <ModelCacheFooter />
    </>
  );
};
