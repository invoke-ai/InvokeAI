import { Flex, forwardRef, typedMemo } from '@invoke-ai/ui-library';
import type { Components } from 'react-virtuoso';
import type { S } from 'services/api/types';

const QueueListComponent: Components<S['SessionQueueItem']>['List'] = typedMemo(
  forwardRef((props, ref) => {
    return (
      <Flex {...props} ref={ref} flexDirection="column" gap={0.5}>
        {props.children}
      </Flex>
    );
  })
);

QueueListComponent.displayName = 'QueueListComponent';

export default QueueListComponent;
