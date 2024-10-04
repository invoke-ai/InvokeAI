import { Flex, forwardRef, typedMemo } from '@invoke-ai/ui-library';
import type { Components } from 'react-virtuoso';
import type { SessionQueueItemDTO } from 'services/api/types';

import type { ListContext } from './types';

const QueueListComponent: Components<SessionQueueItemDTO, ListContext>['List'] = typedMemo(
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
