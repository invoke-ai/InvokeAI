import { Flex, forwardRef } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { Components } from 'react-virtuoso';
import type { SessionQueueItemDTO } from 'services/api/types';

import type { ListContext } from './types';

const QueueListComponent: Components<SessionQueueItemDTO, ListContext>['List'] = memo(
  forwardRef((props, ref) => {
    return (
      <Flex {...props} ref={ref} flexDirection="column" gap={0.5}>
        {props.children}
      </Flex>
    );
  })
);

export default memo(QueueListComponent);
