import { Flex, forwardRef } from '@chakra-ui/react';
import { memo } from 'react';
import { Components } from 'react-virtuoso';
import { SessionQueueItemDTO } from 'services/api/types';
import { ListContext } from './types';

const QueueListComponent: Components<SessionQueueItemDTO, ListContext>['List'] =
  memo(
    forwardRef((props, ref) => {
      return (
        <Flex {...props} ref={ref} flexDirection="column" gap={0.5}>
          {props.children}
        </Flex>
      );
    })
  );

export default memo(QueueListComponent);
