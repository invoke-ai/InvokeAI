import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

const NodeManagerView = lazy(() =>
  import('@workbench/launchpad/nodes/NodeManagerView').then((module) => ({ default: module.NodeManagerView }))
);

export const NodesPage = () => (
  <Suspense
    fallback={
      <Center h="full">
        <Spinner color="fg.muted" size="sm" />
      </Center>
    }
  >
    <NodeManagerView />
  </Suspense>
);
