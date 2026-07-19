import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

const NodeManagerView = lazy(() =>
  import('@features/nodes/ui/NodeManagerView').then((module) => ({ default: module.NodeManagerView }))
);

const FALLBACK = (
  <Center h="full">
    <Spinner color="fg.muted" size="sm" />
  </Center>
);

export const NodesPage = () => (
  <Suspense fallback={FALLBACK}>
    <NodeManagerView />
  </Suspense>
);
