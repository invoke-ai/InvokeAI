import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

/**
 * The Launchpad's nodes manager: pack library, add-nodes flow, and install
 * activity. The Launchpad doesn't need any of it at first paint, so it is
 * code-split: the chunk loads the first time this tab is opened (the shell's
 * `lazyMount` keeps it off the critical path until then) and stays warm
 * afterwards. All pack state lives in external stores, so the view needs no
 * workbench providers here.
 */
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
