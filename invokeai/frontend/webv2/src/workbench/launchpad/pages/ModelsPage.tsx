import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

/**
 * The Launchpad's model manager: full library, add-models flow, credentials,
 * and install queue. It is the widest page, so it runs full-bleed (the view owns
 * its own padding and internal master-detail scroll).
 *
 * The manager pulls in ~6k lines of model UI and stores, none of which the
 * Launchpad needs at first paint, so it is code-split: the chunk loads the
 * first time this tab is opened (the shell's `lazyMount` keeps it off the
 * critical path until then) and stays warm afterwards. All model state lives in
 * external stores, so the view needs no workbench providers here.
 */
const ModelManagerView = lazy(() =>
  import('@workbench/launchpad/models/ModelManagerView').then((module) => ({ default: module.ModelManagerView }))
);

export const ModelsPage = () => (
  <Suspense
    fallback={
      <Center h="full">
        <Spinner color="fg.muted" size="sm" />
      </Center>
    }
  >
    <ModelManagerView />
  </Suspense>
);
