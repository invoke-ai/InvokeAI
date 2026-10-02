import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

/** Lazy-load the full-bleed manager on first tab visit. External stores preserve state without workbench providers. */
const ModelManagerView = lazy(() =>
  import('@features/models/ui/ModelManagerView').then((module) => ({ default: module.ModelManagerView }))
);

const FALLBACK = (
  <Center h="full">
    <Spinner color="fg.muted" size="sm" />
  </Center>
);

export const ModelsPage = () => (
  <Suspense fallback={FALLBACK}>
    <ModelManagerView />
  </Suspense>
);
