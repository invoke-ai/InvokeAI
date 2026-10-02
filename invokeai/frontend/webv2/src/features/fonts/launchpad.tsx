import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

/** Launchpad entry kept separate so the editor's initial chunk does not load the library view. */
const LazyFontsPage = lazy(() => import('./ui/FontsPage').then((module) => ({ default: module.FontsPage })));

const FALLBACK = (
  <Center h="full">
    <Spinner color="fg.muted" size="sm" />
  </Center>
);

export const FontsPage = () => (
  <Suspense fallback={FALLBACK}>
    <LazyFontsPage />
  </Suspense>
);
