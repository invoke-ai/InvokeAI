import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

/** Launchpad entry kept separate so the settings catalog and editors load only when the page opens. */
const LazyPreferencesPage = lazy(() =>
  import('./PreferencesPage').then((module) => ({ default: module.PreferencesPage }))
);

const FALLBACK = (
  <Center h="full">
    <Spinner color="fg.muted" size="sm" />
  </Center>
);

export const PreferencesPage = () => (
  <Suspense fallback={FALLBACK}>
    <LazyPreferencesPage />
  </Suspense>
);
