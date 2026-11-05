import type { ComponentProps } from 'react';

import { Box, Center, Spinner } from '@chakra-ui/react';
import { RetryBoundary } from '@platform/ui/RetryBoundary';
import { Suspense, use } from 'react';
import { useTranslation } from 'react-i18next';

import type * as UsersModule from './ui/UsersPage';
import type { UsersPage as UsersPageView } from './ui/UsersPage';

let resource: Promise<typeof UsersModule> | undefined;
let importFailed = false;
const load = () =>
  (resource ??= import('./ui/UsersPage').catch((error: unknown) => {
    importFailed = true;
    throw error;
  }));
const retry = () => {
  if (importFailed) {
    // Browsers retain failed module fetches in this document's module map; a fresh document can fetch them again.
    window.location.reload();
    return Promise.resolve();
  }
  return load();
};
const LoadedUsersPage = (props: ComponentProps<typeof UsersPageView>) => {
  const { UsersPage } = use(load());
  return <UsersPage {...props} />;
};
const FALLBACK = (
  <Center h="full">
    <Spinner />
  </Center>
);

export const UsersPage = (props: ComponentProps<typeof UsersPageView>) => {
  const { t } = useTranslation();
  return (
    <Center h="full">
      <RetryBoundary message={t('users.loadFailed')} retryLabel={t('common.retry')} retry={retry}>
        <Box h="full" w="full" minW="0">
          <Suspense fallback={FALLBACK}>
            <LoadedUsersPage {...props} />
          </Suspense>
        </Box>
      </RetryBoundary>
    </Center>
  );
};
