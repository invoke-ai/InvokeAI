import type { SystemStyleObject } from '@chakra-ui/react';

import { Box } from '@chakra-ui/react';
import { Scrollable } from '@workbench/components/ui';
import { UsersManagementPanel } from '@workbench/users';
import { useTranslation } from 'react-i18next';

/**
 * Admin-only section: create, edit, and remove users. Mirrors the Projects
 * page's centered measure and self-scroll so the chrome stays put.
 */
const USERS_PAGE_MEASURE_SX: SystemStyleObject = {
  maxW: '6xl',
  mx: 'auto',
  p: { base: 4, md: 8 },
  w: 'full',
};

export const UsersPage = () => {
  const { t } = useTranslation();

  return (
    <Scrollable h="full" label={t('users.management')} minH="0">
      <Box css={USERS_PAGE_MEASURE_SX}>
        <UsersManagementPanel />
      </Box>
    </Scrollable>
  );
};
