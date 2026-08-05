import { PageShell } from '@platform/ui/PageShell';
import { useTranslation } from 'react-i18next';

import { UsersManagementPanel } from './UserManagement';

/**
 * Admin-only section: create, edit, and remove users. Shares the Launchpad's
 * page chrome, so it carries the same heading, measure, and self-scroll as the
 * project library instead of its own copy of them.
 */
export const UsersPage = () => {
  const { t } = useTranslation();

  return (
    <PageShell description={t('users.description')} regionLabel={t('users.management')} title={t('users.title')}>
      <UsersManagementPanel />
    </PageShell>
  );
};
