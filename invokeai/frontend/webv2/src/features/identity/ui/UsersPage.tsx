import { PageShell } from '@platform/ui/PageShell';
import { useTranslation } from 'react-i18next';

import { UsersManagementPanel } from './UserManagement';

export const UsersPage = ({
  onManageIntermediates,
}: {
  onManageIntermediates?: (userId: string, label: string) => void;
}) => {
  const { t } = useTranslation();

  return (
    <PageShell description={t('users.description')} regionLabel={t('users.management')} title={t('users.title')}>
      <UsersManagementPanel onManageIntermediates={onManageIntermediates} />
    </PageShell>
  );
};
