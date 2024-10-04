import { Badge } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const AutoAddBadge = memo(() => {
  const { t } = useTranslation();
  return (
    <Badge color="invokeBlue.400" borderColor="invokeBlue.700" borderWidth={1} bg="transparent" flexShrink={0}>
      {t('common.auto')}
    </Badge>
  );
});

AutoAddBadge.displayName = 'AutoAddBadge';
