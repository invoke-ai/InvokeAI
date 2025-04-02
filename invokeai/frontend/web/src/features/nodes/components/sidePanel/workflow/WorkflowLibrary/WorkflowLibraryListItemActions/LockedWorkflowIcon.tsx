import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockBold } from 'react-icons/pi';

export const LockedWorkflowIcon = memo(() => {
  const { t } = useTranslation();

  return (
    <Tooltip label={t('workflows.builder.publishedWorkflowsLocked')} closeOnScroll>
      <IconButton
        size="sm"
        cursor="not-allowed"
        variant="link"
        alignSelf="stretch"
        aria-label={t('workflows.builder.publishedWorkflowsLocked')}
        icon={<PiLockBold />}
      />
    </Tooltip>
  );
});

LockedWorkflowIcon.displayName = 'LockedWorkflowIcon';
