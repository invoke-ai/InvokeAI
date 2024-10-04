import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowBendUpLeftBold } from 'react-icons/pi';

type MetadataItemProps = Omit<IconButtonProps, 'aria-label'> & {
  label: string;
};

export const RecallButton = memo(({ label, ...rest }: MetadataItemProps) => {
  const { t } = useTranslation();

  return (
    <Tooltip label={t('metadata.recallParameter', { label })}>
      <IconButton
        aria-label={t('metadata.recallParameter', { label })}
        icon={<PiArrowBendUpLeftBold />}
        size="xs"
        variant="ghost"
        {...rest}
      />
    </Tooltip>
  );
});

RecallButton.displayName = 'RecallButton';
