import { IconButton } from '@workbench/components/ui';
import { MoreHorizontalIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const LayersHeaderActions = () => {
  const { t } = useTranslation();

  return (
    <IconButton aria-label={t('widgets.layers.options')} color="fg.muted" size="2xs" variant="ghost">
      <MoreHorizontalIcon />
    </IconButton>
  );
};
