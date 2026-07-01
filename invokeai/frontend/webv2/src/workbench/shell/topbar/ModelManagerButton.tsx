import { Link } from '@tanstack/react-router';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { BoxIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const ModelManagerButton = () => {
  const { t } = useTranslation();

  return (
    <Tooltip content={t('models.manager')} showArrow>
      <IconButton aria-label={t('models.manager')} asChild size="sm" variant="ghost">
        <Link to="/models">
          <BoxIcon />
        </Link>
      </IconButton>
    </Tooltip>
  );
};
