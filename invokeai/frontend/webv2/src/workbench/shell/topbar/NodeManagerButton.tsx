import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { Link } from '@tanstack/react-router';
import { BlocksIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const NodesManagerButton = () => {
  const { t } = useTranslation();

  return (
    <Tooltip content={t('nodes.manager')} showArrow>
      <IconButton aria-label={t('nodes.manager')} asChild size="sm" variant="ghost">
        <Link to="/nodes">
          <Icon as={BlocksIcon} />
        </Link>
      </IconButton>
    </Tooltip>
  );
};
