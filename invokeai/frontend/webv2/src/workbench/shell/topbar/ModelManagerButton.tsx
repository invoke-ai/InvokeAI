import { Link } from '@tanstack/react-router';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { BoxIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ModelManagerButton = () => {
  const { t } = useTranslation();
  const projectId = useActiveProjectId();
  const search = useMemo(() => ({ project: projectId }), [projectId]);

  return (
    <Tooltip content={t('models.manager')} showArrow>
      <IconButton aria-label={t('models.manager')} asChild size="sm" variant="ghost">
        <Link search={search} to="/models">
          <BoxIcon />
        </Link>
      </IconButton>
    </Tooltip>
  );
};
