import type { ModelConfig } from '@features/models/core/types';

import { Badge } from '@chakra-ui/react';
import { getModelBaseColorPalette, getModelBaseLabel } from '@features/models/core/baseIdentity';
import { getModelFormatLabel } from '@features/models/core/taxonomy';
import { useTranslation } from 'react-i18next';

export const ModelBaseBadge = ({ base }: { base: ModelConfig['base'] }) => (
  <Badge colorPalette={getModelBaseColorPalette(base)} flexShrink={0} fontSize="xs" size="lg" variant="surface">
    {getModelBaseLabel(base)}
  </Badge>
);

export const ModelFormatBadge = ({ format }: { format: ModelConfig['format'] }) => (
  <Badge colorPalette="gray" flexShrink={0} fontSize="xs" size="lg" variant="surface">
    {getModelFormatLabel(format)}
  </Badge>
);

export const MissingFileBadge = () => {
  const { t } = useTranslation();

  return (
    <Badge colorPalette="red" flexShrink={0} fontSize="xs" size="lg" variant="surface">
      {t('models.missingFile')}
    </Badge>
  );
};
