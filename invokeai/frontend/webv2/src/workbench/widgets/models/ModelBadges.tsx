import type { ModelConfig } from '@workbench/models/types';

import { Badge, HStack } from '@chakra-ui/react';
import { getModelBaseColorPalette, getModelBaseLabel, getModelFormatLabel } from '@workbench/models/taxonomy';

export const ModelBaseBadge = ({ base }: { base: ModelConfig['base'] }) => (
  <Badge colorPalette={getModelBaseColorPalette(base)} flexShrink={0} fontSize="2xs" size="sm" variant="surface">
    {getModelBaseLabel(base)}
  </Badge>
);

export const ModelFormatBadge = ({ format }: { format: ModelConfig['format'] }) => (
  <Badge colorPalette="gray" flexShrink={0} fontSize="2xs" size="sm" variant="outline">
    {getModelFormatLabel(format)}
  </Badge>
);

export const MissingFileBadge = () => (
  <Badge colorPalette="red" flexShrink={0} fontSize="2xs" size="sm" variant="surface">
    Missing file
  </Badge>
);

export const ModelBadgeRow = ({ isMissing, model }: { isMissing?: boolean; model: ModelConfig }) => (
  <HStack gap="1" minW="0" wrap="wrap">
    <ModelBaseBadge base={model.base} />
    <ModelFormatBadge format={model.format} />
    {isMissing ? <MissingFileBadge /> : null}
  </HStack>
);
