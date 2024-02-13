import { Icon, Tooltip } from '@invoke-ai/ui-library';
import { useNodeClassification } from 'features/nodes/hooks/useNodeClassification';
import type { Classification } from 'features/nodes/types/common';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlaskBold, PiHammerBold } from 'react-icons/pi';

interface Props {
  nodeId: string;
}

const InvocationNodeClassificationIcon = ({ nodeId }: Props) => {
  const classification = useNodeClassification(nodeId);

  if (!classification || classification === 'stable') {
    return null;
  }

  return (
    <Tooltip
      label={<ClassificationTooltipContent classification={classification} />}
      placement="top"
      shouldWrapChildren
    >
      <Icon as={getIcon(classification)} display="block" boxSize={4} color="base.400" />
    </Tooltip>
  );
};

export default memo(InvocationNodeClassificationIcon);

const ClassificationTooltipContent = memo(({ classification }: { classification: Classification }) => {
  const { t } = useTranslation();

  if (classification === 'beta') {
    return t('nodes.betaDesc');
  }

  if (classification === 'prototype') {
    return t('nodes.prototypeDesc');
  }

  return null;
});

ClassificationTooltipContent.displayName = 'ClassificationTooltipContent';

const getIcon = (classification: Classification) => {
  if (classification === 'beta') {
    return PiHammerBold;
  }

  if (classification === 'prototype') {
    return PiFlaskBold;
  }

  return undefined;
};
