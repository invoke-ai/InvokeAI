import { Icon } from '@chakra-ui/react';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { useNodeClassification } from 'features/nodes/hooks/useNodeClassification';
import type { Classification } from 'features/nodes/types/common';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlaskBold, PiHammerBold } from 'react-icons/pi'

interface Props {
  nodeId: string;
}

const InvocationNodeClassificationIcon = ({ nodeId }: Props) => {
  const classification = useNodeClassification(nodeId);

  if (!classification || classification === 'stable') {
    return null;
  }

  return (
    <InvTooltip
      label={<ClassificationTooltipContent classification={classification} />}
      placement="top"
      shouldWrapChildren
    >
      <Icon
        as={getIcon(classification)}
        display="block"
        boxSize={4}
        color="base.400"
      />
    </InvTooltip>
  );
};

export default memo(InvocationNodeClassificationIcon);

const ClassificationTooltipContent = memo(
  ({ classification }: { classification: Classification }) => {
    const { t } = useTranslation();

    if (classification === 'beta') {
      return t('nodes.betaDesc');
    }

    if (classification === 'prototype') {
      return t('nodes.prototypeDesc');
    }

    return null;
  }
);

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
