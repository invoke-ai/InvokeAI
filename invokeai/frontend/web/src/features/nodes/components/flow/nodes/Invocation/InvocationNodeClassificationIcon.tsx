import { Icon, Tooltip } from '@invoke-ai/ui-library';
import { useNodeClassification } from 'features/nodes/hooks/useNodeClassification';
import type { Classification } from 'features/nodes/types/common';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCircuitryBold, PiFlaskBold, PiHammerBold, PiLightningFill } from 'react-icons/pi';

interface Props {
  nodeId: string;
}

const InvocationNodeClassificationIcon = (_: Props) => {
  const classification = useNodeClassification();

  if (!classification || classification === 'stable') {
    return null;
  }

  return (
    <Tooltip
      label={<ClassificationTooltipContent classification={classification} />}
      placement="top"
      shouldWrapChildren
    >
      <ClassificationIcon classification={classification} />
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

  if (classification === 'internal') {
    return t('nodes.internalDesc');
  }

  if (classification === 'special') {
    return t('nodes.specialDesc');
  }

  return null;
});

ClassificationTooltipContent.displayName = 'ClassificationTooltipContent';

const ClassificationIcon = ({ classification }: { classification: Classification }) => {
  if (classification === 'beta') {
    return <Icon as={PiHammerBold} display="block" boxSize={4} color="invokeYellow.300" />;
  }

  if (classification === 'prototype') {
    return <Icon as={PiFlaskBold} display="block" boxSize={4} color="invokeRed.300" />;
  }

  if (classification === 'internal') {
    return <Icon as={PiCircuitryBold} display="block" boxSize={4} color="invokePurple.300" />;
  }

  if (classification === 'special') {
    return <Icon as={PiLightningFill} display="block" boxSize={4} color="invokeGreen.300" />;
  }

  return null;
};
