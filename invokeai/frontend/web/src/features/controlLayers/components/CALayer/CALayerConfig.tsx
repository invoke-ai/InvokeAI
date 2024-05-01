import { Box, Flex, Icon, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CALayerModelCombobox } from 'features/controlLayers/components/CALayer/CALayerModelCombobox';
import { selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretUpBold } from 'react-icons/pi';
import { useToggle } from 'react-use';

import { CALayerBeginEndStepPct } from './CALayerBeginEndStepPct';
import { CALayerControlMode } from './CALayerControlMode';
import { CALayerImagePreview } from './CALayerImagePreview';
import { CALayerProcessor } from './CALayerProcessor';
import { CALayerProcessorCombobox } from './CALayerProcessorCombobox';
import { CALayerWeight } from './CALayerWeight';

type Props = {
  layerId: string;
};

export const CALayerConfig = memo(({ layerId }: Props) => {
  const caType = useAppSelector((s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter.type);
  const { t } = useTranslation();
  const [isExpanded, toggleIsExpanded] = useToggle(false);

  return (
    <Flex flexDir="column" gap={4} position="relative" w="full">
      <Flex gap={3} alignItems="center" w="full">
        <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
          <CALayerModelCombobox layerId={layerId} />
        </Box>

        <IconButton
          size="sm"
          tooltip={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          aria-label={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          onClick={toggleIsExpanded}
          variant="ghost"
          icon={
            <Icon
              boxSize={4}
              as={PiCaretUpBold}
              transform={isExpanded ? 'rotate(0deg)' : 'rotate(180deg)'}
              transitionProperty="common"
              transitionDuration="normal"
            />
          }
        />
      </Flex>
      <Flex gap={4} w="full" alignItems="center">
        <Flex flexDir="column" gap={3} w="full">
          {caType === 'controlnet' && <CALayerControlMode layerId={layerId} />}
          <CALayerWeight layerId={layerId} />
          <CALayerBeginEndStepPct layerId={layerId} />
        </Flex>
        <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
          <CALayerImagePreview layerId={layerId} />
        </Flex>
      </Flex>
      {isExpanded && (
        <>
          <CALayerProcessorCombobox layerId={layerId} />
          <CALayerProcessor layerId={layerId} />
        </>
      )}
    </Flex>
  );
});

CALayerConfig.displayName = 'CALayerConfig';
