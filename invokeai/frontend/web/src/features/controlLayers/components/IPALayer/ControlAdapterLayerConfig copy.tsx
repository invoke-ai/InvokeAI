import { Box, Flex, Icon, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import ControlAdapterProcessorComponent from 'features/controlAdapters/components/ControlAdapterProcessorComponent';
import ControlAdapterShouldAutoConfig from 'features/controlAdapters/components/ControlAdapterShouldAutoConfig';
import ParamControlAdapterIPMethod from 'features/controlAdapters/components/parameters/ParamControlAdapterIPMethod';
import ParamControlAdapterProcessorSelect from 'features/controlAdapters/components/parameters/ParamControlAdapterProcessorSelect';
import { ParamControlAdapterBeginEnd } from 'features/controlLayers/components/CALayer/CALayerBeginEndStepPct';
import ParamControlAdapterControlMode from 'features/controlLayers/components/CALayer/CALayerControlMode';
import { CALayerImagePreview } from 'features/controlLayers/components/CALayer/CALayerImagePreview';
import ParamControlAdapterModel from 'features/controlLayers/components/CALayer/CALayerModelCombobox';
import ParamControlAdapterWeight from 'features/controlLayers/components/CALayer/CALayerWeight';
import { selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretUpBold } from 'react-icons/pi';
import { useToggle } from 'react-use';

type Props = {
  layerId: string;
};

export const CALayerCAConfig = memo(({ layerId }: Props) => {
  const caType = useAppSelector((s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter.type);
  const { t } = useTranslation();
  const [isExpanded, toggleIsExpanded] = useToggle(false);

  return (
    <Flex flexDir="column" gap={4} position="relative" w="full">
      <Flex gap={3} alignItems="center" w="full">
        <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
          <ParamControlAdapterModel id={id} />{' '}
        </Box>

        {controlAdapterType !== 'ip_adapter' && (
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
        )}
      </Flex>
      <Flex gap={4} w="full" alignItems="center">
        <Flex flexDir="column" gap={3} w="full">
          {controlAdapterType === 'ip_adapter' && <ParamControlAdapterIPMethod id={id} />}
          {controlAdapterType === 'controlnet' && <ParamControlAdapterControlMode id={id} />}
          <ParamControlAdapterWeight id={id} />
          <ParamControlAdapterBeginEnd id={id} />
        </Flex>
        <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
          <CALayerImagePreview id={id} isSmall />
        </Flex>
      </Flex>
      {isExpanded && (
        <>
          <ControlAdapterShouldAutoConfig id={id} />
          <ParamControlAdapterProcessorSelect id={id} />
          <ControlAdapterProcessorComponent id={id} />
        </>
      )}
    </Flex>
  );
});

CALayerCAConfig.displayName = 'CALayerCAConfig';
