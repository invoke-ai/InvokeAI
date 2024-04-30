import { Box, Flex, Icon, IconButton } from '@invoke-ai/ui-library';
import ControlAdapterProcessorComponent from 'features/controlAdapters/components/ControlAdapterProcessorComponent';
import ControlAdapterShouldAutoConfig from 'features/controlAdapters/components/ControlAdapterShouldAutoConfig';
import ParamControlAdapterIPMethod from 'features/controlAdapters/components/parameters/ParamControlAdapterIPMethod';
import ParamControlAdapterProcessorSelect from 'features/controlAdapters/components/parameters/ParamControlAdapterProcessorSelect';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretUpBold } from 'react-icons/pi';
import { useToggle } from 'react-use';

import ControlAdapterImagePreview from './ControlAdapterImagePreview';
import { ParamControlAdapterBeginEnd } from './ParamControlAdapterBeginEnd';
import ParamControlAdapterControlMode from './ParamControlAdapterControlMode';
import ParamControlAdapterModel from './ParamControlAdapterModel';
import ParamControlAdapterWeight from './ParamControlAdapterWeight';

const ControlAdapterLayerConfig = (props: { id: string }) => {
  const { id } = props;
  const controlAdapterType = useControlAdapterType(id);
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
          <ControlAdapterImagePreview id={id} isSmall />
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
};

export default memo(ControlAdapterLayerConfig);
