import { Box, Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/storeHooks';
import { memo } from 'react';
import LinearContent from './LinearContent';
import LinearParameters from './LinearParameters';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import { RootState } from 'app/store';
import Scrollable from '../../common/Scrollable';
import ParametersSlide from '../../common/ParametersSlide';
import AnimatedImageToImagePanel from 'features/parameters/components/AnimatedImageToImagePanel';

const LinearWorkspace = () => {
  const shouldPinParametersPanel = useAppSelector(
    (state: RootState) => state.ui.shouldPinParametersPanel
  );

  return (
    <Flex
      flexDirection={{ base: 'column-reverse', xl: 'row' }}
      w="full"
      h="full"
      gap={4}
    >
      {shouldPinParametersPanel ? (
        <Flex sx={{ flexDirection: 'row-reverse' }}>
          <AnimatedImageToImagePanel />
          <Flex
            sx={{
              flexDirection: 'column',
              width: '28rem',
              flexShrink: 0,
              position: 'relative',
            }}
          >
            <Scrollable>
              <LinearParameters />
            </Scrollable>
            <PinParametersPanelButton
              sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
            />
          </Flex>
        </Flex>
      ) : (
        <ParametersSlide>
          <LinearParameters />
        </ParametersSlide>
      )}
      <LinearContent />
    </Flex>
  );
};

export default memo(LinearWorkspace);
