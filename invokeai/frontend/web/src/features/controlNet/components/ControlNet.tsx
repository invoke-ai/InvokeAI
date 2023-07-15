import { Box, Flex, useColorMode } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { FaCopy, FaTrash } from 'react-icons/fa';
import {
  controlNetDuplicated,
  controlNetRemoved,
  controlNetToggled,
} from '../store/controlNetSlice';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';

import { ChevronUpIcon } from '@chakra-ui/icons';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISwitch from 'common/components/IAISwitch';
import { useToggle } from 'react-use';
import { mode } from 'theme/util/mode';
import { v4 as uuidv4 } from 'uuid';
import ControlNetImagePreview from './ControlNetImagePreview';
import ControlNetProcessorComponent from './ControlNetProcessorComponent';
import ParamControlNetShouldAutoConfig from './ParamControlNetShouldAutoConfig';
import ParamControlNetBeginEnd from './parameters/ParamControlNetBeginEnd';
import ParamControlNetControlMode from './parameters/ParamControlNetControlMode';
import ParamControlNetProcessorSelect from './parameters/ParamControlNetProcessorSelect';

type ControlNetProps = {
  controlNetId: string;
};

const ControlNet = (props: ControlNetProps) => {
  const { controlNetId } = props;
  const dispatch = useAppDispatch();

  const selector = createSelector(
    stateSelector,
    ({ controlNet }) => {
      const { isEnabled, shouldAutoConfig } =
        controlNet.controlNets[controlNetId];

      return { isEnabled, shouldAutoConfig };
    },
    defaultSelectorOptions
  );

  const { isEnabled, shouldAutoConfig } = useAppSelector(selector);

  const [isExpanded, toggleIsExpanded] = useToggle(false);
  const { colorMode } = useColorMode();
  const handleDelete = useCallback(() => {
    dispatch(controlNetRemoved({ controlNetId }));
  }, [controlNetId, dispatch]);

  const handleDuplicate = useCallback(() => {
    dispatch(
      controlNetDuplicated({
        sourceControlNetId: controlNetId,
        newControlNetId: uuidv4(),
      })
    );
  }, [controlNetId, dispatch]);

  const handleToggleIsEnabled = useCallback(() => {
    dispatch(controlNetToggled({ controlNetId }));
  }, [controlNetId, dispatch]);

  return (
    <Flex
      sx={{
        flexDir: 'column',
        gap: 2,
        p: 3,
        bg: mode('base.200', 'base.850')(colorMode),
        borderRadius: 'base',
        position: 'relative',
      }}
    >
      <Flex sx={{ gap: 2 }}>
        <IAISwitch
          tooltip="Toggle"
          aria-label="Toggle"
          isChecked={isEnabled}
          onChange={handleToggleIsEnabled}
        />
        <Box
          sx={{
            w: 'full',
            minW: 0,
            opacity: isEnabled ? 1 : 0.5,
            pointerEvents: isEnabled ? 'auto' : 'none',
            transitionProperty: 'common',
            transitionDuration: '0.1s',
          }}
        >
          <ParamControlNetModel controlNetId={controlNetId} />
        </Box>
        <IAIIconButton
          size="sm"
          tooltip="Duplicate"
          aria-label="Duplicate"
          onClick={handleDuplicate}
          icon={<FaCopy />}
        />
        <IAIIconButton
          size="sm"
          tooltip="Delete"
          aria-label="Delete"
          colorScheme="error"
          onClick={handleDelete}
          icon={<FaTrash />}
        />
        <IAIIconButton
          size="sm"
          aria-label="Show All Options"
          onClick={toggleIsExpanded}
          variant="link"
          icon={
            <ChevronUpIcon
              sx={{
                boxSize: 4,
                color: mode('base.700', 'base.300')(colorMode),
                transform: isExpanded ? 'rotate(0deg)' : 'rotate(180deg)',
                transitionProperty: 'common',
                transitionDuration: 'normal',
              }}
            />
          }
        />

        {!shouldAutoConfig && (
          <Box
            sx={{
              position: 'absolute',
              w: 1.5,
              h: 1.5,
              borderRadius: 'full',
              bg: mode('error.700', 'error.200')(colorMode),
              top: 4,
              insetInlineEnd: 4,
            }}
          />
        )}
      </Flex>
      <Flex alignItems="flex-end" gap="2">
        <ParamControlNetProcessorSelect controlNetId={controlNetId} />
        <ParamControlNetShouldAutoConfig controlNetId={controlNetId} />
      </Flex>
      {isEnabled && (
        <>
          <Flex sx={{ w: 'full', flexDirection: 'column' }}>
            <Flex sx={{ gap: 4, w: 'full' }}>
              <Flex
                sx={{
                  flexDir: 'column',
                  gap: 3,
                  w: 'full',
                  paddingInlineStart: 1,
                  paddingInlineEnd: isExpanded ? 1 : 0,
                  pb: 2,
                  justifyContent: 'space-between',
                }}
              >
                <ParamControlNetWeight
                  controlNetId={controlNetId}
                  mini={!isExpanded}
                />
                <ParamControlNetBeginEnd
                  controlNetId={controlNetId}
                  mini={!isExpanded}
                />
              </Flex>
              {!isExpanded && (
                <Flex
                  sx={{
                    alignItems: 'center',
                    justifyContent: 'center',
                    h: 24,
                    w: 24,
                    aspectRatio: '1/1',
                  }}
                >
                  <ControlNetImagePreview
                    controlNetId={controlNetId}
                    height={24}
                  />
                </Flex>
              )}
            </Flex>
            <ParamControlNetControlMode controlNetId={controlNetId} />
          </Flex>

          {isExpanded && (
            <>
              <Box mt={2}>
                <ControlNetImagePreview
                  controlNetId={controlNetId}
                  height={96}
                />
              </Box>
              <ControlNetProcessorComponent controlNetId={controlNetId} />
            </>
          )}
        </>
      )}
    </Flex>
  );
};

export default memo(ControlNet);
