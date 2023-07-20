import { Box, Flex } from '@chakra-ui/react';
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
import { v4 as uuidv4 } from 'uuid';
import ControlNetImagePreview from './ControlNetImagePreview';
import ControlNetProcessorComponent from './ControlNetProcessorComponent';
import ParamControlNetShouldAutoConfig from './ParamControlNetShouldAutoConfig';
import ParamControlNetBeginEnd from './parameters/ParamControlNetBeginEnd';
import ParamControlNetControlMode from './parameters/ParamControlNetControlMode';
import ParamControlNetProcessorSelect from './parameters/ParamControlNetProcessorSelect';
import ParamControlNetResizeMode from './parameters/ParamControlNetResizeMode';

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
        gap: 3,
        p: 3,
        borderRadius: 'base',
        position: 'relative',
        bg: 'base.200',
        _dark: {
          bg: 'base.850',
        },
      }}
    >
      <Flex sx={{ gap: 2, alignItems: 'center' }}>
        <IAISwitch
          tooltip={'Toggle this ControlNet'}
          aria-label={'Toggle this ControlNet'}
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
          tooltip={isExpanded ? 'Hide Advanced' : 'Show Advanced'}
          aria-label={isExpanded ? 'Hide Advanced' : 'Show Advanced'}
          onClick={toggleIsExpanded}
          variant="link"
          icon={
            <ChevronUpIcon
              sx={{
                boxSize: 6,
                color: 'base.700',
                transform: isExpanded ? 'rotate(0deg)' : 'rotate(180deg)',
                transitionProperty: 'common',
                transitionDuration: 'normal',
                _dark: {
                  color: 'base.300',
                },
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
              top: 4,
              insetInlineEnd: 4,
              bg: 'accent.700',
              _dark: {
                bg: 'accent.400',
              },
            }}
          />
        )}
      </Flex>
      <Flex sx={{ w: 'full', flexDirection: 'column', gap: 3 }}>
        <Flex sx={{ gap: 4, w: 'full', alignItems: 'center' }}>
          <Flex
            sx={{
              flexDir: 'column',
              gap: 3,
              h: 28,
              w: 'full',
              paddingInlineStart: 1,
              paddingInlineEnd: isExpanded ? 1 : 0,
              pb: 2,
              justifyContent: 'space-between',
            }}
          >
            <ParamControlNetWeight controlNetId={controlNetId} />
            <ParamControlNetBeginEnd controlNetId={controlNetId} />
          </Flex>
          {!isExpanded && (
            <Flex
              sx={{
                alignItems: 'center',
                justifyContent: 'center',
                h: 28,
                w: 28,
                aspectRatio: '1/1',
              }}
            >
              <ControlNetImagePreview controlNetId={controlNetId} height={28} />
            </Flex>
          )}
        </Flex>
        <Flex sx={{ gap: 2 }}>
          <ParamControlNetControlMode controlNetId={controlNetId} />
          <ParamControlNetResizeMode controlNetId={controlNetId} />
        </Flex>
        <ParamControlNetProcessorSelect controlNetId={controlNetId} />
      </Flex>

      {isExpanded && (
        <>
          <ControlNetImagePreview controlNetId={controlNetId} height="392px" />
          <ParamControlNetShouldAutoConfig controlNetId={controlNetId} />
          <ControlNetProcessorComponent controlNetId={controlNetId} />
        </>
      )}
    </Flex>
  );
};

export default memo(ControlNet);
