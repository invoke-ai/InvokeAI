import { Box, ChakraProps, Flex, useColorMode } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { FaCopy, FaTrash } from 'react-icons/fa';
import {
  ControlNetConfig,
  controlNetAdded,
  controlNetRemoved,
  controlNetToggled,
} from '../store/controlNetSlice';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';

import { ChevronUpIcon } from '@chakra-ui/icons';
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
import { mode } from 'theme/util/mode';

const expandedControlImageSx: ChakraProps['sx'] = { maxH: 96 };

type ControlNetProps = {
  controlNet: ControlNetConfig;
};

const ControlNet = (props: ControlNetProps) => {
  const {
    controlNetId,
    isEnabled,
    model,
    weight,
    beginStepPct,
    endStepPct,
    controlMode,
    controlImage,
    processedControlImage,
    processorNode,
    processorType,
    shouldAutoConfig,
  } = props.controlNet;
  const dispatch = useAppDispatch();
  const [isExpanded, toggleIsExpanded] = useToggle(false);
  const { colorMode } = useColorMode();
  const handleDelete = useCallback(() => {
    dispatch(controlNetRemoved({ controlNetId }));
  }, [controlNetId, dispatch]);

  const handleDuplicate = useCallback(() => {
    dispatch(
      controlNetAdded({ controlNetId: uuidv4(), controlNet: props.controlNet })
    );
  }, [dispatch, props.controlNet]);

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
          <ParamControlNetModel controlNetId={controlNetId} model={model} />
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
                  weight={weight}
                  mini={!isExpanded}
                />
                <ParamControlNetBeginEnd
                  controlNetId={controlNetId}
                  beginStepPct={beginStepPct}
                  endStepPct={endStepPct}
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
                    controlNet={props.controlNet}
                    height={24}
                  />
                </Flex>
              )}
            </Flex>
            <ParamControlNetControlMode
              controlNetId={controlNetId}
              controlMode={controlMode}
            />
          </Flex>

          {isExpanded && (
            <>
              <Box mt={2}>
                <ControlNetImagePreview
                  controlNet={props.controlNet}
                  height={96}
                />
              </Box>
              <ParamControlNetProcessorSelect
                controlNetId={controlNetId}
                processorNode={processorNode}
              />
              <ControlNetProcessorComponent
                controlNetId={controlNetId}
                processorNode={processorNode}
              />
              <ParamControlNetShouldAutoConfig
                controlNetId={controlNetId}
                shouldAutoConfig={shouldAutoConfig}
              />
            </>
          )}
        </>
      )}
    </Flex>
  );
};

export default memo(ControlNet);
