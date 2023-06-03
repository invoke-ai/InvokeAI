import { memo, useCallback } from 'react';
import {
  ControlNetConfig,
  controlNetAdded,
  controlNetRemoved,
  controlNetToggled,
} from '../store/controlNetSlice';
import { useAppDispatch } from 'app/store/storeHooks';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';
import {
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  HStack,
  TabList,
  TabPanels,
  Tabs,
  Tab,
  TabPanel,
  Box,
} from '@chakra-ui/react';
import { FaCopy, FaPlus, FaTrash, FaWrench } from 'react-icons/fa';

import ParamControlNetBeginEnd from './parameters/ParamControlNetBeginEnd';
import ControlNetImagePreview from './ControlNetImagePreview';
import IAIIconButton from 'common/components/IAIIconButton';
import { v4 as uuidv4 } from 'uuid';
import { useToggle } from 'react-use';
import ParamControlNetProcessorSelect from './parameters/ParamControlNetProcessorSelect';
import ControlNetProcessorComponent from './ControlNetProcessorComponent';
import ControlNetPreprocessButton from './ControlNetPreprocessButton';
import IAIButton from 'common/components/IAIButton';
import IAISwitch from 'common/components/IAISwitch';
import { ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons';

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
    controlImage,
    processedControlImage,
    processorNode,
    processorType,
  } = props.controlNet;
  const dispatch = useAppDispatch();
  const [shouldShowAdvanced, onToggleAdvanced] = useToggle(false);

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
        bg: 'base.850',
        borderRadius: 'base',
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
          aria-label="Expand"
          onClick={onToggleAdvanced}
          variant="link"
          icon={
            <ChevronUpIcon
              sx={{
                boxSize: 4,
                color: 'base.300',
                transform: shouldShowAdvanced
                  ? 'rotate(0deg)'
                  : 'rotate(180deg)',
                transitionProperty: 'common',
                transitionDuration: 'normal',
              }}
            />
          }
        />
      </Flex>
      {isEnabled && (
        <>
          <Flex sx={{ gap: 4 }}>
            <Flex
              sx={{
                flexDir: 'column',
                gap: 2,
                w: 'full',
                h: 24,
                paddingInlineStart: 1,
                paddingInlineEnd: shouldShowAdvanced ? 1 : 0,
                pb: 2,
                justifyContent: 'space-between',
              }}
            >
              <ParamControlNetWeight
                controlNetId={controlNetId}
                weight={weight}
                mini
              />
              <ParamControlNetBeginEnd
                controlNetId={controlNetId}
                beginStepPct={beginStepPct}
                endStepPct={endStepPct}
                mini
              />
            </Flex>
            {!shouldShowAdvanced && (
              <Flex
                sx={{
                  alignItems: 'center',
                  justifyContent: 'center',
                  h: 24,
                  w: 24,
                  aspectRatio: '1/1',
                }}
              >
                <ControlNetImagePreview controlNet={props.controlNet} />
              </Flex>
            )}
          </Flex>
          {shouldShowAdvanced && (
            <>
              <Box pt={2}>
                <ControlNetImagePreview controlNet={props.controlNet} />
              </Box>
              <ParamControlNetProcessorSelect
                controlNetId={controlNetId}
                processorNode={processorNode}
              />
              <ControlNetProcessorComponent
                controlNetId={controlNetId}
                processorNode={processorNode}
              />
            </>
          )}
        </>
      )}
    </Flex>
  );

  return (
    <Flex sx={{ flexDir: 'column', gap: 3 }}>
      <ControlNetImagePreview controlNet={props.controlNet} />
      <ParamControlNetModel controlNetId={controlNetId} model={model} />
      <Tabs
        isFitted
        orientation="horizontal"
        variant="enclosed"
        size="sm"
        colorScheme="accent"
      >
        <TabList>
          <Tab
            sx={{ 'button&': { _selected: { borderBottomColor: 'base.800' } } }}
          >
            Model Config
          </Tab>
          <Tab
            sx={{ 'button&': { _selected: { borderBottomColor: 'base.800' } } }}
          >
            Preprocess
          </Tab>
        </TabList>
        <TabPanels sx={{ pt: 2 }}>
          <TabPanel sx={{ p: 0 }}>
            <ParamControlNetWeight
              controlNetId={controlNetId}
              weight={weight}
            />
            <ParamControlNetBeginEnd
              controlNetId={controlNetId}
              beginStepPct={beginStepPct}
              endStepPct={endStepPct}
            />
          </TabPanel>
          <TabPanel sx={{ p: 0 }}>
            <ParamControlNetProcessorSelect
              controlNetId={controlNetId}
              processorNode={processorNode}
            />
            <ControlNetProcessorComponent
              controlNetId={controlNetId}
              processorNode={processorNode}
            />
            <ControlNetPreprocessButton controlNet={props.controlNet} />
            {/* <IAIButton
              size="sm"
              leftIcon={<FaUndo />}
              onClick={handleReset}
              isDisabled={Boolean(!processedControlImage)}
            >
              Reset Processing
            </IAIButton> */}
          </TabPanel>
        </TabPanels>
      </Tabs>
      <IAIButton onClick={handleDelete}>Remove ControlNet</IAIButton>
    </Flex>
  );
};

export default memo(ControlNet);
