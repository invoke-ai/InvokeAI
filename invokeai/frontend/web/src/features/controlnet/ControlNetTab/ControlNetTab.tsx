import { Flex, Text } from '@chakra-ui/react';
import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { ControlNetConfig } from '../store/controlnetTypes';
import ControlNetEnabled from './ControlNetEnabled';
import ControlNetModels from './ControlNetModels';
import ControlNetProcessor from './ControlNetProcessor';

interface ControlNetTabProps {
  label: string;
  controlnet: ControlNetConfig;
  setControlnet: ActionCreatorWithPayload<ControlNetConfig>;
}

export default function ControlNetTab(props: ControlNetTabProps) {
  const { label, controlnet, setControlnet } = props;

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'max-content',
        flexDirection: 'column',
        rowGap: 2,
        background: `${
          controlnet.controlnetEnabled ? 'accent.600' : 'base.850'
        }`,
        padding: 4,
        borderRadius: 4,
      }}
    >
      <Flex gap={2}>
        <ControlNetEnabled
          controlnet={controlnet}
          setControlnet={setControlnet}
        />
        <Text>{label}</Text>
      </Flex>
      <Flex gap={2}>
        <ControlNetProcessor
          controlnet={controlnet}
          setControlnet={setControlnet}
        />
        <ControlNetModels
          controlnet={controlnet}
          setControlnet={setControlnet}
        />
      </Flex>
    </Flex>
  );
}
