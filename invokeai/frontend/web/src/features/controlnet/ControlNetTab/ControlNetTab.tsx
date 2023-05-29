import { Flex, Text } from '@chakra-ui/react';
import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { ControlNetConfig } from '../store/controlnetTypes';
import ControlNetEnabled from './ControlNetEnabled';
import ControlNetEnd from './ControlNetEnd';
import ControlNetImage from './ControlNetImage';
import ControlNetModels from './ControlNetModels';
import ControlNetProcessor from './ControlNetProcessor';
import ControlNetStart from './ControlNetStart';
import ControlNetWeight from './ControlNetWeight';

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
          controlnet.controlnetEnabled ? 'accent.800' : 'base.800'
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
      <ControlNetImage />
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
      <ControlNetWeight controlnet={controlnet} setControlnet={setControlnet} />
      <ControlNetStart controlnet={controlnet} setControlnet={setControlnet} />
      <ControlNetEnd controlnet={controlnet} setControlnet={setControlnet} />
    </Flex>
  );
}
