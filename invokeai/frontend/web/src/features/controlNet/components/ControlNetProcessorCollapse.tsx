import { useDisclosure } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo, useState } from 'react';
import CannyProcessor from './processors/CannyProcessor';
import { ImageDTO } from 'services/api';
import IAICustomSelect from 'common/components/IAICustomSelect';
import {
  CONTROLNET_PROCESSORS,
  ControlNetProcessor,
} from '../store/controlNetSlice';

export type ControlNetProcessorProps = {
  controlNetId: string;
  image: ImageDTO;
  type: ControlNetProcessor;
};

const ProcessorComponent = (props: ControlNetProcessorProps) => {
  const { type } = props;
  if (type === 'canny') {
    return <CannyProcessor {...props} />;
  }
  return null;
};

type ControlNetProcessorCollapseProps = {
  controlNetId: string;
  image: ImageDTO | null;
};

const ControlNetProcessorCollapse = (
  props: ControlNetProcessorCollapseProps
) => {
  const { image, controlNetId } = props;
  const { isOpen, onToggle } = useDisclosure();

  const [processorType, setProcessorType] =
    useState<ControlNetProcessor>('canny');

  const handleProcessorTypeChanged = (type: string | null | undefined) => {
    setProcessorType(type as ControlNetProcessor);
  };

  return (
    <IAICollapse
      isOpen={Boolean(isOpen && image)}
      onToggle={onToggle}
      label="Process Image"
      withSwitch
    >
      <IAICustomSelect
        items={CONTROLNET_PROCESSORS}
        selectedItem={processorType}
        setSelectedItem={handleProcessorTypeChanged}
      />
      {image && (
        <ProcessorComponent
          controlNetId={controlNetId}
          image={image}
          type={processorType}
        />
      )}
    </IAICollapse>
  );
};

export default memo(ControlNetProcessorCollapse);
