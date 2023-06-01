import { memo } from 'react';
import { ControlNetProcessorNode } from '../store/types';
import { ImageDTO } from 'services/api';
import CannyProcessor from './processors/CannyProcessor';

export type ControlNetProcessorProps = {
  controlNetId: string;
  image: ImageDTO;
  type: ControlNetProcessorNode['type'];
};

const renderProcessorComponent = (props: ControlNetProcessorProps) => {
  const { type } = props;
  if (type === 'canny_image_processor') {
    return <CannyProcessor {...props} />;
  }
};

const ControlNet = () => {
  return (
    <div>
      <h1>ControlNet</h1>
    </div>
  );
};

export default memo(ControlNet);
