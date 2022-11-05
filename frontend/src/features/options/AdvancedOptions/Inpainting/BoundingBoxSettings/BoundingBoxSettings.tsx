import { Flex } from '@chakra-ui/react';
import BoundingBoxDarkenOutside from './BoundingBoxDarkenOutside';
import BoundingBoxDimensionSlider from './BoundingBoxDimensionSlider';
import BoundingBoxLock from './BoundingBoxLock';
import BoundingBoxVisibility from './BoundingBoxVisibility';

const BoundingBoxSettings = () => {
  return (
    <div className="inpainting-bounding-box-settings">
      <div className="inpainting-bounding-box-header">
        <p>Inpaint Box</p>
        <BoundingBoxVisibility />
      </div>
      <div className="inpainting-bounding-box-settings-items">
        <BoundingBoxDimensionSlider dimension="width" />
        <BoundingBoxDimensionSlider dimension="height" />
        <Flex alignItems={'center'} justifyContent={'space-between'}>
          <BoundingBoxDarkenOutside />
          <BoundingBoxLock />
        </Flex>
      </div>
    </div>
  );
};

export default BoundingBoxSettings;
