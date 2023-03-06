import { ButtonGroup } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  BiReset,
  BiRotateLeft,
  BiRotateRight,
  BiZoomIn,
  BiZoomOut,
} from 'react-icons/bi';
import { MdFlip } from 'react-icons/md';
import { useTransformContext } from 'react-zoom-pan-pinch';

type ReactPanZoomButtonsProps = {
  flipHorizontally: () => void;
  flipVertically: () => void;
  rotateCounterClockwise: () => void;
  rotateClockwise: () => void;
  reset: () => void;
};

const ReactPanZoomButtons = ({
  flipHorizontally,
  flipVertically,
  rotateCounterClockwise,
  rotateClockwise,
  reset,
}: ReactPanZoomButtonsProps) => {
  const { zoomIn, zoomOut, resetTransform } = useTransformContext();

  return (
    <ButtonGroup isAttached orientation="vertical">
      <IAIIconButton
        icon={<BiZoomIn />}
        aria-label="Zoom In"
        tooltip="Zoom In"
        onClick={() => zoomIn()}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiZoomOut />}
        aria-label="Zoom Out"
        tooltip="Zoom Out"
        onClick={() => zoomOut()}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiRotateLeft />}
        aria-label="Rotate Counter-Clockwise"
        tooltip="Rotate Counter-Clockwise"
        onClick={rotateCounterClockwise}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiRotateRight />}
        aria-label="Rotate Clockwise"
        tooltip="Rotate Clockwise"
        onClick={rotateClockwise}
        fontSize={20}
      />

      <IAIIconButton
        icon={<MdFlip />}
        aria-label="Flip Horizontally"
        tooltip="Flip Horizontally"
        onClick={flipHorizontally}
        fontSize={20}
      />

      <IAIIconButton
        icon={<MdFlip style={{ transform: 'rotate(90deg)' }} />}
        aria-label="Flip Vertically"
        tooltip="Flip Vertically"
        onClick={flipVertically}
        fontSize={20}
      />

      <IAIIconButton
        icon={<BiReset />}
        aria-label="Reset"
        tooltip="Reset"
        onClick={() => {
          resetTransform();
          reset();
        }}
        fontSize={20}
      />
    </ButtonGroup>
  );
};

export default ReactPanZoomButtons;
