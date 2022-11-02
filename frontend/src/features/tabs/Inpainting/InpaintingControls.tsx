import InpaintingBrushControl from './InpaintingControls/InpaintingBrushControl';
import InpaintingEraserControl from './InpaintingControls/InpaintingEraserControl';
import InpaintingUndoControl from './InpaintingControls/InpaintingUndoControl';
import InpaintingRedoControl from './InpaintingControls/InpaintingRedoControl';
import { ButtonGroup } from '@chakra-ui/react';
import InpaintingMaskClear from './InpaintingControls/InpaintingMaskControls/InpaintingMaskClear';
import InpaintingMaskVisibilityControl from './InpaintingControls/InpaintingMaskControls/InpaintingMaskVisibilityControl';
import InpaintingMaskInvertControl from './InpaintingControls/InpaintingMaskControls/InpaintingMaskInvertControl';
import InpaintingLockBoundingBoxControl from './InpaintingControls/InpaintingLockBoundingBoxControl';
import InpaintingShowHideBoundingBoxControl from './InpaintingControls/InpaintingShowHideBoundingBoxControl';
import ImageUploaderIconButton from '../../../common/components/ImageUploaderIconButton';

const InpaintingControls = () => {
  return (
    <div className="inpainting-settings">
      <ButtonGroup isAttached={true}>
        <InpaintingBrushControl />
        <InpaintingEraserControl />
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <InpaintingMaskVisibilityControl />
        <InpaintingMaskInvertControl />
        <InpaintingLockBoundingBoxControl />
        <InpaintingShowHideBoundingBoxControl />
        <InpaintingMaskClear />
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <InpaintingUndoControl />
        <InpaintingRedoControl />
      </ButtonGroup>
      <ButtonGroup isAttached={true}>
        <ImageUploaderIconButton />
      </ButtonGroup>
    </div>
  );
};

export default InpaintingControls;
