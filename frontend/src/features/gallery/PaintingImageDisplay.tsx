import { RootState, useAppSelector } from "../../app/store";
import { MdPhoto } from "react-icons/md";
import PaintingCanvas from "./Canvas/PaintingCanvas";
import PaintingButtons from "./PaintingButtons";

import CurrentCanvasImage from "./Canvas/CurrentCanvasImage";
import MaskCanvas from "./Canvas/MaskCanvas";

const PaintingImageDisplay = () => {
  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const imageToDisplay = intermediateImage || currentImage;

  return imageToDisplay ? (
    <div className="painting-image-display">
      <div className="painting-image-tools">
        <PaintingButtons />
      </div>
      <PaintingCanvas>
        <CurrentCanvasImage />
      </PaintingCanvas>
    </div>
  ) : (
    <div className="painting-image-display-placeholder">
      <MdPhoto />
    </div>
  );
}

export default PaintingImageDisplay;