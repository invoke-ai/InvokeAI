import { useState } from "react";
import { RootState, useAppSelector } from "../../app/store";
import { MdPhoto } from "react-icons/md";
import PaintingCanvas from "./Canvas/PaintingCanvas";
import PaintingButtons from "./PaintingButtons";

import CurrentCanvasImage from "./Canvas/CurrentCanvasImage";
import { tabMap } from "../tabs/InvokeTabs";
import OutpaintingOutline from "./Canvas/OutpaintingOutline";
import OutpaintingHighlight from "./Canvas/OutpaintingHighlight";

const PaintingImageDisplay = () => {
  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );
  const activeTab = useAppSelector(
    (state: RootState) => tabMap[state.options.activeTab]
  );

  // To improve: not to use useState
  const [onBrushClear, setOnBrushClear] = useState<(() => void) | undefined>(undefined);

  const imageToDisplay = intermediateImage || currentImage;

  return imageToDisplay ? (
    <div className="painting-image-display">
      <div className="painting-image-tools">
        <PaintingButtons onBrushClear={onBrushClear} />
      </div>
      <PaintingCanvas setOnBrushClear={setOnBrushClear} >
        <CurrentCanvasImage x={0} y={0} />
        {activeTab === "outpainting" && <OutpaintingOutline />}
        {activeTab === "outpainting" && <OutpaintingHighlight />}
      </PaintingCanvas>
    </div>
  ) : (
    <div className="painting-image-display-placeholder">
      <MdPhoto />
    </div>
  );
}

export default PaintingImageDisplay;