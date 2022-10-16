import { RootState, useAppSelector } from "../../app/store";
import CurrentImageButtons from "./CurrentImageButtons";
import { MdPhoto } from "react-icons/md";
import OutpaintingImagePreview from "./OutpaintingImagePreview";

const OutpaintingImageDisplay = () => {
  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const shouldShowImageDetails = useAppSelector(
    (state: RootState) => state.options.shouldShowImageDetails
  );

  const imageToDisplay = intermediateImage || currentImage;

  return imageToDisplay ? (
    <div className="outpainting-image-display">
      <div className="outpainting-image-tools">
        <CurrentImageButtons image={imageToDisplay} />
      </div>
      <OutpaintingImagePreview imageToDisplay={imageToDisplay} />
    </div>
  ) : (
    <div className="outpainting-image-display-placeholder">
      <MdPhoto />
    </div>
  );
}

export default OutpaintingImageDisplay;