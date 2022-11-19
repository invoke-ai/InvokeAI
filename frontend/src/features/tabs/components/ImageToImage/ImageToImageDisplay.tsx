import { RootState, useAppSelector } from 'app/store';
import ImageUploadButton from 'common/components/ImageUploaderButton';
import CurrentImageDisplay from 'features/gallery/components/CurrentImageDisplay';
import InitImagePreview from './InitImagePreview';

const ImageToImageDisplay = () => {
  const initialImage = useAppSelector(
    (state: RootState) => state.options.initialImage
  );

  const { currentImage } = useAppSelector((state: RootState) => state.gallery);

  const imageToImageComponent = initialImage ? (
    <div className="image-to-image-area">
      <InitImagePreview />
    </div>
  ) : (
    <ImageUploadButton />
  );

  return (
    <div className="workarea-split-view">
      <div className="workarea-split-view-left">{imageToImageComponent}</div>
      {currentImage && (
        <div className="workarea-split-view-right">
          <CurrentImageDisplay />
        </div>
      )}
    </div>
  );
};

export default ImageToImageDisplay;
