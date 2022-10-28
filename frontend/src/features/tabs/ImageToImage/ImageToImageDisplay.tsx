import { uploadImage } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import InvokeImageUploader from '../../../common/components/InvokeImageUploader';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import InitImagePreview from './InitImagePreview';

const ImageToImageDisplay = () => {
  const dispatch = useAppDispatch();

  const initialImage = useAppSelector(
    (state: RootState) => state.options.initialImage
  );

  const { currentImage } = useAppSelector((state: RootState) => state.gallery);

  const imageToImageComponent = initialImage ? (
    <div className="image-to-image-area">
      <InitImagePreview />
    </div>
  ) : (
    <InvokeImageUploader
      handleFile={(file: File) =>
        dispatch(uploadImage({ file, destination: 'img2img' }))
      }
    />
  );

  return (
    <div className="workarea-split-view">
      <div className="workarea-split-view-left">{imageToImageComponent} </div>
      {currentImage && (
        <div className="workarea-split-view-right">
          <CurrentImageDisplay />
        </div>
      )}
    </div>
  );
};

export default ImageToImageDisplay;
