import { IconButton } from '@chakra-ui/react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import CurrentImageButtons from 'features/gallery/components/CurrentImageButtons';
import { imagesSelector } from 'features/gallery/components/CurrentImagePreview';
import {
  selectNextImage,
  selectPrevImage,
} from 'features/gallery/store/gallerySlice';
import ImageGallery from 'features/gallery/components/ImageGallery';
import ImageMetadataViewer from 'features/gallery/components/ImageMetaDataViewer/ImageMetadataViewer';
import { setIsLightBoxOpen } from 'features/options/store/optionsSlice';
import React, { useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { BiExit } from 'react-icons/bi';
import { FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import ReactPanZoom from './ReactPanZoom';

export default function Lightbox() {
  const dispatch = useAppDispatch();
  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.options.isLightBoxOpen
  );

  const {
    viewerImageToDisplay,
    shouldShowImageDetails,
    isOnFirstImage,
    isOnLastImage,
  } = useAppSelector(imagesSelector);

  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const handleCurrentImagePreviewMouseOver = () => {
    setShouldShowNextPrevButtons(true);
  };

  const handleCurrentImagePreviewMouseOut = () => {
    setShouldShowNextPrevButtons(false);
  };

  const handleClickPrevButton = () => {
    dispatch(selectPrevImage());
  };

  const handleClickNextButton = () => {
    dispatch(selectNextImage());
  };

  useHotkeys(
    'Esc',
    () => {
      if (isLightBoxOpen) dispatch(setIsLightBoxOpen(false));
    },
    [isLightBoxOpen]
  );

  return (
    <div className="lightbox-container">
      <IAIIconButton
        icon={<BiExit />}
        aria-label="Exit Viewer"
        className="lightbox-close-btn"
        onClick={() => {
          dispatch(setIsLightBoxOpen(false));
        }}
        fontSize={20}
      />

      <div className="lightbox-display-container">
        <div className="lightbox-preview-wrapper">
          <CurrentImageButtons />
          {!shouldShowImageDetails && (
            <div className="current-image-next-prev-buttons">
              <div
                className="next-prev-button-trigger-area prev-button-trigger-area"
                onMouseOver={handleCurrentImagePreviewMouseOver}
                onMouseOut={handleCurrentImagePreviewMouseOut}
              >
                {shouldShowNextPrevButtons && !isOnFirstImage && (
                  <IconButton
                    aria-label="Previous image"
                    icon={<FaAngleLeft className="next-prev-button" />}
                    variant="unstyled"
                    onClick={handleClickPrevButton}
                  />
                )}
              </div>
              <div
                className="next-prev-button-trigger-area next-button-trigger-area"
                onMouseOver={handleCurrentImagePreviewMouseOver}
                onMouseOut={handleCurrentImagePreviewMouseOut}
              >
                {shouldShowNextPrevButtons && !isOnLastImage && (
                  <IconButton
                    aria-label="Next image"
                    icon={<FaAngleRight className="next-prev-button" />}
                    variant="unstyled"
                    onClick={handleClickNextButton}
                  />
                )}
              </div>
            </div>
          )}
          {viewerImageToDisplay && (
            <>
              <ReactPanZoom
                image={viewerImageToDisplay.url}
                styleClass="lightbox-image"
              />
              {shouldShowImageDetails && (
                <ImageMetadataViewer image={viewerImageToDisplay} />
              )}
            </>
          )}
        </div>
        <ImageGallery />
      </div>
    </div>
  );
}
