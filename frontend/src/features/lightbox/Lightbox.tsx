import {
  IconButton,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalOverlay,
  useDisclosure,
} from '@chakra-ui/react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { imagesSelector } from 'features/gallery/CurrentImagePreview';
import {
  selectNextImage,
  selectPrevImage,
} from 'features/gallery/gallerySlice';
import ImageGallery from 'features/gallery/ImageGallery';
import { setIsLightBoxOpen } from 'features/options/optionsSlice';
import React, { cloneElement, ReactElement, useEffect, useState } from 'react';
import { FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import ReactPanZoom from './ReactPanZoom';

type LightBoxProps = {
  children: ReactElement;
};

export default function Lightbox({ children }: LightBoxProps) {
  const {
    isOpen: lightBoxOpen,
    onOpen: onLightBoxOpen,
    onClose: onLightBoxClose,
  } = useDisclosure();

  const dispatch = useAppDispatch();
  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.options.isLightBoxOpen
  );

  useEffect(() => {
    dispatch(setIsLightBoxOpen(lightBoxOpen));
  }, [lightBoxOpen, dispatch]);

  const {
    imageToDisplay,
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

  return (
    <>
      {cloneElement(children, {
        onClick: onLightBoxOpen,
      })}

      <Modal isOpen={isLightBoxOpen} onClose={onLightBoxClose} size="full">
        <ModalOverlay />
        <ModalContent className="modal lightbox-container">
          <ModalCloseButton
            className="modal-close-btn lightbox-close-btn"
            onClick={() => {
              dispatch(setIsLightBoxOpen(true));
            }}
          />
          <ModalBody>
            <div className="lightbox-display-container">
              <div className="lightbox-preview-wrapper">
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
                {imageToDisplay && (
                  <ReactPanZoom
                    image={imageToDisplay.url}
                    styleClass="lightbox-image"
                  />
                )}
              </div>
              {isLightBoxOpen && <ImageGallery />}
            </div>
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
}
