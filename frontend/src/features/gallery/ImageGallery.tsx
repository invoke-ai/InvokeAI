import { Button, IconButton } from '@chakra-ui/button';
import { Resizable } from 're-resizable';

import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdClear, MdPhotoLibrary } from 'react-icons/md';
import { requestImages } from '../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import {
  setAutoRefresh,
  selectNextImage,
  selectPrevImage,
} from './gallerySlice';
import HoverableImage from './HoverableImage';
import { setShouldShowGallery } from '../options/optionsSlice';
import IAISwitch from '../../common/components/IAISwitch';

export default function ImageGallery() {
  const { images, currentImageUuid, areMoreImagesAvailable } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const shouldShowGallery = useAppSelector(
    (state: RootState) => state.options.shouldShowGallery
  );

  const activeTab = useAppSelector(
    (state: RootState) => state.options.activeTab
  );

  const shouldRefreshCurrentImage = useAppSelector(
    (state: RootState) => state.gallery.autoRefresh
  );

  const dispatch = useAppDispatch();

  const handleShowGalleryToggle = () => {
    dispatch(setShouldShowGallery(!shouldShowGallery));
  };

  const handleGalleryClose = () => {
    dispatch(setShouldShowGallery(false));
  };

  const handleToggleAutoRefresh = () => {
    dispatch(setAutoRefresh(!shouldRefreshCurrentImage));
  };

  const handleClickLoadMore = () => {
    dispatch(requestImages());
  };

  useHotkeys(
    'g',
    () => {
      handleShowGalleryToggle();
    },
    [shouldShowGallery]
  );

  useHotkeys(
    'left',
    () => {
      dispatch(selectPrevImage());
    },
    []
  );

  useHotkeys(
    'right',
    () => {
      dispatch(selectNextImage());
    },
    []
  );

  return (
    <div className="image-gallery-area">
      {!shouldShowGallery && (
        <IAIIconButton
          tooltip="Show Gallery"
          tooltipPlacement="top"
          aria-label="Show Gallery"
          onClick={handleShowGalleryToggle}
          className="image-gallery-popup-btn"
        >
          <MdPhotoLibrary />
        </IAIIconButton>
      )}

      {shouldShowGallery && (
        <Resizable
          defaultSize={{ width: '300', height: '100%' }}
          minWidth={'300'}
          maxWidth={activeTab == 1 ? '300' : '600'}
          className="image-gallery-popup"
        >
          <div className="image-gallery-header">
            <h1>Your Invocations</h1>
            <IconButton
              size={'sm'}
              aria-label={'Close Gallery'}
              onClick={handleGalleryClose}
              className="image-gallery-close-btn"
              icon={<MdClear />}
            />
          </div>
          <div className="image-gallery-auto-refresh-container">
            <h2>Auto Upate Selected Image</h2>
            <IAISwitch
              isChecked={shouldRefreshCurrentImage}
              onChange={handleToggleAutoRefresh}
              className="image-gallery-auto-refresh-switch"
              aria-lable={'Auto refresh current image'}
            />
          </div>
          <div className="image-gallery-container">
            {images.length ? (
              <div className="image-gallery">
                {images.map((image) => {
                  const { uuid } = image;
                  const isSelected = currentImageUuid === uuid;
                  return (
                    <HoverableImage
                      key={uuid}
                      image={image}
                      isSelected={isSelected}
                    />
                  );
                })}
              </div>
            ) : (
              <div className="image-gallery-container-placeholder">
                <MdPhotoLibrary />
                <p>No Images In Gallery</p>
              </div>
            )}
            <Button
              onClick={handleClickLoadMore}
              isDisabled={!areMoreImagesAvailable}
              className="image-gallery-load-more-btn"
            >
              {areMoreImagesAvailable ? 'Load More' : 'All Images Loaded'}
            </Button>
          </div>
        </Resizable>
      )}
    </div>
  );
}
