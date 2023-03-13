import { ButtonGroup, Flex, Grid, Icon, Text } from '@chakra-ui/react';
import { requestImages } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { imageGallerySelector } from 'features/gallery/store/gallerySelectors';
import {
  setCurrentCategory,
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setShouldAutoSwitchToNewImages,
  setShouldUseSingleGalleryColumn,
} from 'features/gallery/store/gallerySlice';
import { togglePinGalleryPanel } from 'features/ui/store/uiSlice';

import { ChangeEvent, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { FaImage, FaUser, FaWrench } from 'react-icons/fa';
import { MdPhotoLibrary } from 'react-icons/md';
import HoverableImage from './HoverableImage';

import Scrollable from 'features/ui/components/common/Scrollable';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';

const GALLERY_SHOW_BUTTONS_MIN_WIDTH = 290;

const ImageGalleryContent = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const resizeObserverRef = useRef<HTMLDivElement>(null);
  const [shouldShouldIconButtons, setShouldShouldIconButtons] = useState(true);

  const {
    images,
    currentCategory,
    currentImageUuid,
    shouldPinGallery,
    galleryImageMinimumWidth,
    galleryGridTemplateColumns,
    galleryImageObjectFit,
    shouldAutoSwitchToNewImages,
    areMoreImagesAvailable,
    shouldUseSingleGalleryColumn,
  } = useAppSelector(imageGallerySelector);

  const handleClickLoadMore = () => {
    dispatch(requestImages(currentCategory));
  };

  const handleChangeGalleryImageMinimumWidth = (v: number) => {
    dispatch(setGalleryImageMinimumWidth(v));
  };

  const handleSetShouldPinGallery = () => {
    dispatch(togglePinGalleryPanel());
    dispatch(requestCanvasRescale());
  };

  useEffect(() => {
    if (!resizeObserverRef.current) {
      return;
    }
    const resizeObserver = new ResizeObserver(() => {
      if (!resizeObserverRef.current) {
        return;
      }

      if (
        resizeObserverRef.current.clientWidth < GALLERY_SHOW_BUTTONS_MIN_WIDTH
      ) {
        setShouldShouldIconButtons(true);
        return;
      }

      setShouldShouldIconButtons(false);
    });
    resizeObserver.observe(resizeObserverRef.current);
    return () => resizeObserver.disconnect(); // clean up
  }, []);

  return (
    <Flex flexDirection="column" w="full" h="full" gap={4}>
      <Flex
        ref={resizeObserverRef}
        alignItems="center"
        justifyContent="space-between"
      >
        <ButtonGroup
          size="sm"
          isAttached
          w="max-content"
          justifyContent="stretch"
        >
          {shouldShouldIconButtons ? (
            <>
              <IAIIconButton
                aria-label={t('gallery.showGenerations')}
                tooltip={t('gallery.showGenerations')}
                isChecked={currentCategory === 'result'}
                role="radio"
                icon={<FaImage />}
                onClick={() => dispatch(setCurrentCategory('result'))}
              />
              <IAIIconButton
                aria-label={t('gallery.showUploads')}
                tooltip={t('gallery.showUploads')}
                role="radio"
                isChecked={currentCategory === 'user'}
                icon={<FaUser />}
                onClick={() => dispatch(setCurrentCategory('user'))}
              />
            </>
          ) : (
            <>
              <IAIButton
                size="sm"
                isChecked={currentCategory === 'result'}
                onClick={() => dispatch(setCurrentCategory('result'))}
                flexGrow={1}
              >
                {t('gallery.generations')}
              </IAIButton>
              <IAIButton
                size="sm"
                isChecked={currentCategory === 'user'}
                onClick={() => dispatch(setCurrentCategory('user'))}
                flexGrow={1}
              >
                {t('gallery.uploads')}
              </IAIButton>
            </>
          )}
        </ButtonGroup>

        <Flex gap={2}>
          <IAIPopover
            triggerComponent={
              <IAIIconButton
                size="sm"
                aria-label={t('gallery.gallerySettings')}
                icon={<FaWrench />}
              />
            }
          >
            <Flex direction="column" gap={2}>
              <IAISlider
                value={galleryImageMinimumWidth}
                onChange={handleChangeGalleryImageMinimumWidth}
                min={32}
                max={256}
                hideTooltip={true}
                label={t('gallery.galleryImageSize')}
                withReset
                handleReset={() => dispatch(setGalleryImageMinimumWidth(64))}
              />
              <IAICheckbox
                label={t('gallery.maintainAspectRatio')}
                isChecked={galleryImageObjectFit === 'contain'}
                onChange={() =>
                  dispatch(
                    setGalleryImageObjectFit(
                      galleryImageObjectFit === 'contain' ? 'cover' : 'contain'
                    )
                  )
                }
              />
              <IAICheckbox
                label={t('gallery.autoSwitchNewImages')}
                isChecked={shouldAutoSwitchToNewImages}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setShouldAutoSwitchToNewImages(e.target.checked))
                }
              />
              <IAICheckbox
                label={t('gallery.singleColumnLayout')}
                isChecked={shouldUseSingleGalleryColumn}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  dispatch(setShouldUseSingleGalleryColumn(e.target.checked))
                }
              />
            </Flex>
          </IAIPopover>

          <IAIIconButton
            size="sm"
            aria-label={t('gallery.pinGallery')}
            tooltip={`${t('gallery.pinGallery')} (Shift+G)`}
            onClick={handleSetShouldPinGallery}
            icon={shouldPinGallery ? <BsPinAngleFill /> : <BsPinAngle />}
          />
        </Flex>
      </Flex>
      <Scrollable>
        <Flex direction="column" gap={2} h="full">
          {images.length || areMoreImagesAvailable ? (
            <>
              <Grid
                gap={2}
                style={{ gridTemplateColumns: galleryGridTemplateColumns }}
              >
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
              </Grid>
              <IAIButton
                onClick={handleClickLoadMore}
                isDisabled={!areMoreImagesAvailable}
                flexShrink={0}
              >
                {areMoreImagesAvailable
                  ? t('gallery.loadMore')
                  : t('gallery.allImagesLoaded')}
              </IAIButton>
            </>
          ) : (
            <Flex
              sx={{
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 2,
                padding: 8,
                h: '100%',
                w: '100%',
                color: 'base.500',
              }}
            >
              <Icon
                as={MdPhotoLibrary}
                sx={{
                  w: 16,
                  h: 16,
                }}
              />
              <Text textAlign="center">{t('gallery.noImagesInGallery')}</Text>
            </Flex>
          )}
        </Flex>
      </Scrollable>
    </Flex>
  );
};

ImageGalleryContent.displayName = 'ImageGalleryContent';
export default ImageGalleryContent;
