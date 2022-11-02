import { Button } from '@chakra-ui/button';
import { NumberSize, Resizable } from 're-resizable';

import { ChangeEvent, useEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdPhotoLibrary } from 'react-icons/md';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { requestImages } from '../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import {
  selectNextImage,
  selectPrevImage,
  setCurrentCategory,
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setGalleryScrollPosition,
  setGalleryWidth,
  setShouldAutoSwitchToNewImages,
  setShouldHoldGalleryOpen,
  setShouldPinGallery,
} from './gallerySlice';
import HoverableImage from './HoverableImage';
import { setShouldShowGallery } from '../gallery/gallerySlice';
import { ButtonGroup, useToast } from '@chakra-ui/react';
import { CSSTransition } from 'react-transition-group';
import { Direction } from 're-resizable/lib/resizer';
import { imageGallerySelector } from './gallerySliceSelectors';
import { FaImage, FaUser, FaWrench } from 'react-icons/fa';
import IAIPopover from '../../common/components/IAIPopover';
import IAISlider from '../../common/components/IAISlider';
import { BiReset } from 'react-icons/bi';
import IAICheckbox from '../../common/components/IAICheckbox';
import { setNeedsCache } from '../tabs/Inpainting/inpaintingSlice';
import _ from 'lodash';
import useClickOutsideWatcher from '../../common/hooks/useClickOutsideWatcher';

const GALLERY_SHOW_BUTTONS_MIN_WIDTH = 320;

export default function ImageGallery() {
  const dispatch = useAppDispatch();
  const toast = useToast();

  const {
    images,
    currentCategory,
    currentImageUuid,
    shouldPinGallery,
    shouldShowGallery,
    galleryScrollPosition,
    galleryImageMinimumWidth,
    galleryGridTemplateColumns,
    activeTabName,
    galleryImageObjectFit,
    shouldHoldGalleryOpen,
    shouldAutoSwitchToNewImages,
    areMoreImagesAvailable,
    galleryWidth,
  } = useAppSelector(imageGallerySelector);

  const [galleryMinWidth, setGalleryMinWidth] = useState<number>(300);
  const [galleryMaxWidth, setGalleryMaxWidth] = useState<number>(590);

  const [shouldShowButtons, setShouldShowButtons] = useState<boolean>(
    galleryWidth >= GALLERY_SHOW_BUTTONS_MIN_WIDTH
  );

  useEffect(() => {
    if (!shouldPinGallery) return;

    if (activeTabName === 'inpainting') {
      dispatch(setGalleryWidth(190));
      setGalleryMinWidth(190);
      setGalleryMaxWidth(190);
    } else if (activeTabName === 'img2img') {
      dispatch(
        setGalleryWidth(Math.min(Math.max(Number(galleryWidth), 0), 490))
      );
      setGalleryMaxWidth(490);
    } else {
      dispatch(
        setGalleryWidth(Math.min(Math.max(Number(galleryWidth), 0), 590))
      );
      setGalleryMaxWidth(590);
    }
    dispatch(setNeedsCache(true));
  }, [dispatch, activeTabName, shouldPinGallery, galleryWidth]);

  useEffect(() => {
    if (!shouldPinGallery) {
      setGalleryMaxWidth(window.innerWidth);
    }
  }, [shouldPinGallery]);

  const galleryRef = useRef<HTMLDivElement>(null);
  const galleryContainerRef = useRef<HTMLDivElement>(null);
  const timeoutIdRef = useRef<number | null>(null);

  const handleSetShouldPinGallery = () => {
    dispatch(setShouldPinGallery(!shouldPinGallery));
    dispatch(setNeedsCache(true));
  };

  const handleToggleGallery = () => {
    shouldShowGallery ? handleCloseGallery() : handleOpenGallery();
  };

  const handleOpenGallery = () => {
    dispatch(setShouldShowGallery(true));
    shouldPinGallery && dispatch(setNeedsCache(true));
  };

  const handleCloseGallery = () => {
    // if (shouldPinGallery) return;
    dispatch(setShouldShowGallery(false));
    dispatch(
      setGalleryScrollPosition(
        galleryContainerRef.current ? galleryContainerRef.current.scrollTop : 0
      )
    );
    dispatch(setShouldHoldGalleryOpen(false));
    // dispatch(setNeedsCache(true));
  };

  const handleClickLoadMore = () => {
    dispatch(requestImages(currentCategory));
  };

  const handleChangeGalleryImageMinimumWidth = (v: number) => {
    dispatch(setGalleryImageMinimumWidth(v));
    dispatch(setNeedsCache(true));
  };

  const setCloseGalleryTimer = () => {
    timeoutIdRef.current = window.setTimeout(() => handleCloseGallery(), 500);
  };

  const cancelCloseGalleryTimer = () => {
    timeoutIdRef.current && window.clearTimeout(timeoutIdRef.current);
  };

  useHotkeys(
    'g',
    () => {
      handleToggleGallery();
    },
    [shouldShowGallery]
  );

  useHotkeys('left', () => {
    dispatch(selectPrevImage());
  });

  useHotkeys('right', () => {
    dispatch(selectNextImage());
  });

  useHotkeys(
    'shift+g',
    () => {
      handleSetShouldPinGallery();
    },
    [shouldPinGallery]
  );

  useHotkeys(
    'esc',
    () => {
      if (shouldPinGallery) return;
      dispatch(setShouldShowGallery(false));
    },
    [shouldPinGallery]
  );

  const IMAGE_SIZE_STEP = 32;

  useHotkeys(
    'shift+up',
    () => {
      if (galleryImageMinimumWidth >= 256) {
        return;
      }
      if (galleryImageMinimumWidth < 256) {
        const newMinWidth = galleryImageMinimumWidth + IMAGE_SIZE_STEP;
        if (newMinWidth <= 256) {
          dispatch(setGalleryImageMinimumWidth(newMinWidth));
          toast({
            title: `Gallery Thumbnail Size set to ${newMinWidth}`,
            status: 'success',
            duration: 1000,
            isClosable: true,
          });
        } else {
          dispatch(setGalleryImageMinimumWidth(256));
          toast({
            title: `Gallery Thumbnail Size set to 256`,
            status: 'success',
            duration: 1000,
            isClosable: true,
          });
        }
      }
    },
    [galleryImageMinimumWidth]
  );

  useHotkeys(
    'shift+down',
    () => {
      if (galleryImageMinimumWidth <= 32) {
        return;
      }
      if (galleryImageMinimumWidth > 32) {
        const newMinWidth = galleryImageMinimumWidth - IMAGE_SIZE_STEP;
        if (newMinWidth > 32) {
          dispatch(setGalleryImageMinimumWidth(newMinWidth));
          toast({
            title: `Gallery Thumbnail Size set to ${newMinWidth}`,
            status: 'success',
            duration: 1000,
            isClosable: true,
          });
        } else {
          dispatch(setGalleryImageMinimumWidth(32));
          toast({
            title: `Gallery Thumbnail Size set to 32`,
            status: 'success',
            duration: 1000,
            isClosable: true,
          });
        }
      }
    },
    [galleryImageMinimumWidth]
  );

  useHotkeys(
    'shift+r',
    () => {
      dispatch(setGalleryImageMinimumWidth(64));
      toast({
        title: `Reset Gallery Image Size`,
        status: 'success',
        duration: 2500,
        isClosable: true,
      });
    },
    [galleryImageMinimumWidth]
  );

  // set gallery scroll position
  useEffect(() => {
    if (!galleryContainerRef.current) return;
    galleryContainerRef.current.scrollTop = galleryScrollPosition;
  }, [galleryScrollPosition, shouldShowGallery]);

  useEffect(() => {
    setShouldShowButtons(galleryWidth >= 280);
  }, [galleryWidth]);

  useClickOutsideWatcher(galleryRef, handleCloseGallery, !shouldPinGallery);

  return (
    <CSSTransition
      nodeRef={galleryRef}
      in={shouldShowGallery || (shouldHoldGalleryOpen && !shouldPinGallery)}
      unmountOnExit
      timeout={200}
      classNames="image-gallery-wrapper"
    >
      <div
        className="image-gallery-wrapper"
        style={{ zIndex: shouldPinGallery ? 1 : 100 }}
        data-pinned={shouldPinGallery}
        ref={galleryRef}
        onMouseLeave={!shouldPinGallery ? setCloseGalleryTimer : undefined}
        onMouseEnter={!shouldPinGallery ? cancelCloseGalleryTimer : undefined}
        onMouseOver={!shouldPinGallery ? cancelCloseGalleryTimer : undefined}
      >
        <Resizable
          minWidth={galleryMinWidth}
          maxWidth={galleryMaxWidth}
          className={'image-gallery-popup'}
          handleStyles={{ left: { width: '15px' } }}
          enable={{
            top: false,
            right: false,
            bottom: false,
            left: true,
            topRight: false,
            bottomRight: false,
            bottomLeft: false,
            topLeft: false,
          }}
          size={{
            width: galleryWidth,
            height: shouldPinGallery ? '100%' : '100vh',
          }}
          onResizeStop={(
            _event: MouseEvent | TouchEvent,
            _direction: Direction,
            elementRef: HTMLElement,
            delta: NumberSize
          ) => {
            dispatch(
              setGalleryWidth(
                _.clamp(
                  Number(galleryWidth) + delta.width,
                  0,
                  Number(galleryMaxWidth)
                )
              )
            );
            elementRef.removeAttribute('data-resize-alert');
          }}
          onResize={(
            _event: MouseEvent | TouchEvent,
            _direction: Direction,
            elementRef: HTMLElement,
            delta: NumberSize
          ) => {
            const newWidth = _.clamp(
              Number(galleryWidth) + delta.width,
              0,
              Number(galleryMaxWidth)
            );

            if (newWidth >= 280 && !shouldShowButtons) {
              setShouldShowButtons(true);
            } else if (newWidth < 280 && shouldShowButtons) {
              setShouldShowButtons(false);
            }

            if (newWidth >= galleryMaxWidth) {
              elementRef.setAttribute('data-resize-alert', 'true');
            } else {
              elementRef.removeAttribute('data-resize-alert');
            }
          }}
        >
          <div className="image-gallery-header">
            <div>
              <ButtonGroup
                size="sm"
                isAttached
                variant="solid"
                className="image-gallery-category-btn-group"
              >
                {shouldShowButtons ? (
                  <>
                    <Button
                      data-selected={currentCategory === 'result'}
                      onClick={() => dispatch(setCurrentCategory('result'))}
                    >
                      Invocations
                    </Button>
                    <Button
                      data-selected={currentCategory === 'user'}
                      onClick={() => dispatch(setCurrentCategory('user'))}
                    >
                      User
                    </Button>
                  </>
                ) : (
                  <>
                    <IAIIconButton
                      aria-label="Show Invocations"
                      tooltip="Show Invocations"
                      data-selected={currentCategory === 'result'}
                      icon={<FaImage />}
                      onClick={() => dispatch(setCurrentCategory('result'))}
                    />
                    <IAIIconButton
                      aria-label="Show Uploads"
                      tooltip="Show Uploads"
                      data-selected={currentCategory === 'user'}
                      icon={<FaUser />}
                      onClick={() => dispatch(setCurrentCategory('user'))}
                    />
                  </>
                )}
              </ButtonGroup>
            </div>
            <div>
              <IAIPopover
                isLazy
                trigger="hover"
                placement={'left'}
                triggerComponent={
                  <IAIIconButton
                    size={'sm'}
                    aria-label={'Gallery Settings'}
                    icon={<FaWrench />}
                    className="image-gallery-icon-btn"
                    cursor={'pointer'}
                  />
                }
              >
                <div className="image-gallery-settings-popover">
                  <div>
                    <IAISlider
                      value={galleryImageMinimumWidth}
                      onChange={handleChangeGalleryImageMinimumWidth}
                      min={32}
                      max={256}
                      width={100}
                      label={'Image Size'}
                      formLabelProps={{ style: { fontSize: '0.9rem' } }}
                      sliderThumbTooltipProps={{
                        label: `${galleryImageMinimumWidth}px`,
                      }}
                    />
                    <IAIIconButton
                      size={'sm'}
                      aria-label={'Reset'}
                      tooltip={'Reset Size'}
                      onClick={() => dispatch(setGalleryImageMinimumWidth(64))}
                      icon={<BiReset />}
                      data-selected={shouldPinGallery}
                      styleClass="image-gallery-icon-btn"
                    />
                  </div>
                  <div>
                    <IAICheckbox
                      label="Maintain Aspect Ratio"
                      isChecked={galleryImageObjectFit === 'contain'}
                      onChange={() =>
                        dispatch(
                          setGalleryImageObjectFit(
                            galleryImageObjectFit === 'contain'
                              ? 'cover'
                              : 'contain'
                          )
                        )
                      }
                    />
                  </div>
                  <div>
                    <IAICheckbox
                      label="Auto-Switch to New Images"
                      isChecked={shouldAutoSwitchToNewImages}
                      onChange={(e: ChangeEvent<HTMLInputElement>) =>
                        dispatch(
                          setShouldAutoSwitchToNewImages(e.target.checked)
                        )
                      }
                    />
                  </div>
                </div>
              </IAIPopover>

              <IAIIconButton
                size={'sm'}
                className={'image-gallery-icon-btn'}
                aria-label={'Pin Gallery'}
                tooltip={'Pin Gallery (Shift+G)'}
                onClick={handleSetShouldPinGallery}
                icon={shouldPinGallery ? <BsPinAngleFill /> : <BsPinAngle />}
              />
            </div>
          </div>
          <div className="image-gallery-container" ref={galleryContainerRef}>
            {images.length || areMoreImagesAvailable ? (
              <>
                <div
                  className="image-gallery"
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
                </div>
                <Button
                  onClick={handleClickLoadMore}
                  isDisabled={!areMoreImagesAvailable}
                  className="image-gallery-load-more-btn"
                >
                  {areMoreImagesAvailable ? 'Load More' : 'All Images Loaded'}
                </Button>
              </>
            ) : (
              <div className="image-gallery-container-placeholder">
                <MdPhotoLibrary />
                <p>No Images In Gallery</p>
              </div>
            )}
          </div>
        </Resizable>
      </div>
    </CSSTransition>
  );
}
