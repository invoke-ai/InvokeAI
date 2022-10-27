import { Button } from '@chakra-ui/button';
import { NumberSize, Resizable, Size } from 're-resizable';

import { useEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdClear, MdPhotoLibrary } from 'react-icons/md';
import { BsPinAngleFill } from 'react-icons/bs';
import { requestImages } from '../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import {
  selectNextImage,
  selectPrevImage,
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setGalleryScrollPosition,
  setShouldHoldGalleryOpen,
  setShouldPinGallery,
} from './gallerySlice';
import HoverableImage from './HoverableImage';
import { setShouldShowGallery } from '../gallery/gallerySlice';
import { Spacer, useToast } from '@chakra-ui/react';
import { CSSTransition } from 'react-transition-group';
import { Direction } from 're-resizable/lib/resizer';
import { imageGallerySelector } from './gallerySliceSelectors';
import { FaWrench } from 'react-icons/fa';
import IAIPopover from '../../common/components/IAIPopover';
import IAISlider from '../../common/components/IAISlider';
import { BiReset } from 'react-icons/bi';
import IAICheckbox from '../../common/components/IAICheckbox';

export default function ImageGallery() {
  const dispatch = useAppDispatch();
  const toast = useToast();

  const {
    images,
    currentImageUuid,
    areMoreImagesAvailable,
    shouldPinGallery,
    shouldShowGallery,
    galleryScrollPosition,
    galleryImageMinimumWidth,
    galleryGridTemplateColumns,
    activeTabName,
    galleryImageObjectFit,
    shouldHoldGalleryOpen,
  } = useAppSelector(imageGallerySelector);

  const [gallerySize, setGallerySize] = useState<Size>({
    width: '300',
    height: '100%',
  });

  const [galleryMaxSize, setGalleryMaxSize] = useState<Size>({
    width: '590', // keep max at 590 for any tab
    height: '100%',
  });

  const [galleryMinSize, setGalleryMinSize] = useState<Size>({
    width: '300', // keep max at 590 for any tab
    height: '100%',
  });

  useEffect(() => {
    if (activeTabName === 'inpainting' && shouldPinGallery) {
      setGalleryMinSize((prevSize) => {
        return { ...prevSize, width: '200' };
      });
      setGalleryMaxSize((prevSize) => {
        return { ...prevSize, width: '200' };
      });
      setGallerySize((prevSize) => {
        return { ...prevSize, width: Math.min(Number(prevSize.width), 200) };
      });
    } else {
      setGalleryMaxSize((prevSize) => {
        return { ...prevSize, width: '590', height: '100%' };
      });
      setGallerySize((prevSize) => {
        return { ...prevSize, width: Math.min(Number(prevSize.width), 590) };
      });
    }
  }, [activeTabName, shouldPinGallery, setGalleryMaxSize]);

  useEffect(() => {
    if (!shouldPinGallery) {
      setGalleryMaxSize((prevSize) => {
        // calculate vh in px
        return {
          ...prevSize,
          width: window.innerWidth,
        };
      });
    }
  }, [shouldPinGallery]);

  const galleryRef = useRef<HTMLDivElement>(null);
  const galleryContainerRef = useRef<HTMLDivElement>(null);
  const timeoutIdRef = useRef<number | null>(null);

  const handleSetShouldPinGallery = () => {
    dispatch(setShouldPinGallery(!shouldPinGallery));
    setGallerySize({
      ...gallerySize,
      height: shouldPinGallery ? '100vh' : '100%',
    });
  };

  const handleToggleGallery = () => {
    shouldShowGallery ? handleCloseGallery() : handleOpenGallery();
  };

  const handleOpenGallery = () => {
    dispatch(setShouldShowGallery(true));
  };

  const handleCloseGallery = () => {
    dispatch(
      setGalleryScrollPosition(
        galleryContainerRef.current ? galleryContainerRef.current.scrollTop : 0
      )
    );
    if (!shouldHoldGalleryOpen) {
      dispatch(setShouldHoldGalleryOpen(false));
      setCloseGalleryTimer();
    }
    dispatch(setShouldShowGallery(false));
  };

  const handleClickLoadMore = () => {
    dispatch(requestImages());
  };

  const handleChangeGalleryImageMinimumWidth = (v: number) => {
    dispatch(setGalleryImageMinimumWidth(v));
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
    'shift+p',
    () => {
      handleSetShouldPinGallery();
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

  return (
    <CSSTransition
      nodeRef={galleryRef}
      in={shouldShowGallery || shouldHoldGalleryOpen}
      unmountOnExit
      timeout={200}
      classNames="image-gallery-area"
    >
      <div
        className="image-gallery-area"
        data-pinned={shouldPinGallery}
        ref={galleryRef}
        onMouseLeave={!shouldPinGallery ? setCloseGalleryTimer : undefined}
        onMouseEnter={!shouldPinGallery ? cancelCloseGalleryTimer : undefined}
        onMouseOver={!shouldPinGallery ? cancelCloseGalleryTimer : undefined}
      >
        <Resizable
          minWidth={galleryMinSize.width}
          maxWidth={galleryMaxSize.width}
          maxHeight={'100%'}
          className={'image-gallery-popup'}
          handleStyles={{ left: { width: '20px' } }}
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
          size={gallerySize}
          onResizeStop={(
            _event: MouseEvent | TouchEvent,
            _direction: Direction,
            elementRef: HTMLElement,
            delta: NumberSize
          ) => {
            setGallerySize({
              width: Number(gallerySize.width) + delta.width,
              height: '100%',
            });
            elementRef.removeAttribute('data-resize-alert');
          }}
          onResize={(
            _event: MouseEvent | TouchEvent,
            _direction: Direction,
            elementRef: HTMLElement,
            delta: NumberSize
          ) => {
            const newWidth = Number(gallerySize.width) + delta.width;
            if (newWidth >= galleryMaxSize.width) {
              elementRef.setAttribute('data-resize-alert', 'true');
            } else {
              elementRef.removeAttribute('data-resize-alert');
            }
          }}
        >
          <div className="image-gallery-header">
            {activeTabName !== 'inpainting' ? (
              <>
                <h1>Your Invocations</h1>
                <Spacer />
              </>
            ) : null}

            <IAIPopover
              trigger="click"
              hasArrow={activeTabName === 'inpainting' ? false : true}
              // styleClass="image-gallery-settings-popover"
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
              </div>
            </IAIPopover>

            <IAIIconButton
              size={'sm'}
              aria-label={'Pin Gallery'}
              tooltip={'Pin Gallery (Shift+P)'}
              onClick={handleSetShouldPinGallery}
              icon={<BsPinAngleFill />}
              data-selected={shouldPinGallery}
            />

            <IAIIconButton
              size={'sm'}
              aria-label={'Close Gallery'}
              tooltip={'Close Gallery (G)'}
              onClick={handleCloseGallery}
              className="image-gallery-icon-btn"
              icon={<MdClear />}
            />
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
