// import { NumberSize, Resizable } from 're-resizable';

// import {
//   Box,
//   ButtonGroup,
//   Flex,
//   Grid,
//   Icon,
//   chakra,
//   useTheme,
// } from '@chakra-ui/react';
// import { requestImages } from 'app/socketio/actions';
// import { useAppDispatch, useAppSelector } from 'app/storeHooks';
// import IAIButton from 'common/components/IAIButton';
// import IAICheckbox from 'common/components/IAICheckbox';
// import IAIIconButton from 'common/components/IAIIconButton';
// import IAIPopover from 'common/components/IAIPopover';
// import IAISlider from 'common/components/IAISlider';
// import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
// import { imageGallerySelector } from 'features/gallery/store/gallerySelectors';
// import {
//   selectNextImage,
//   selectPrevImage,
//   setCurrentCategory,
//   setGalleryImageMinimumWidth,
//   setGalleryImageObjectFit,
//   setGalleryWidth,
//   setShouldAutoSwitchToNewImages,
//   setShouldHoldGalleryOpen,
//   setShouldUseSingleGalleryColumn,
// } from 'features/gallery/store/gallerySlice';
// import {
//   setShouldPinGallery,
//   setShouldShowGallery,
// } from 'features/ui/store/uiSlice';
// import { InvokeTabName } from 'features/ui/store/tabMap';

// import { clamp } from 'lodash';
// import { Direction } from 're-resizable/lib/resizer';
// import React, {
//   ChangeEvent,
//   useCallback,
//   useEffect,
//   useRef,
//   useState,
// } from 'react';
// import { useHotkeys } from 'react-hotkeys-hook';
// import { useTranslation } from 'react-i18next';
// import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
// import { FaImage, FaUser, FaWrench } from 'react-icons/fa';
// import { MdPhotoLibrary } from 'react-icons/md';
// import { CSSTransition } from 'react-transition-group';
// import HoverableImage from './HoverableImage';
// import { APP_GALLERY_HEIGHT_PINNED } from 'theme/util/constants';

// import './ImageGallery.css';
// import { no_scrollbar } from 'theme/components/scrollbar';
// import ImageGalleryContent from './ImageGalleryContent';

// const ChakraResizeable = chakra(Resizable, {
//   shouldForwardProp: (prop) => !['sx'].includes(prop),
// });

// const GALLERY_SHOW_BUTTONS_MIN_WIDTH = 320;
// const GALLERY_IMAGE_WIDTH_OFFSET = 40;

// const GALLERY_TAB_WIDTHS: Record<
//   InvokeTabName,
//   { galleryMinWidth: number; galleryMaxWidth: number }
// > = {
//   txt2img: { galleryMinWidth: 200, galleryMaxWidth: 500 },
//   img2img: { galleryMinWidth: 200, galleryMaxWidth: 500 },
//   unifiedCanvas: { galleryMinWidth: 200, galleryMaxWidth: 200 },
//   nodes: { galleryMinWidth: 200, galleryMaxWidth: 500 },
//   postprocess: { galleryMinWidth: 200, galleryMaxWidth: 500 },
//   training: { galleryMinWidth: 200, galleryMaxWidth: 500 },
// };

// const LIGHTBOX_GALLERY_WIDTH = 400;

// export default function ImageGallery() {
//   const dispatch = useAppDispatch();
//   const { direction } = useTheme();

//   const { t } = useTranslation();

//   const {
//     images,
//     currentCategory,
//     currentImageUuid,
//     shouldPinGallery,
//     shouldShowGallery,
//     galleryImageMinimumWidth,
//     galleryGridTemplateColumns,
//     activeTabName,
//     galleryImageObjectFit,
//     shouldHoldGalleryOpen,
//     shouldAutoSwitchToNewImages,
//     areMoreImagesAvailable,
//     galleryWidth,
//     isLightboxOpen,
//     isStaging,
//     shouldEnableResize,
//     shouldUseSingleGalleryColumn,
//   } = useAppSelector(imageGallerySelector);

//   const { galleryMinWidth, galleryMaxWidth } = isLightboxOpen
//     ? {
//         galleryMinWidth: LIGHTBOX_GALLERY_WIDTH,
//         galleryMaxWidth: LIGHTBOX_GALLERY_WIDTH,
//       }
//     : GALLERY_TAB_WIDTHS[activeTabName];

//   const [shouldShowButtons, setShouldShowButtons] = useState<boolean>(
//     galleryWidth >= GALLERY_SHOW_BUTTONS_MIN_WIDTH
//   );

//   const [isResizing, setIsResizing] = useState(false);
//   const [galleryResizeHeight, setGalleryResizeHeight] = useState(0);

//   const galleryRef = useRef<HTMLDivElement>(null);
//   const galleryContainerRef = useRef<HTMLDivElement>(null);
//   const timeoutIdRef = useRef<number | null>(null);

//   useEffect(() => {
//     setShouldShowButtons(galleryWidth >= GALLERY_SHOW_BUTTONS_MIN_WIDTH);
//   }, [galleryWidth]);

//   const handleSetShouldPinGallery = () => {
//     !shouldPinGallery && dispatch(setShouldShowGallery(true));
//     dispatch(setShouldPinGallery(!shouldPinGallery));
//     dispatch(setDoesCanvasNeedScaling(true));
//   };

//   const handleToggleGallery = () => {
//     shouldShowGallery ? handleCloseGallery() : handleOpenGallery();
//   };

//   const handleOpenGallery = () => {
//     dispatch(setShouldShowGallery(true));
//     shouldPinGallery && dispatch(setDoesCanvasNeedScaling(true));
//   };

//   const handleCloseGallery = useCallback(() => {
//     dispatch(setShouldShowGallery(false));
//     dispatch(setShouldHoldGalleryOpen(false));
//     setTimeout(
//       () => shouldPinGallery && dispatch(setDoesCanvasNeedScaling(true)),
//       400
//     );
//   }, [dispatch, shouldPinGallery]);

//   const handleClickLoadMore = () => {
//     dispatch(requestImages(currentCategory));
//   };

//   const handleChangeGalleryImageMinimumWidth = (v: number) => {
//     dispatch(setGalleryImageMinimumWidth(v));
//   };

//   const setCloseGalleryTimer = () => {
//     if (shouldHoldGalleryOpen) return;
//     timeoutIdRef.current = window.setTimeout(() => handleCloseGallery(), 500);
//   };

//   const cancelCloseGalleryTimer = () => {
//     timeoutIdRef.current && window.clearTimeout(timeoutIdRef.current);
//   };

//   useHotkeys(
//     'g',
//     () => {
//       handleToggleGallery();
//     },
//     [shouldShowGallery, shouldPinGallery]
//   );

//   useHotkeys(
//     'left',
//     () => {
//       dispatch(selectPrevImage());
//     },
//     {
//       enabled: !isStaging || activeTabName !== 'unifiedCanvas',
//     },
//     [isStaging]
//   );

//   useHotkeys(
//     'right',
//     () => {
//       dispatch(selectNextImage());
//     },
//     {
//       enabled: !isStaging || activeTabName !== 'unifiedCanvas',
//     },
//     [isStaging]
//   );

//   useHotkeys(
//     'shift+g',
//     () => {
//       handleSetShouldPinGallery();
//     },
//     [shouldPinGallery]
//   );

//   useHotkeys(
//     'esc',
//     () => {
//       dispatch(setShouldShowGallery(false));
//     },
//     {
//       enabled: () => !shouldPinGallery,
//       preventDefault: true,
//     },
//     [shouldPinGallery]
//   );

//   const IMAGE_SIZE_STEP = 32;

//   useHotkeys(
//     'shift+up',
//     () => {
//       if (galleryImageMinimumWidth < 256) {
//         const newMinWidth = clamp(
//           galleryImageMinimumWidth + IMAGE_SIZE_STEP,
//           32,
//           256
//         );
//         dispatch(setGalleryImageMinimumWidth(newMinWidth));
//       }
//     },
//     [galleryImageMinimumWidth]
//   );

//   useHotkeys(
//     'shift+down',
//     () => {
//       if (galleryImageMinimumWidth > 32) {
//         const newMinWidth = clamp(
//           galleryImageMinimumWidth - IMAGE_SIZE_STEP,
//           32,
//           256
//         );
//         dispatch(setGalleryImageMinimumWidth(newMinWidth));
//       }
//     },
//     [galleryImageMinimumWidth]
//   );

//   useEffect(() => {
//     function handleClickOutside(e: MouseEvent) {
//       if (
//         !shouldPinGallery &&
//         galleryRef.current &&
//         !galleryRef.current.contains(e.target as Node)
//       ) {
//         handleCloseGallery();
//       }
//     }
//     document.addEventListener('mousedown', handleClickOutside);
//     return () => {
//       document.removeEventListener('mousedown', handleClickOutside);
//     };
//   }, [handleCloseGallery, shouldPinGallery]);

//   return (
//     <CSSTransition
//       nodeRef={galleryRef}
//       in={shouldShowGallery || shouldHoldGalleryOpen}
//       unmountOnExit
//       timeout={200}
//       classNames={`${direction}-image-gallery-css-transition`}
//     >
//       <Box
//         className={`${direction}-image-gallery-css-transition`}
//         sx={
//           shouldPinGallery
//             ? { zIndex: 1, insetInlineEnd: 0 }
//             : {
//                 zIndex: 100,
//                 position: 'fixed',
//                 height: '100vh',
//                 top: 0,
//                 insetInlineEnd: 0,
//               }
//         }
//         ref={galleryRef}
//         onMouseLeave={!shouldPinGallery ? setCloseGalleryTimer : undefined}
//         onMouseEnter={!shouldPinGallery ? cancelCloseGalleryTimer : undefined}
//         onMouseOver={!shouldPinGallery ? cancelCloseGalleryTimer : undefined}
//       >
//         <ChakraResizeable
//           sx={{
//             padding: 4,
//             display: 'flex',
//             flexDirection: 'column',
//             rowGap: 4,
//             borderRadius: shouldPinGallery ? 'base' : 0,
//             borderInlineStartWidth: 5,
//             // boxShadow: '0 0 1rem blackAlpha.700',
//             bg: 'base.850',
//             borderColor: 'base.700',
//           }}
//           minWidth={galleryMinWidth}
//           maxWidth={shouldPinGallery ? galleryMaxWidth : window.innerWidth}
//           data-pinned={shouldPinGallery}
//           handleStyles={
//             direction === 'rtl'
//               ? {
//                   right: {
//                     width: '15px',
//                   },
//                 }
//               : {
//                   left: {
//                     width: '15px',
//                   },
//                 }
//           }
//           enable={
//             direction === 'rtl'
//               ? {
//                   right: shouldEnableResize,
//                 }
//               : {
//                   left: shouldEnableResize,
//                 }
//           }
//           size={{
//             width: galleryWidth,
//             height: shouldPinGallery ? '100%' : '100vh',
//           }}
//           onResizeStart={(
//             _event:
//               | React.MouseEvent<HTMLElement>
//               | React.TouchEvent<HTMLElement>,
//             _direction: Direction,
//             elementRef: HTMLElement
//           ) => {
//             setGalleryResizeHeight(elementRef.clientHeight);
//             elementRef.style.height = `${elementRef.clientHeight}px`;
//             if (shouldPinGallery) {
//               elementRef.style.position = 'fixed';
//               elementRef.style.insetInlineEnd = '1rem';
//               setIsResizing(true);
//             }
//           }}
//           onResizeStop={(
//             _event: MouseEvent | TouchEvent,
//             _direction: Direction,
//             elementRef: HTMLElement,
//             delta: NumberSize
//           ) => {
//             const newWidth = shouldPinGallery
//               ? clamp(
//                   Number(galleryWidth) + delta.width,
//                   galleryMinWidth,
//                   Number(galleryMaxWidth)
//                 )
//               : Number(galleryWidth) + delta.width;
//             dispatch(setGalleryWidth(newWidth));

//             elementRef.removeAttribute('data-resize-alert');

//             if (shouldPinGallery) {
//               console.log('unpin');
//               elementRef.style.position = 'relative';
//               elementRef.style.removeProperty('inset-inline-end');
//               elementRef.style.setProperty(
//                 'height',
//                 shouldPinGallery ? '100%' : '100vh'
//               );
//               setIsResizing(false);
//               dispatch(setDoesCanvasNeedScaling(true));
//             }
//           }}
//           onResize={(
//             _event: MouseEvent | TouchEvent,
//             _direction: Direction,
//             elementRef: HTMLElement,
//             delta: NumberSize
//           ) => {
//             const newWidth = clamp(
//               Number(galleryWidth) + delta.width,
//               galleryMinWidth,
//               Number(
//                 shouldPinGallery ? galleryMaxWidth : 0.95 * window.innerWidth
//               )
//             );

//             if (
//               newWidth >= GALLERY_SHOW_BUTTONS_MIN_WIDTH &&
//               !shouldShowButtons
//             ) {
//               setShouldShowButtons(true);
//             } else if (
//               newWidth < GALLERY_SHOW_BUTTONS_MIN_WIDTH &&
//               shouldShowButtons
//             ) {
//               setShouldShowButtons(false);
//             }

//             if (
//               galleryImageMinimumWidth >
//               newWidth - GALLERY_IMAGE_WIDTH_OFFSET
//             ) {
//               dispatch(
//                 setGalleryImageMinimumWidth(
//                   newWidth - GALLERY_IMAGE_WIDTH_OFFSET
//                 )
//               );
//             }

//             if (shouldPinGallery) {
//               if (newWidth >= galleryMaxWidth) {
//                 elementRef.setAttribute('data-resize-alert', 'true');
//               } else {
//                 elementRef.removeAttribute('data-resize-alert');
//               }
//             }

//             elementRef.style.height = `${galleryResizeHeight}px`;
//           }}
//         >
//           <ImageGalleryContent />
//         </ChakraResizeable>
//         {isResizing && (
//           <Box
//             style={{
//               width: `${galleryWidth}px`,
//               height: '100%',
//             }}
//           />
//         )}
//       </Box>
//     </CSSTransition>
//   );
// }

export default {};
