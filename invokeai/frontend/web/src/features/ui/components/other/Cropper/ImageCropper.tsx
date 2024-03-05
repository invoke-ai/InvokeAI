import './cropper.min.css';

import {
  Button,
  Checkbox,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  Image,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  StandaloneAccordion,
  Text,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import type { ChangeEvent } from 'react';
import { createRef, memo, useCallback, useEffect, useMemo, useState } from 'react';
import type { ReactCropperElement } from 'react-cropper';
import Cropper from 'react-cropper';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

import { resizeBase64Image } from './cropperUtils';
import { isCropperModalOpenChanged } from './store/slice';

const ImageCropper = () => {
  // Global State
  const imageToCrop = useAppSelector((state) => state.cropper.imageToCrop);
  const isCropperModalOpen = useAppSelector((state) => state.cropper.isCropperModalOpen);

  const { data: imageDTO } = useGetImageDTOQuery(imageToCrop?.image_name ?? skipToken);

  const cropperRef = createRef<ReactCropperElement>();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  // Canvas State
  const [cropData, setCropData] = useState<string | null>(null);
  const [cropBoxSize, setCropBoxSize] = useState<{ width: number; height: number }>({
    width: 512,
    height: 512,
  });
  const [containerSize, setContainerSize] = useState<{ width: number; height: number }>({
    width: 512,
    height: 512,
  });

  // Custom Crop Size State
  const [isCustomCropSizeEnabled, setIsCustomCropSizeEnabled] = useState<boolean>(false);
  const [customCropSize, setCustomCropSize] = useState<{ width: number; height: number }>({
    width: 512,
    height: 512,
  });
  const [isCustomCropSizeMultiplierEnabled, setIsCustomCropSizeMultiplierEnabled] = useState<boolean>(true);
  const [customCropSizeMultiplier, setCustomCropSizeMultiplier] = useState<number>(2);

  const cropSizeDerived = useMemo(
    () => ({
      width: !isCustomCropSizeEnabled
        ? isCustomCropSizeMultiplierEnabled
          ? Math.floor(cropBoxSize.width * customCropSizeMultiplier)
          : customCropSize.width
        : cropBoxSize.width,
      height: !isCustomCropSizeEnabled
        ? isCustomCropSizeMultiplierEnabled
          ? Math.floor(cropBoxSize.height * customCropSizeMultiplier)
          : customCropSize.height
        : cropBoxSize.height,
    }),
    [cropBoxSize, customCropSize, customCropSizeMultiplier, isCustomCropSizeMultiplierEnabled, isCustomCropSizeEnabled]
  );

  // Hooks
  const { isOpen: isAccordionOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'cropper-custom-save-size',
    defaultIsOpen: false,
  });

  // Canvas Handlers
  const getCropData = useCallback(() => {
    if (typeof cropperRef.current?.cropper !== 'undefined') {
      const croppedCanvas = cropperRef.current?.cropper
        .getCroppedCanvas({
          width: cropBoxSize.width,
          height: cropBoxSize.height,
        })
        .toDataURL();

      resizeBase64Image(croppedCanvas, cropSizeDerived.width, cropSizeDerived.height).then((finalCrop) => {
        setCropData(finalCrop as string);
      });
    }
  }, [cropperRef, cropSizeDerived, cropBoxSize]);

  // Crop Box Size Handlers
  const handleCropBoxWidthChange = useCallback(
    (width: number) => {
      setCropBoxSize({ ...cropBoxSize, width: width });
      cropperRef.current?.cropper.setCropBoxData({ ...cropperRef.current.cropper.getCropBoxData(), width: width });
    },
    [cropBoxSize, cropperRef]
  );

  const handleCropBoxHeightChange = useCallback(
    (height: number) => {
      setCropBoxSize({ ...cropBoxSize, height: height });
      cropperRef.current?.cropper.setCropBoxData({ ...cropperRef.current.cropper.getCropBoxData(), height: height });
    },
    [cropBoxSize, cropperRef]
  );

  // Custom Crop Size Handlers
  const onCustomSaveSizeToggle = useCallback(() => {
    if (isAccordionOpen) {
      setIsCustomCropSizeEnabled(true);
    } else {
      setIsCustomCropSizeEnabled(false);
    }
    onToggle();
  }, [isAccordionOpen, onToggle]);

  const handleCustomCropSizeMultiplierEnabledChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setIsCustomCropSizeMultiplierEnabled(e.target.checked);
  }, []);

  const handleCustomCropSizeMultiplierChange = useCallback((multiplier: number) => {
    setCustomCropSizeMultiplier(multiplier);
  }, []);

  const handleCustomCropSizeWidthChange = useCallback(
    (width: number) => {
      setCustomCropSize({ ...customCropSize, width: width });
    },
    [customCropSize]
  );

  const handleCustomCropSizeHeightChange = useCallback(
    (height: number) => {
      setCustomCropSize({ ...customCropSize, height: height });
    },
    [customCropSize]
  );

  // Cropper Handlers
  const onCropperClose = useCallback(() => {
    setCropData(null);
    dispatch(isCropperModalOpenChanged(false));
  }, [dispatch]);

  const handleCropperInitialization = useCallback(() => {
    const cropper = cropperRef.current?.cropper;

    if (!cropper) {
      // Wait for the cropper to be ready
      setTimeout(() => handleCropperInitialization(), 100);
      return;
    }

    cropper.setCropBoxData({
      ...cropper.getCropBoxData(),
      width: 512,
      height: 512,
    });

    setCropBoxSize({
      width: cropper.getCropBoxData().width,
      height: cropper.getCropBoxData().height,
    });

    setContainerSize({
      width: cropper.getContainerData().width,
      height: cropper.getContainerData().height,
    });
  }, [cropperRef]);

  useEffect(() => {
    if (!cropperRef.current?.cropper) {
      handleCropperInitialization();
    }
  }, [handleCropperInitialization, cropperRef]);

  const handleCropEnd = useCallback(() => {
    if (cropperRef.current) {
      setCropBoxSize({
        width: Math.floor(cropperRef.current.cropper.getCropBoxData().width),
        height: Math.floor(cropperRef.current.cropper.getCropBoxData().height),
      });
    }
  }, [cropperRef]);

  // Badges
  const customCropSizeBadges: (string | number)[] = [`${cropSizeDerived.width}x${cropSizeDerived.height}`];

  return (
    <Modal isOpen={isCropperModalOpen} onClose={onCropperClose} isCentered>
      <ModalOverlay />
      <ModalContent w="80vw" h="85vh" maxW="unset" maxH="unset">
        <ModalHeader>{t('controlnet.crop')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody as={Flex} gap={4} w="full" h="full" pb={4}>
          {imageDTO && (
            <Flex w="30%" gap={4} flexDir="column">
              {/* Crop Box Size */}
              <Flex flexDir="column" gap={2}>
                <FormControl>
                  <FormLabel>{t('cropper.cropBoxWidth')}</FormLabel>
                  <CompositeSlider
                    value={cropBoxSize.width ? cropBoxSize.width : 512}
                    onChange={handleCropBoxWidthChange}
                    defaultValue={512}
                    min={1}
                    max={containerSize.width}
                    step={8}
                    fineStep={1}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>{t('cropper.cropBoxHeight')}</FormLabel>
                  <CompositeSlider
                    value={cropBoxSize.height ? cropBoxSize.height : 512}
                    onChange={handleCropBoxHeightChange}
                    defaultValue={512}
                    min={1}
                    max={containerSize.height}
                    step={8}
                    fineStep={1}
                  />
                </FormControl>
              </Flex>

              {/* Custom Save Size Accordion */}
              <StandaloneAccordion
                label={t('cropper.customSaveSize.title')}
                isOpen={isAccordionOpen}
                badges={customCropSizeBadges}
                onToggle={onCustomSaveSizeToggle}
              >
                <Flex p={4} gap={4} flexDir="column" background="base.750" borderRadius="base" borderTopRadius={0}>
                  <Flex gap={4}>
                    <FormControl sx={{ width: 'max-content' }}>
                      <Checkbox
                        isChecked={isCustomCropSizeMultiplierEnabled}
                        onChange={handleCustomCropSizeMultiplierEnabledChange}
                      />
                    </FormControl>
                    <FormControl>
                      <FormLabel>{t('cropper.customSaveSize.scaleBy')}</FormLabel>
                      <CompositeSlider
                        value={customCropSizeMultiplier}
                        onChange={handleCustomCropSizeMultiplierChange}
                        defaultValue={1}
                        min={0.01}
                        max={20}
                        step={1}
                        fineStep={0.01}
                        marks={[0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
                        isDisabled={!isCustomCropSizeMultiplierEnabled}
                      />
                    </FormControl>
                  </Flex>
                  <Flex gap={4}>
                    <FormControl>
                      <FormLabel>{t('cropper.width')}</FormLabel>
                      <CompositeNumberInput
                        value={customCropSize.width}
                        onChange={handleCustomCropSizeWidthChange}
                        min={1}
                        max={4096}
                        step={1}
                        defaultValue={512}
                        isDisabled={!isCustomCropSizeEnabled && isCustomCropSizeMultiplierEnabled}
                      />
                    </FormControl>
                    <FormControl>
                      <FormLabel>{t('cropper.height')}</FormLabel>
                      <CompositeNumberInput
                        value={customCropSize.height}
                        onChange={handleCustomCropSizeHeightChange}
                        min={1}
                        max={4096}
                        step={1}
                        defaultValue={512}
                        isDisabled={!isCustomCropSizeEnabled && isCustomCropSizeMultiplierEnabled}
                      />
                    </FormControl>
                  </Flex>
                </Flex>
              </StandaloneAccordion>

              <Flex w="full" gap={2}>
                <Button w="full" onClick={getCropData}>
                  {t('cropper.preview')}
                </Button>
              </Flex>

              {/* Crop Preview */}
              <Flex flexDir="column" gap={2} width="full" height="full" position="relative">
                <Text>{t('cropper.preview')}</Text>
                <Flex
                  w="full"
                  h="full"
                  position="absolute"
                  alignItems="center"
                  justifyContent="center"
                  borderRadius="base"
                  bg="base.850"
                >
                  {cropData ? (
                    <Image
                      src={cropData}
                      w={imageDTO.width}
                      objectFit="contain"
                      maxW="full"
                      maxH="full"
                      borderRadius="base"
                    />
                  ) : (
                    <Text>{t('cropper.noPreview')}</Text>
                  )}
                </Flex>
              </Flex>
            </Flex>
          )}

          {/* Cropper Module */}
          <Flex gap={4} w="full" h="full" position="relative">
            <Cropper
              ref={cropperRef}
              style={{ height: '100%', width: '100%', overflow: 'hidden' }}
              zoomTo={1}
              autoCropArea={1}
              src={imageDTO?.image_url}
              viewMode={0}
              dragMode="move"
              responsive={true}
              restore={true}
              checkOrientation={false}
              guides={true}
              modal={true}
              center={true}
              background={true}
              onInitialized={handleCropperInitialization}
              cropend={handleCropEnd}
            />
            <Flex position="absolute" top={2} left={2} gap={2}>
              <Text padding={2} background="base.800" borderRadius="base">
                W {cropBoxSize.width}
              </Text>
              <Text padding={2} background="base.800" borderRadius="base">
                H {cropBoxSize.height}
              </Text>
            </Flex>
          </Flex>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};

export default memo(ImageCropper);
