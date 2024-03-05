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
  useDisclosure,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import type { ChangeEvent, ReactElement } from 'react';
import { cloneElement, createRef, memo, useCallback, useEffect, useMemo, useState } from 'react';
import type { ReactCropperElement } from 'react-cropper';
import Cropper from 'react-cropper';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

import { resizeBase64Image } from './cropperUtils';

type ImageCropperProps = {
  imageDTO: ImageDTO | undefined;
  children: ReactElement;
};

export const ImageCropper = (props: ImageCropperProps) => {
  const { data: imageDTO } = useGetImageDTOQuery(props.imageDTO?.image_name ?? skipToken);
  const cropperRef = createRef<ReactCropperElement>();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();

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

  const onCustomSaveSizeToggle = useCallback(() => {
    if (isAccordionOpen) {
      setIsCustomCropSizeEnabled(true);
    } else {
      setIsCustomCropSizeEnabled(false);
    }
    onToggle();
  }, [isAccordionOpen, onToggle]);

  // Handlers

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

  const handleCustomCropSizeMultiplierChange = useCallback((multiplier: number) => {
    setCustomCropSizeMultiplier(multiplier);
  }, []);

  const handleCustomCropSizeMultiplierEnabledChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setIsCustomCropSizeMultiplierEnabled(e.target.checked);
  }, []);

  // Cropper Handlers

  const onCropperOpen = useCallback(() => {
    onOpen();
  }, [onOpen]);

  const onCropperClose = useCallback(() => {
    setCropData(null);
    onClose();
  }, [onClose]);

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

    onOpen();
  }, [cropperRef, onOpen]);

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
    <>
      {cloneElement(props.children, {
        onClick: onCropperOpen,
      })}
      <Modal isOpen={isOpen} onClose={onCropperClose} isCentered>
        <ModalOverlay />
        <ModalContent w="80vw" h="85vh" maxW="unset" maxH="unset">
          <ModalHeader>{t('controlnet.crop')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody as={Flex} gap={4} w="full" h="full" pb={4}>
            {imageDTO && (
              <Flex w="30%" gap={4} flexDir="column">
                <FormControl>
                  <FormLabel>{t('cropper.cropBoxWidth')}</FormLabel>
                  <CompositeSlider
                    value={cropBoxSize.width ? cropBoxSize.width : 512}
                    onChange={handleCropBoxWidthChange}
                    defaultValue={512}
                    min={0}
                    max={containerSize.width}
                    step={1}
                    fineStep={8}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>{t('cropper.cropBoxHeight')}</FormLabel>
                  <CompositeSlider
                    value={cropBoxSize.height ? cropBoxSize.height : 512}
                    onChange={handleCropBoxHeightChange}
                    defaultValue={512}
                    min={0}
                    max={containerSize.height}
                    step={1}
                    fineStep={8}
                  />
                </FormControl>

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
                          min={0}
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
                          min={0}
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
                          min={0}
                          max={4096}
                          step={1}
                          defaultValue={512}
                          isDisabled={!isCustomCropSizeEnabled && isCustomCropSizeMultiplierEnabled}
                        />
                      </FormControl>
                    </Flex>
                  </Flex>
                </StandaloneAccordion>

                <Button onClick={getCropData}>{t('cropper.preview')}</Button>
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
              <Text position="absolute" padding={2} background="black">
                {cropBoxSize.width}X{cropBoxSize.height}
              </Text>
            </Flex>
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(ImageCropper);
