import './cropper.min.css';

import {
  Button,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import type { ReactElement } from 'react';
import { cloneElement, createRef, memo, useCallback, useEffect, useState } from 'react';
import type { ReactCropperElement } from 'react-cropper';
import Cropper from 'react-cropper';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

type ImageCropperProps = {
  imageDTO: ImageDTO | undefined;
  children: ReactElement;
};

export const ImageCropper = (props: ImageCropperProps) => {
  const { data: imageDTO } = useGetImageDTOQuery(props.imageDTO?.image_name ?? skipToken);
  const [cropData, setCropData] = useState<string | null>(null);
  const cropperRef = createRef<ReactCropperElement>();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const [cropBoxSize, setCropBoxSize] = useState<{ width: number; height: number }>({
    width: 512,
    height: 512,
  });

  const [containerSize, setContainerSize] = useState<{ width: number; height: number }>({
    width: 512,
    height: 512,
  });

  const getCropData = useCallback(() => {
    if (typeof cropperRef.current?.cropper !== 'undefined') {
      setCropData(
        cropperRef.current?.cropper
          .getCroppedCanvas({ width: cropBoxSize.width, height: cropBoxSize.height })
          .toDataURL()
      );
    }
  }, [cropperRef, cropBoxSize]);

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
                <Button onClick={getCropData}>{t('cropper.preview')}</Button>
                <Flex flexDir="column" w="full" h="full" gap={2}>
                  <Text>{t('cropper.preview')}</Text>
                  <Flex w="full" h="full" alignItems="center" justifyContent="center" borderRadius="base" bg="base.850">
                    {cropData ? <img src={cropData} alt="cropped" /> : <Text>{t('cropper.noPreview')}</Text>}
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
                viewMode={3}
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
