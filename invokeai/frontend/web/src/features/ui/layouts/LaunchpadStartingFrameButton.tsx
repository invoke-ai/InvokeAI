import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { LaunchpadButton } from 'features/ui/layouts/LaunchpadButton';
import { startingFrameImageChanged } from 'features/parameters/store/videoSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

export const LaunchpadStartingFrameButton = memo(() => {
	const { t } = useTranslation();
	const dispatch = useAppDispatch();

	const onUpload = useCallback((imageDTO: ImageDTO) => {
		dispatch(startingFrameImageChanged(imageDTOToImageWithDims(imageDTO)));
	}, [dispatch]);

	const uploadApi = useImageUploadButton({ allowMultiple: false, onUpload });

	return (
		<LaunchpadButton {...uploadApi.getUploadButtonProps()} position="relative" gap={8}>
			<Icon as={PiUploadBold} boxSize={8} color="base.500" />
			<Flex flexDir="column" alignItems="flex-start" gap={2}>
				<Heading size="sm">{t('ui.launchpad.addStartingFrame.title')}</Heading>
				<Text>{t('ui.launchpad.addStartingFrame.description')}</Text>
			</Flex>
			<Flex position="absolute" right={3} bottom={3}>
				<PiUploadBold />
				<input {...uploadApi.getUploadInputProps()} />
			</Flex>
		</LaunchpadButton>
	);
});

LaunchpadStartingFrameButton.displayName = 'LaunchpadStartingFrameButton';