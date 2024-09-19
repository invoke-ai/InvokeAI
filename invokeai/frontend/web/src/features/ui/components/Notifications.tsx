import {
  Box,
  Flex,
  IconButton,
  Image,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverHeader,
  PopoverTrigger,
  Text,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { shouldShowNotificationIndicatorChanged } from 'features/ui/store/uiSlice';
import InvokeSymbol from 'public/assets/images/invoke-favicon.png';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightbulbFilamentBold } from 'react-icons/pi';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

import { CanvasV2Announcement } from './CanvasV2Announcement';

export const Notifications = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const shouldShowNotificationIndicator = useAppSelector((s) => s.ui.shouldShowNotificationIndicator);
  const resetIndicator = useCallback(() => {
    dispatch(shouldShowNotificationIndicatorChanged(false));
  }, [dispatch]);
  const { data } = useGetAppVersionQuery();

  if (!data) {
    return null;
  }

  return (
    <Popover onOpen={resetIndicator} placement="top-start">
      <PopoverTrigger>
        <Flex pos="relative">
          <IconButton
            aria-label="Notifications"
            variant="link"
            icon={<PiLightbulbFilamentBold fontSize={20} />}
            boxSize={8}
          />
          {shouldShowNotificationIndicator && (
            <Box
              pos="absolute"
              top={0}
              right="2px"
              w={2}
              h={2}
              backgroundColor="invokeYellow.500"
              borderRadius="100%"
            />
          )}
        </Flex>
      </PopoverTrigger>
      <PopoverContent p={2}>
        <PopoverArrow />
        <PopoverHeader fontSize="md" fontWeight="semibold">
          <Flex alignItems="center" gap={3}>
            <Image src={InvokeSymbol} boxSize={6} />
            {t('whatsNew.whatsNewInInvoke')}
            <Text variant="subtext">{`v${data.version}`}</Text>
          </Flex>
        </PopoverHeader>
        <PopoverBody p={2}>
          <CanvasV2Announcement />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
