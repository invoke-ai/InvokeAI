import {
  Flex,
  IconButton,
  Image,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverHeader,
  PopoverTrigger,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { $didStudioInit } from 'app/hooks/useStudioInitAction';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { shouldShowNotificationChanged } from 'features/ui/store/uiSlice';
import InvokeSymbol from 'public/assets/images/invoke-favicon.png';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightbulbFilamentBold } from 'react-icons/pi';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

import { WhatsNew } from './WhatsNew';

const selectIsLocal = createSelector(selectConfigSlice, (config) => config.isLocal);

export const Notifications = () => {
  const { t } = useTranslation();
  const didStudioInit = useStore($didStudioInit);
  const dispatch = useAppDispatch();
  const shouldShowNotification = useAppSelector((s) => s.ui.shouldShowNotificationV2);
  const resetIndicator = useCallback(() => {
    dispatch(shouldShowNotificationChanged(false));
  }, [dispatch]);
  const { data } = useGetAppVersionQuery();
  const isLocal = useAppSelector(selectIsLocal);

  if (!data || !didStudioInit) {
    return null;
  }

  return (
    <Popover onClose={resetIndicator} placement="top-start" autoFocus={false} defaultIsOpen={shouldShowNotification}>
      <PopoverTrigger>
        <Flex pos="relative">
          <IconButton
            aria-label="Notifications"
            variant="link"
            icon={<PiLightbulbFilamentBold fontSize={20} />}
            boxSize={8}
          />
        </Flex>
      </PopoverTrigger>
      <PopoverContent p={2}>
        <PopoverArrow />
        <PopoverCloseButton />
        <PopoverHeader fontSize="md" fontWeight="semibold" pt={5}>
          <Flex alignItems="center" gap={3}>
            <Image src={InvokeSymbol} boxSize={6} />
            {t('whatsNew.whatsNewInInvoke')}
            {!!data.version.length &&
              (isLocal ? (
                <Text variant="subtext">{`v${data.version}`}</Text>
              ) : (
                <Text variant="subtext">{data.version}</Text>
              ))}
          </Flex>
        </PopoverHeader>
        <PopoverBody p={2} maxW={300}>
          <WhatsNew />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
