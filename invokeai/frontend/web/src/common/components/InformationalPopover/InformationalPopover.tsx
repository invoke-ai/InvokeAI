import {
  Button,
  Divider,
  Flex,
  Heading,
  Image,
  Popover,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectSystemSlice, setShouldEnableInformationalPopovers } from 'features/system/store/systemSlice';
import { toast } from 'features/toast/toast';
import { merge, omit } from 'lodash-es';
import type { ReactElement } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

import type { Feature, PopoverData } from './constants';
import { OPEN_DELAY, POPOVER_DATA, POPPER_MODIFIERS } from './constants';

type Props = {
  feature: Feature;
  inPortal?: boolean;
  children: ReactElement;
};

const selectShouldEnableInformationalPopovers = createSelector(selectSystemSlice, system => system.shouldEnableInformationalPopovers);

export const InformationalPopover = memo(({ feature, children, inPortal = true, ...rest }: Props) => {
  const shouldEnableInformationalPopovers = useAppSelector(selectShouldEnableInformationalPopovers);

  const data = useMemo(() => POPOVER_DATA[feature], [feature]);

  const popoverProps = useMemo(() => merge(omit(data, ['image', 'href', 'buttonLabel']), rest), [data, rest]);

  if (!shouldEnableInformationalPopovers) {
    return children;
  }

  return (
    <Popover
      isLazy
      closeOnBlur={false}
      trigger="hover"
      variant="informational"
      openDelay={OPEN_DELAY}
      modifiers={POPPER_MODIFIERS}
      placement="top"
      {...popoverProps}
    >
      <PopoverTrigger>{children}</PopoverTrigger>
      {inPortal ? (
        <Portal>
          <Content data={data} feature={feature} />
        </Portal>
      ) : (
        <Content data={data} feature={feature} />
      )}
    </Popover>
  );
});

InformationalPopover.displayName = 'InformationalPopover';

type ContentProps = {
  data?: PopoverData;
  feature: Feature;
};

const Content = ({ data, feature }: ContentProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const heading = useMemo<string | undefined>(() => t(`popovers.${feature}.heading`), [feature, t]);

  const paragraphs = useMemo<string[]>(
    () =>
      t(`popovers.${feature}.paragraphs`, {
        returnObjects: true,
      }) ?? [],
    [feature, t]
  );

  const onClickLearnMore = useCallback(() => {
    if (!data?.href) {
      return;
    }
    window.open(data.href);
  }, [data?.href]);

  const onClickDontShowMeThese = useCallback(() => {
    dispatch(setShouldEnableInformationalPopovers(false));
    toast({
      title: t('settings.informationalPopoversDisabled'),
      description: t('settings.informationalPopoversDisabledDesc'),
      status: 'info',
    });
  }, [dispatch, t]);

  return (
    <PopoverContent maxW={300}>
      <PopoverCloseButton top={2} />
      <PopoverBody>
        <Flex gap={2} flexDirection="column" alignItems="flex-start">
          {heading && (
            <>
              <Heading size="sm">{heading}</Heading>
              <Divider />
            </>
          )}
          {data?.image && (
            <>
              <Image
                objectFit="contain"
                maxW="60%"
                maxH="60%"
                backgroundColor="white"
                src={data.image}
                alt="Optional Image"
              />
              <Divider />
            </>
          )}
          {paragraphs.map((p) => (
            <Text key={p}>{p}</Text>
          ))}

          <Divider />
          <Flex alignItems="center" justifyContent="space-between" w="full">
            <Button onClick={onClickDontShowMeThese} variant="link" size="sm">
              {t('common.dontShowMeThese')}
            </Button>
            <Spacer />
            {data?.href && (
              <Button onClick={onClickLearnMore} leftIcon={<PiArrowSquareOutBold />} variant="link" size="sm">
                {t('common.learnMore') ?? heading}
              </Button>
            )}
          </Flex>
        </Flex>
      </PopoverBody>
    </PopoverContent>
  );
};
