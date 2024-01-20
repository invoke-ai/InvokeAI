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
  Text,
} from '@invoke-ai/ui';
import { useAppSelector } from 'app/store/storeHooks';
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

const IAIInformationalPopover = ({
  feature,
  children,
  inPortal = true,
  ...rest
}: Props) => {
  const shouldEnableInformationalPopovers = useAppSelector(
    (s) => s.system.shouldEnableInformationalPopovers
  );

  const data = useMemo(() => POPOVER_DATA[feature], [feature]);

  const popoverProps = useMemo(
    () => merge(omit(data, ['image', 'href', 'buttonLabel']), rest),
    [data, rest]
  );

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
};

export default memo(IAIInformationalPopover);

type ContentProps = {
  data?: PopoverData;
  feature: Feature;
};

const Content = ({ data, feature }: ContentProps) => {
  const { t } = useTranslation();

  const heading = useMemo<string | undefined>(
    () => t(`popovers.${feature}.heading`),
    [feature, t]
  );

  const paragraphs = useMemo<string[]>(
    () =>
      t(`popovers.${feature}.paragraphs`, {
        returnObjects: true,
      }) ?? [],
    [feature, t]
  );

  const handleClick = useCallback(() => {
    if (!data?.href) {
      return;
    }
    window.open(data.href);
  }, [data?.href]);

  return (
    <PopoverContent w={96}>
      <PopoverCloseButton />
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
          {data?.href && (
            <>
              <Divider />
              <Button
                pt={1}
                onClick={handleClick}
                leftIcon={<PiArrowSquareOutBold />}
                alignSelf="flex-end"
                variant="link"
              >
                {t('common.learnMore') ?? heading}
              </Button>
            </>
          )}
        </Flex>
      </PopoverBody>
    </PopoverContent>
  );
};
