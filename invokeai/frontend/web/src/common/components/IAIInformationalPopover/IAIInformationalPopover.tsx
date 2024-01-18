import { Divider, Flex, Image, Portal } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvHeading } from 'common/components/InvHeading/wrapper';
import {
  InvPopover,
  InvPopoverBody,
  InvPopoverCloseButton,
  InvPopoverContent,
  InvPopoverTrigger,
} from 'common/components/InvPopover/wrapper';
import { InvText } from 'common/components/InvText/wrapper';
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
    <InvPopover
      isLazy
      closeOnBlur={false}
      trigger="hover"
      variant="informational"
      openDelay={OPEN_DELAY}
      modifiers={POPPER_MODIFIERS}
      placement="top"
      {...popoverProps}
    >
      <InvPopoverTrigger>{children}</InvPopoverTrigger>
      {inPortal ? (
        <Portal>
          <PopoverContent data={data} feature={feature} />
        </Portal>
      ) : (
        <PopoverContent data={data} feature={feature} />
      )}
    </InvPopover>
  );
};

export default memo(IAIInformationalPopover);

type PopoverContentProps = {
  data?: PopoverData;
  feature: Feature;
};

const PopoverContent = ({ data, feature }: PopoverContentProps) => {
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
    <InvPopoverContent w={96}>
      <InvPopoverCloseButton />
      <InvPopoverBody>
        <Flex gap={2} flexDirection="column" alignItems="flex-start">
          {heading && (
            <>
              <InvHeading size="sm">{heading}</InvHeading>
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
            <InvText key={p}>{p}</InvText>
          ))}
          {data?.href && (
            <>
              <Divider />
              <InvButton
                pt={1}
                onClick={handleClick}
                leftIcon={<PiArrowSquareOutBold />}
                alignSelf="flex-end"
                variant="link"
              >
                {t('common.learnMore') ?? heading}
              </InvButton>
            </>
          )}
        </Flex>
      </InvPopoverBody>
    </InvPopoverContent>
  );
};
