import { Flex, Kbd, Spacer, Text } from '@invoke-ai/ui-library';
import type { Hotkey } from 'features/system/components/HotkeysModal/useHotkeyData';
import { Fragment, memo } from 'react';
import { useTranslation } from 'react-i18next';

interface Props {
  hotkey: Hotkey;
}

const HotkeyListItem = ({ hotkey }: Props) => {
  const { t } = useTranslation();
  const { id, platformKeys, title, desc } = hotkey;
  return (
    <Flex flexDir="column" gap={2} px={2}>
      <Flex lineHeight={1} gap={1} alignItems="center">
        <Text fontWeight="semibold">{title}</Text>
        <Spacer />
        {platformKeys.map((hotkey, i1) => {
          return (
            <Fragment key={`${id}-${i1}`}>
              {hotkey.map((key, i2) => (
                <Fragment key={`${id}-${key}-${i1}-${i2}`}>
                  <Kbd textTransform="lowercase">{key}</Kbd>
                  {i2 !== hotkey.length - 1 && (
                    <Text as="span" fontWeight="semibold">
                      +
                    </Text>
                  )}
                </Fragment>
              ))}
              {i1 !== platformKeys.length - 1 && (
                <Text as="span" px={2} variant="subtext" fontWeight="semibold">
                  {t('common.or')}
                </Text>
              )}
            </Fragment>
          );
        })}
      </Flex>
      <Text variant="subtext">{desc}</Text>
    </Flex>
  );
};

export default memo(HotkeyListItem);
