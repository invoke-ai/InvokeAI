import { Flex, Kbd, Spacer, Text } from '@invoke-ai/ui-library';
import { Fragment, memo } from 'react';
import { useTranslation } from 'react-i18next';

interface HotkeysModalProps {
  hotkeys: string[][];
  title: string;
  description: string;
}

const HotkeyListItem = (props: HotkeysModalProps) => {
  const { t } = useTranslation();
  const { title, hotkeys, description } = props;
  return (
    <Flex flexDir="column" gap={2} px={2}>
      <Flex lineHeight={1} gap={1} alignItems="center">
        <Text fontWeight="semibold">{title}</Text>
        <Spacer />
        {hotkeys.map((hotkey, index) => {
          return (
            <Fragment key={`${hotkey}-${index}`}>
              {hotkey.map((key, index) => (
                <>
                  <Kbd textTransform="lowercase" key={`${hotkey}-${key}-${index}`}>
                    {key}
                  </Kbd>
                  {index !== hotkey.length - 1 && (
                    <Text as="span" fontWeight="semibold">
                      +
                    </Text>
                  )}
                </>
              ))}
              {index !== hotkeys.length - 1 && (
                <Text as="span" px={2} variant="subtext" fontWeight="semibold">
                  {t('common.or')}
                </Text>
              )}
            </Fragment>
          );
        })}
      </Flex>
      <Text variant="subtext">{description}</Text>
    </Flex>
  );
};

export default memo(HotkeyListItem);
