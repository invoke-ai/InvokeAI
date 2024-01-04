import { Flex, Kbd, Spacer } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
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
        <InvText fontWeight="semibold">{title}</InvText>
        <Spacer />
        {hotkeys.map((hotkey, index) => {
          return (
            <Fragment key={`${hotkey}-${index}`}>
              {hotkey.map((key, index) => (
                <>
                  <Kbd
                    textTransform="lowercase"
                    key={`${hotkey}-${key}-${index}`}
                  >
                    {key}
                  </Kbd>
                  {index !== hotkey.length - 1 && (
                    <InvText as="span" fontWeight="semibold">
                      +
                    </InvText>
                  )}
                </>
              ))}
              {index !== hotkeys.length - 1 && (
                <InvText
                  as="span"
                  px={2}
                  variant="subtext"
                  fontWeight="semibold"
                >
                  {t('common.or')}
                </InvText>
              )}
            </Fragment>
          );
        })}
      </Flex>
      <InvText variant="subtext">{description}</InvText>
    </Flex>
  );
};

export default memo(HotkeyListItem);
