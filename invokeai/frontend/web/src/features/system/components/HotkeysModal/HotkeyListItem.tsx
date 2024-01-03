/* eslint-disable i18next/no-literal-string */
import { Flex, Kbd, Spacer } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';
import { memo } from 'react';

interface HotkeysModalProps {
  hotkeys: string[][];
  title: string;
  description: string;
}

const HotkeyListItem = (props: HotkeysModalProps) => {
  const { title, hotkeys, description } = props;
  return (
    <Flex flexDir="column" gap={2} px={2}>
      <Flex lineHeight={1} gap={1} alignItems="center">
        <InvText fontWeight="semibold">{title}</InvText>
        <Spacer />
        {hotkeys.map((hotkey, index) => {
          return (
            <>
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
                  or
                </InvText>
              )}
            </>
          );
        })}
      </Flex>
      <InvText variant="subtext">{description}</InvText>
    </Flex>
  );
};

export default memo(HotkeyListItem);
