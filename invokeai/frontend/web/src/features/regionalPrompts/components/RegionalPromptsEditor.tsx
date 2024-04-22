/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { RegionalPromptsToolbar } from 'features/regionalPrompts/components/RegionalPromptsToolbar';
import { StageComponent } from 'features/regionalPrompts/components/StageComponent';
import { memo } from 'react';

export const RegionalPromptsEditor = memo(() => {
  return (
    <Flex
      position="relative"
      flexDirection="column"
      height="100%"
      width="100%"
      rowGap={4}
      alignItems="center"
      justifyContent="center"
    >
      <RegionalPromptsToolbar />
      <StageComponent />
    </Flex>
  );
});

RegionalPromptsEditor.displayName = 'RegionalPromptsEditor';
