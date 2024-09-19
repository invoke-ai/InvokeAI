import { Box,Flex, IconButton } from '@invoke-ai/ui-library';
import { PiLightbulbFilamentBold } from 'react-icons/pi';

export const NotificationIcon = ({ showIndicator }: { showIndicator: boolean }) => {
  return (
    <Flex pos="relative">
      <IconButton
        aria-label="Notifications"
        variant="link"
        icon={<PiLightbulbFilamentBold fontSize={20} />}
        boxSize={8}
      />
      {showIndicator && (
        <Box pos="absolute" top={0} right="2px" w={2} h={2} backgroundColor="invokeYellow.500" borderRadius="100%" />
      )}
    </Flex>
  );
};
